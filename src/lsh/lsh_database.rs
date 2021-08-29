use itertools::zip_eq;
use std::collections::{HashSet, HashMap};
use std::hash::{Hash, Hasher};
use std::vec::Vec;
use std::rc::Rc;
use crate::lsh::vector::Vector;
use crate::lsh::stable_hash::StableHashFunction;

pub trait Cacheable {
    fn cache_id(&self) -> u128;
}

struct CacheItem<T: Cacheable> {
    value: T,
    hash: u128
}

impl<T: Cacheable> CacheItem<T> {
    fn new(item: T) -> Self {
        let hashcode = item.cache_id();
        CacheItem {
            value: item,
            hash: hashcode
        }
    }
}

impl<T: Cacheable> Hash for CacheItem<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

impl<T: Cacheable> PartialEq for CacheItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl<T: Cacheable> Eq for CacheItem<T> {}

struct LocalitySensitiveHashTable<T> 
where
    T: Vector<DType=f32> + Cacheable,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    table: HashMap<u64, HashSet<Rc<CacheItem<T>>>>,
    hashfn: StableHashFunction<T>
}

impl<T: Vector<DType=f32>> LocalitySensitiveHashTable<T> 
where
    T: Vector<DType=f32> + Cacheable,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    fn new(dimension: usize) -> Self {
        LocalitySensitiveHashTable {
            table: HashMap::<u64, HashSet<Rc<CacheItem<T>>>>::new(),
            hashfn: StableHashFunction::<T>::new(64, dimension)
        }
    }

    fn insert(&mut self, item: Rc<CacheItem<T>>) {
        let lsh_key = self.hashfn.hash(&item.value);
        if let Some(container) = self.table.get_mut(&lsh_key) {
            container.insert(Rc::clone(&item));
        }
        else {
            let mut value = HashSet::<Rc<CacheItem<T>>>::new();
            value.insert(Rc::clone(&item));
            self.table.insert(lsh_key, value);
        }
    }

    fn query_set<'a>(&'a self, item: &T) -> Option<&'a HashSet<Rc<CacheItem<T>>>> {
        let key = &self.hashfn.hash(item);
        self.table.get(&key)
    }
}

pub struct LocalitySensitiveHashDatabase<T> 
where
    T: Vector<DType=f32> + Cacheable,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    items: HashSet<Rc<CacheItem<T>>>,
    tables: Vec<LocalitySensitiveHashTable<T>>
}

impl<T> LocalitySensitiveHashDatabase<T> 
where
    T: Vector<DType=f32> + Cacheable,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    pub fn len(&self) -> usize {
        self.items.len()    
    }

    pub fn insert(&mut self, item: T) {
        let value = Rc::new( CacheItem::new(item));
        self.items.insert(Rc::clone(&value));
        for table in self.tables.iter_mut() {
            table.insert(Rc::clone(&value));
        }
    }
    
    pub fn new(replicas: usize, dimension: usize) -> Self {
        LocalitySensitiveHashDatabase {
            items: HashSet::<Rc<CacheItem<T>>>::new(),
            tables: (0..replicas).
                map(|_| LocalitySensitiveHashTable::<T>::new(dimension)).
                collect::<Vec<LocalitySensitiveHashTable<T>>>()
        }
    }
    
    pub fn query<'a>(&'a self, item: &T) -> Option<&'a T> {
        let empty = HashSet::<Rc<CacheItem<T>>>::new();
        // We deduplicate the results returned from each replica before
        // computing Euclidian distances to find the nearest neighbour
        let candidates = self.tables.
            iter().
            flat_map(|table| {
                match table.query_set(item) {
                    None => empty.iter(),
                    Some(items) => items.iter() 
                }
            }).
            cloned().
            collect::<HashSet<Rc<CacheItem<T>>>>();
        
        let maybe_nearest_neighbour = 
            candidates.
                iter().
                map(|x| (item.distance(&x.value), x)).
                fold(None, |acc, (d, v)| {
                    match acc {
                        None => Some((d, Rc::clone(v))), 
                        Some((min, w)) => {
                            if d < min { 
                                Some((d, Rc::clone(v)))
                            }
                            else {
                                Some((min, w))
                            }
                        }
                    }
                });

        match maybe_nearest_neighbour {
            Some((_, neighbour)) => {
                if let Some(result) = self.items.get(&neighbour) {
                    Some(&result.value)
                }
                else {
                    None
                }
            },
            None => None
        }
    }
}


#[cfg(test)]
mod lsh_database_test {
    use super::*;
    use crate::simd::vec::SimdVecImpl;
    use crate::simd::sse::f32x4;
    
    #[test]
    fn test_lsh_table_insert() {
        let mut table = LocalitySensitiveHashTable::<SimdVecImpl<f32x4, 4>>::new(16);
        
        // These two items must necessarily hash to two separate values, 
        // as they point in opposite directions
        let item1 = Rc::new(
            CacheItem::new(vec![1f32; 16].into_iter().collect::<SimdVecImpl<f32x4, 4>>())
        );
        let item2 = Rc::new(
            CacheItem::new(vec![-1f32; 16].into_iter().collect::<SimdVecImpl<f32x4, 4>>())
        );

        table.insert(item1);
        table.insert(item2);
        
        assert_eq!(table.table.len(), 2);
        
    }
    
    #[test]
    fn test_lsh_table_query() {
        let mut table = LocalitySensitiveHashTable::<SimdVecImpl<f32x4, 4>>::new(16);
        
        // The next two items will hash to the same value because they are colinear.
        let item1 = Rc::new(
            CacheItem::new(vec![1f32; 16].into_iter().collect::<SimdVecImpl<f32x4, 4>>())
        );
        let item2 = Rc::new(
            CacheItem::new(vec![2f32; 16].into_iter().collect::<SimdVecImpl<f32x4, 4>>())
        );
        // This item will go to its own entry
        let item3 = Rc::new(
            CacheItem::new(vec![-1f32; 16].into_iter().collect::<SimdVecImpl<f32x4, 4>>())
        );

        table.insert(item1);
        table.insert(item2);
        table.insert(item3);

        let mut qtemp    = vec![1f32; 16];
        let mut qtemp_op = vec![-1f32; 16];
        qtemp[0] = 0.99f32;
        qtemp_op[0] = -0.99f32;
        let q = qtemp.into_iter().collect::<SimdVecImpl<f32x4, 4>>();
        let q_op = qtemp_op.into_iter().collect::<SimdVecImpl<f32x4, 4>>();

        let q_result = table.query_set(&q);
        let qop_result = table.query_set(&q_op);
        
        // Assert there are two distinct lsh values in the table
        assert_eq!(table.table.len(), 2);
        // Assert the first query has a result set with two items 
        assert_eq!(q_result.unwrap().len(), 2);
        // Assert the second query has a result set with one item
        assert_eq!(qop_result.unwrap().len(), 1);
        
    }
    
    #[test]
    fn test_lshdb_query() {
        let mut db = LocalitySensitiveHashDatabase::<SimdVecImpl<f32x4, 4>>::new(32, 16);
        
        // The next two items will hash to the same value because they are colinear.
        let item1 = vec![1f32; 16].into_iter().collect::<SimdVecImpl<f32x4, 4>>();
        let item1_copy = vec![1f32; 16].into_iter().collect::<SimdVecImpl<f32x4, 4>>();
        let item2 = vec![2f32; 16].into_iter().collect::<SimdVecImpl<f32x4, 4>>();
        // This item will go to its own entry
        let item3 = vec![-1f32; 16].into_iter().collect::<SimdVecImpl<f32x4, 4>>();

        db.insert(item1);
        db.insert(item2);
        db.insert(item3);

        let mut qtemp = vec![1f32; 16];
        qtemp[0] = 0.99f32;
        let q = qtemp.into_iter().collect::<SimdVecImpl<f32x4, 4>>();

        let q_result = db.query(&q).unwrap();
        
        assert_eq!(*q_result, item1_copy);
    }
}
