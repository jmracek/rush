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
        let mut table = LocalitySensitiveHashTable::<SimdVecImpl<f32x4, 1>>::new(4);
        
        let item1 = Rc::new(
            CacheItem::new(vec![1f32, 2f32, 3f32, 4f32].into_iter().collect::<SimdVecImpl<f32x4, 1>>())
        );
        let item2 = Rc::new(
            CacheItem::new(vec![-1f32, 0f32, 0f32, -1f32].into_iter().collect::<SimdVecImpl<f32x4, 1>>())
        );

        table.insert(item1);
        table.insert(item2);
        
        assert_eq!(table.table.len(), 2);
        
    }
}

/*

        let result = 
            self.tables.
                iter().
                map(|table| table.query(&item)).
                fold(None, |acc, x| {
                    match acc {
                        None => {
                            match x {
                                None => None,
                                Some(candidate) => Some((candidate, candidate.distance(item)))
                            }
                        },
                        Some((_, min_distance)) => {
                            match x {
                                Some(next_candidate) => {
                                    let current_distance = next_candidate.distance(item));
                                    if  current_distance < min_distance {
                                        Some((next_candidate, current_distance))
                                    }
                                    else {
                                        acc
                                    }
                                },
                                None => acc
                            }
                        }
                    }
                });
        match result {
            Some((nearest_neighbour, _)) => Some(nearest_neighbour),
            _ => None
        }


TODO's:
    1. Implement ability to load existing LocalitySensitiveHashDatabase from an existing saved model.
    2. Implement the ability to query an LSH database
    3. Figure out how a convenient way to implement FromIterator<T> for LSHDatabase.  The problem is
       that in the from_iter function we would need to know the number of replicas, the dimension, and
       the number of projections to use (bits).  
    4. Implement the service layer for the LSH cache.



impl<'a, T: Euclidian> FromIterator<T> for LocalitySensitiveHashDatabase<'a, T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        let mut db = LocalitySensitiveHashDatabase::<T>::new();
        for item in iter {
            db.insert(item);
        }
        db
    }
}
*/

/*
    pub fn query<'a>(&'a self, item: &T) -> Option<&'a T> {
        match self.table.get(&self.hashfn.hash(item)) {
            None => None,
            Some(result_set) => {
                let best_match: Option<(usize, f32)> = 
                    result_set.
                        iter().
                        map(|q| T::distance(&**q, item)).
                        enumerate().
                        fold(None, |acc, (idx, q)| {
                            match acc {
                                None => Some((idx, q)),
                                Some((min_idx, p)) => {
                                    if q < p { Some((idx, q)) } else { Some((min_idx, p)) }
                                }
                            }
                        });
                
                match best_match {
                    Some((idx_closest, _)) => Some(&*result_set[idx_closest]),
                    _ => None
                }
            }
        }
    }



        let (_, candidates) = self.tables.
            iter().
            map(|table| table.query_set(item)).
            fold((observed_hashes, result_set), |(mut obs, mut r), returned_results| {
                match returned_results {
                    Some(items) => {
                        for v in items {
                            let ident = v.hash();
                            if !obs.contains(ident) {
                                obs.insert(ident);
                                r.push(Rc::clone(v))
                            }
                        }
                        (obs, r)
                    }
                    _ => (obs, r)
                }
            });


*/
