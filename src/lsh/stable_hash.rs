use crate::lsh::random_projection::RandomProjection;
use crate::lsh::vector::Vector;
use std::collections::HashMap;
use std::vec::Vec;
use std::rc::Rc;

pub struct StableHashFunction<T: Vector> {
    projections: Vec<RandomProjection<T>>
}

impl<T: Vector<DType=f32>> StableHashFunction<T> {
    pub fn new(bits: usize, dimension: usize) -> Self {
        let projections = (0..bits).
            map(|_| RandomProjection::<T>::new(dimension)).
            collect::<Vec<RandomProjection<T>>>();
        
        StableHashFunction { projections }
    }

    pub fn hash(&self, v: &T) -> u64 {
        self.projections.
            iter().
            map(|proj| proj.hash(v)).
            enumerate().
            fold(0u64, |acc, (i, sgn)| acc | (sgn << i))
    }
}

struct StableHashTable<T: Vector> {
    table: HashMap<u64, Vec<Rc<T>>>,
    hashfn: StableHashFunction<T>
}

impl<T: Vector<DType=f32>> StableHashTable<T> {
    fn new(dimension: usize) -> Self {
        StableHashTable {
            table: HashMap::<u64, Vec<Rc<T>>>::new(),
            hashfn: StableHashFunction::<T>::new(64, dimension)
        }
    }

    fn insert(&mut self, item: Rc<T>) {
        let hash_key = self.hashfn.hash(&*item);
        if let Some(container) = self.table.get_mut(&hash_key) {
            container.push(item);
        }
        else {
            self.table.insert(hash_key, vec![item]);
        }
    }
    
    fn query<'a>(&'a self, item: &T) -> Option<&'a T> {
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

    fn query_set<'a>(&'a self, item: &T) -> Option<&'a Vec<Rc<T>>> {
        self.table.get(&self.hashfn.hash(item))
    }
}
