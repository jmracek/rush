use crate::lsh::random_projection::RandomProjection;
use crate::lsh::vector::Vector;
use std::vec::Vec;

use serde::{Serialize, Deserialize};

use std::fmt::Debug;
#[derive(Debug)]
pub struct StableHashFunction<T> 
where
    T: Vector<DType=f32>,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
    projections: Vec<RandomProjection<T>>
}

impl<T: Vector<DType=f32>> StableHashFunction<T> 
where
    T: Vector<DType=f32>,
    for <'a> &'a T: IntoIterator<Item=<T as Vector>::DType>
{
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::simd::vec::SimdVecImpl;
    use crate::simd::sse::f32x4;
    use serde_json;
    
    #[test]
    fn test_insert_to_stable_hash_table() {
        let f = StableHashFunction::<SimdVecImpl<f32x4, 4>>::new(64, 16);

        use rand::Rng; 
        let mut rng = rand::thread_rng();
        let random_vector = (0..16).
            map(|_| rng.gen_range(-1f32..1f32)).
            collect::<SimdVecImpl<f32x4,4>>();

        let norm = SimdVecImpl::<f32x4,4>::dot(&random_vector, &random_vector).sqrt();
        let random_unit = random_vector / norm;

// [4,0,0,0,0,0,128,127,0,0,128,255,0,0,128,127,0,0,128,255]
    }
}

