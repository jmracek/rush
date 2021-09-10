use crate::simd::base::*;
use crate::lsh::lsh_database::Cacheable;
use crate::simd::sse::f32x4;
//use crate::simd::avx::f32x8;
use crate::simd::murmur::murmur3_x64_128;
use core::ops::{Add, Sub, Div, Mul, AddAssign};
use std::iter::{IntoIterator, FromIterator, Iterator};
use itertools::zip_eq;

trait SimdVec  { } 

#[derive(Debug)]
pub struct SimdVecImpl<T: SimdType, const MMBLOCKS: usize> {
    chunks: [T; MMBLOCKS]
}

pub struct SimdVecImplElementIterator<'a, T: SimdType, const MMBLOCKS: usize> {
    obj: &'a SimdVecImpl<T, MMBLOCKS>,
    cur: usize
}

impl<'a, T: SimdType, const MMBLOCKS: usize> Iterator for SimdVecImplElementIterator<'a, T, MMBLOCKS> {
    type Item = T::ElementType;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= MMBLOCKS * T::LANES {
            None
        }
        else {
            let arr = &self.obj.chunks;
            let chunk_idx = self.cur / T::LANES;
            let lane_idx = (self.cur % T::LANES) as isize;
            unsafe { 
                let chunk_ptr = std::mem::transmute::<&T, *const T::ElementType>(&arr[chunk_idx]);
                let element_ptr = chunk_ptr.offset(lane_idx);
                self.cur += 1;
                Some(*element_ptr)
            }
        }
    }
}

impl<'a, T: SimdType, const MMBLOCKS: usize> IntoIterator for &'a SimdVecImpl<T, MMBLOCKS> {
    type Item = T::ElementType;
    type IntoIter = SimdVecImplElementIterator<'a, T, MMBLOCKS>;
    fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        SimdVecImplElementIterator {
            obj: &self,
            cur: 0
        }
    }
}

struct SimdVecImplIterator<'a, T: SimdType, const MMBLOCKS: usize> {
    obj: &'a SimdVecImpl<T, MMBLOCKS>,
    cur: usize
}

impl<'a, T: SimdType, const MMBLOCKS: usize> Iterator for SimdVecImplIterator<'a, T, MMBLOCKS> {
    type Item = (T, T, T, T);
    fn next(&mut self) -> Option<Self::Item> {
        let arr = &self.obj.chunks;
        let result: Option<Self::Item>;
        
        if self.cur <= MMBLOCKS - 4 {
            result = Some((arr[self.cur], arr[self.cur + 1], arr[self.cur + 2], arr[self.cur + 3]));
        }
        else {
            result = None
        }
        self.cur += 4;        
        result
    }
}

impl<T: SimdType, const MMBLOCKS: usize> SimdVecImpl<T, MMBLOCKS> {
    pub fn new() -> Self where Self: Sized {
        let chunks: [T; MMBLOCKS] = [T::default(); MMBLOCKS];
        SimdVecImpl::<T, MMBLOCKS> {
            chunks 
        }
    }

    fn set_chunk(&mut self, idx: usize, data: T) {
        self.chunks[idx] = data;
    }

    fn chunk_iter<'a>(&'a self) -> SimdVecImplIterator<'a, T, MMBLOCKS> {
        SimdVecImplIterator {
            obj: &self,
            cur: 0
        }
    }
}

impl<T: SimdType, const MMBLOCKS: usize> FromIterator<T::ElementType> for SimdVecImpl<T, MMBLOCKS> {
    // The problem here can be that the number of elements in the iterator is 
    // too large to fit in the array.  If this is the case, we simply ignore
    // any trailing elements.  Similarly, if there are too few elements to fit
    // the array, then we simply pad the remainder with zeros.
    fn from_iter<I: IntoIterator<Item=T::ElementType>>(iter: I) -> Self {
        let mut result = SimdVecImpl::new();
        let mut chunk_vec = vec![T::ElementType::default(); T::LANES];
        let mut idx = 0usize;
        let mut chunk_idx = 0;

        for elt in iter {
            // First, check whether we've run out of room to store elements
            if idx == MMBLOCKS * T::LANES {
                break;
            }

            let lane_idx = idx % T::LANES;
            chunk_vec[lane_idx] = elt;
            
            // If we've pulled off enough elements to form a chunk,
            // set the chunk, reset the container, and increment.
            if lane_idx + 1 == T::LANES {
                result.set_chunk(chunk_idx, T::pack(&chunk_vec));
                for item in chunk_vec.iter_mut() {
                    *item = T::ElementType::default();
                }
                chunk_idx += 1;
            }

            idx += 1;
        }

        // If we didn't complete a chunk, write the last one in.
        if idx % T::LANES > 0 {
            result.set_chunk(chunk_idx, T::pack(&chunk_vec))
        }

        result
    }
}

use crate::lsh::vector::VectorArithmetic;

impl<T: SimdType, const MMBLOCKS: usize> Default for SimdVecImpl<T, MMBLOCKS> {
    fn default() -> Self {
        SimdVecImpl::new()
    }
}

impl<T: SimdType, const MMBLOCKS: usize> Add for SimdVecImpl<T, MMBLOCKS>
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut result = SimdVecImpl::<T, MMBLOCKS>::new();
        for (idx, (a, b)) in zip_eq(self.chunks.iter(), other.chunks.iter()).enumerate() {
            result.set_chunk(idx, *a + *b);
        }
        result
    }
}

impl<T: SimdType, const MMBLOCKS: usize> Sub for SimdVecImpl<T, MMBLOCKS> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let mut result = SimdVecImpl::<T, MMBLOCKS>::new();
        for (idx, (a, b)) in zip_eq(self.chunks.iter(), other.chunks.iter()).enumerate() {
            result.set_chunk(idx, *a - *b);
        }
        result
    }
}

impl<T: SimdType, const MMBLOCKS: usize> Mul for SimdVecImpl<T, MMBLOCKS> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let mut result = SimdVecImpl::<T, MMBLOCKS>::new();
        for (idx, (a, b)) in zip_eq(self.chunks.iter(), other.chunks.iter()).enumerate() {
            result.set_chunk(idx, *a * *b);
        }
        result
    }
}

impl<T: SimdType, const MMBLOCKS: usize> Div for SimdVecImpl<T, MMBLOCKS> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let mut result = SimdVecImpl::<T, MMBLOCKS>::new();
        for (idx, (a, b)) in zip_eq(self.chunks.iter(), other.chunks.iter()).enumerate() {
            result.set_chunk(idx, *a / *b);
        }
        result
    }
}

impl<T: SimdType<ElementType=f32>, const MMBLOCKS: usize> Mul<f32> for SimdVecImpl<T, MMBLOCKS> where
    f32: Mul<T, Output=T>
{
    type Output = Self;
    fn mul(self, c: T::ElementType) -> Self {
        let mut result = SimdVecImpl::<T, MMBLOCKS>::new();
        for (idx, chunk) in self.chunks.iter().enumerate() {
            result.set_chunk(idx, c * *chunk);
        }
        result
    }
}

impl<T: SimdType<ElementType=f32>, const MMBLOCKS: usize> Div<f32> for SimdVecImpl<T, MMBLOCKS> where
    Self: Mul<f32, Output=Self>
{
    type Output = Self;
    fn div(self, c: T::ElementType) -> Self {
        self * (1f32 / c)
    }
}

impl<T: SimdType<ElementType=f32>, const MMBLOCKS: usize> VectorArithmetic for SimdVecImpl<T, MMBLOCKS> where
    f32: Mul<T, Output=T>
{
    type DType = T::ElementType;
}



use crate::lsh::vector::Vector;

impl<T: SimdType<ElementType=f32>, const MMBLOCKS: usize> Vector for SimdVecImpl<T, MMBLOCKS> where
    f32: Mul<T, Output=T>,
    f32: AddAssign<T>
{
    type DType = T::ElementType;

    fn dot(&self, other: &Self) -> <Self as Vector>::DType {
        let (ns1, ns2, ns3, ns4) = 
            zip_eq(self.chunk_iter(), other.chunk_iter()).
            fold((T::default(), T::default(), T::default(), T::default()), 
                 |(acc1, acc2, acc3, acc4), ((x1, x2, x3, x4), (y1, y2, y3, y4))| {
                     (x1.fmadd(y1, acc1), x2.fmadd(y2, acc2), x3.fmadd(y3, acc3), x4.fmadd(y4, acc4))
                 });
        let mut hsum = 0f32;
        let r1 =  ns1 + ns2;
        let r2 =  ns3 + ns4;
        let inner_product = r1 + r2;
        hsum += inner_product;
        hsum 
    }

    fn distance(&self, other: &Self) -> <Self as Vector>::DType {
        let (ns1, ns2, ns3, ns4) = 
            zip_eq(self.chunk_iter(), other.chunk_iter()).
                fold((T::default(), T::default(), T::default(), T::default()), |(acc1, acc2, acc3, acc4), ((x1, x2, x3, x4), (y1, y2, y3, y4))| {
                    let delta1 = x1 - y1;
                    let delta2 = x2 - y2;
                    let delta3 = x3 - y3;
                    let delta4 = x4 - y4;
                    (delta1 * delta1 + acc1, delta2 * delta2 + acc2, delta3 * delta3 + acc3, delta4 * delta4 + acc4)
                    //(delta1.fmadd(delta1, acc1), delta2.fmadd(delta2, acc2), delta3.fmadd(delta3, acc3), delta4.fmadd(delta4, acc4))
                });
        let mut result = 0f32;
        let r1 =  ns1 + ns2;
        let r2 =  ns3 + ns4;
        let norm_squared = r1 + r2;
        result += norm_squared;
        result.sqrt()
    }

    fn dimension(&self) -> usize {
        MMBLOCKS * T::LANES
    }
}

impl<T: SimdType, const MMBLOCKS: usize> PartialEq for SimdVecImpl<T, MMBLOCKS> 
where T: Default+PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        zip_eq(self.chunks.iter(), other.chunks.iter()).
            fold(true, |acc, (x, y)| acc && (x == y))
    }
}

impl<T: SimdType, const MMBLOCKS: usize> Eq for SimdVecImpl<T, MMBLOCKS> 
where T: Default+PartialEq
{}

impl<const MMBLOCKS: usize> Cacheable for SimdVecImpl<f32x4, MMBLOCKS> {
    fn cache_id(&self) -> u128 {
        murmur3_x64_128(&self.chunks, 0u32)
    }
}



/*
pub struct SimdVector {
    vector: Box<dyn SimdVec>,
    dim: usize
}

pub struct SimdError(String);
impl SimdVector {
    pub fn new(dim: usize) -> Result<SimdVector, SimdError> {
        let vector: Option<Box<dyn SimdVec>> = match dim {
            0..=32       => Some(Box::new(SimdVecImpl::<f32x8, 4>::new())),
            33..=128     => Some(Box::new(SimdVecImpl::<f32x8, 16>::new())),
            129..=512    => Some(Box::new(SimdVecImpl::<f32x8, 64>::new())),
            513..=768    => Some(Box::new(SimdVecImpl::<f32x8, 96>::new())),
            769..=1024   => Some(Box::new(SimdVecImpl::<f32x8, 128>::new())),
            1025..=2048  => Some(Box::new(SimdVecImpl::<f32x8, 256>::new())),
            2049..=4096  => Some(Box::new(SimdVecImpl::<f32x8, 512>::new())),
            _ => None 
        };

        if let Some(item) = vector {
            Ok(SimdVector{ vector: item, dim: dim })
        }
        else {
            Err(SimdError("Maximum supported dimension of SIMD vector is 4096".to_string()))
        }
    }
}
*/
/*
#[cfg(test)]
mod simd_vector_impl_tests {
    use super::*;
    use crate::simd::sse::*;

    #[test]
    fn test_simd_vector_impl_add() {
        let mut x = SimdVecImpl::<f32x4, 2>::new();
        x.set_chunk(0, f32x4::pack(0.0, 1.0, 2.0, 3.0));
        x.set_chunk(1, f32x4::pack(1.0, 2.0, 3.0, 4.0));
        
        let mut y = SimdVecImpl::<f32x4, 2>::new();
        y.set_chunk(0, f32x4::pack(1.0, 1.0, 1.0, -1.0));
        y.set_chunk(1, f32x4::pack(1.0, 1.0, 1.0, -1.0));
        
        let mut expected_result = SimdVecImpl::<f32x4, 2>::zero();
        expected_result.set_chunk(0, f32x4::pack(1.0, 2.0, 3.0, 2.0));
        expected_result.set_chunk(1, f32x4::pack(2.0, 3.0, 4.0, 3.0));
        
        let result = SimdVecImpl::add(&x, &y);

        assert_eq!(&result, &expected_result);
    }
    
    #[test]
    fn test_simd_vector_impl_sub() {
        let mut x = SimdVecImpl::<f32x4, 2>::zero();
        x.set_chunk(0, f32x4::pack(0.0, 1.0, 2.0, 3.0));
        x.set_chunk(1, f32x4::pack(1.0, 2.0, 3.0, 4.0));
        
        let mut y = SimdVecImpl::<f32x4, 2>::zero();
        y.set_chunk(0, f32x4::pack(1.0, 1.0, 1.0, -1.0));
        y.set_chunk(1, f32x4::pack(1.0, 1.0, 1.0, -1.0));
        
        let mut expected_result = SimdVecImpl::<f32x4, 2>::zero();
        expected_result.set_chunk(0, f32x4::pack(-1.0, 0.0, 1.0, 4.0));
        expected_result.set_chunk(1, f32x4::pack(0.0, 1.0, 2.0, 5.0));
        
        let result = SimdVecImpl::sub(&x, &y);

        assert_eq!(&result, &expected_result);
    }
}
*/


//impl<T: SimdType, const MMBLOCKS: usize> SimdVec for SimdVecImpl<T, MMBLOCKS> {}


