#![allow(non_camel_case_types)]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::default::Default;
use std::cmp::{PartialEq, Eq};
use core::ops::{Add, Sub, Div, Mul, AddAssign};
use std::iter::{FromIterator, Iterator};
use itertools::zip_eq;
use paste::paste;

pub trait SimdType: Sized {
    type ElementType;
    const LANES: usize;
}

impl SimdType for __m128 { 
    type ElementType = f32; 
    const LANES: usize = 4;
}
impl SimdType for __m256 { 
    type ElementType = f32; 
    const LANES: usize = 8;
}

#[derive(Debug, Copy, Clone)]
struct SimdTypeProxy<T: SimdType>(T);

/*
pub trait SimdFactory<const LANES: usize, T: SimdType<LANES>> {
    fn zero() -> T;
    fn pack(elts: [T::ElementType; LANES]) -> SimdTypeProxy<SimdType<LANES>>;
}*/

impl<T: SimdType> SimdTypeProxy<T> {
    fn new(x: T) -> Self {
        SimdTypeProxy(x)
    }
}

type f32x4  = SimdTypeProxy<__m128>;

impl f32x4 {
    fn pack(w: f32, x: f32, y: f32, z: f32) -> Self {
        unsafe {
            f32x4::new(_mm_set_ps(w, x, y, z))
        }
    }
}

impl Default for f32x4 {
    fn default() -> Self {
        unsafe {
            f32x4::new(_mm_setzero_ps())
        }
    }
}

impl PartialEq for f32x4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let elementwise_result = _mm_cmpeq_ps(self.0, other.0);
            let b_b_d_d = _mm_movehdup_ps(elementwise_result);
            let ab_2b_cd_2d = _mm_and_ps(elementwise_result, b_b_d_d);
            let cd_2d_d_d = _mm_movehl_ps(b_b_d_d, ab_2b_cd_2d);
            let abcd_rest = _mm_and_ps(ab_2b_cd_2d, cd_2d_d_d);
            let reduced_eq = _mm_cvtss_si32(abcd_rest);
            // The only possibilities are that this is zero (neq), or 0xFFFFFFFF.
            // I'm not sure the best way to compare 0xFFFFFFFF as i32 so I'll just
            // compare to neq 0.
            reduced_eq != 0
        }
    }
}

impl Eq for f32x4 {}

macro_rules! create_simd_trait {
    ($trait:ident, $method:ident, $type:ty) => {
        impl $trait for $type {
            type Output = $type;
            fn $method(self, other: Self) -> Self {
                unsafe {
                    <$type>::new(paste!{[<_mm _$method _ps>]}(self.0, other.0))
                }
            }
        }
    };
    
    ($trait:ident, $method:ident, $type:ty, $bits:literal) => {
        impl $trait for $type {
            type Output = $type;
            fn $method(self, other: Self) -> Self {
                unsafe {
                    <$type>::new(paste!{[<_mm $bits _$method _ps>]}(self.0, other.0))
                }
            }
        }
    };
    
    ($trait:ident, $method:ident, $type:ty, $bits:literal, $precision:ident) => {
        impl $trait for $type {
            type Output = $type;
            fn $method(self, other: Self) -> Self {
                unsafe {
                    <$type>::new(paste!{[<_mm $bits _$method _$precision>]}(self.0, other.0))
                }
            }
        }
    };
}

create_simd_trait!(Add, add, f32x4);
create_simd_trait!(Sub, sub, f32x4);
create_simd_trait!(Mul, mul, f32x4);
create_simd_trait!(Div, div, f32x4);

impl AddAssign<f32x4> for f32 {
    fn add_assign(&mut self, rhs: f32x4) {
        let reduction: f32;
        unsafe {
            // In our notation, z := a_b_c_d
            let b_b_d_d = _mm_movehdup_ps(rhs.0);
            let ab_2b_cd_2d = _mm_add_ps(rhs.0, b_b_d_d);
            let cd_2d_d_d = _mm_movehl_ps(b_b_d_d, ab_2b_cd_2d);
            let abcd_rest = _mm_add_ss(ab_2b_cd_2d, cd_2d_d_d);
            reduction = _mm_cvtss_f32(abcd_rest);
        }
        *self += reduction;
    }
}

type f32x8  = SimdTypeProxy<__m256>;

impl f32x8 {
    fn pack(s:f32, t: f32, u: f32, v: f32, w: f32, x: f32, y: f32, z: f32) -> Self {
        unsafe {
            f32x8::new(_mm256_set_ps(s, t, u, v, w, x, y, z))
        }
    }
    
    fn zero() -> Self {
        unsafe {
            f32x8::new(_mm256_setzero_ps())
        }
    }
}

impl PartialEq for f32x8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let self_low:   f32x4 = f32x4::new(_mm256_castps256_ps128(self.0));
            let self_high:  f32x4 = f32x4::new(_mm256_extractf128_ps(self.0, 1));
            let other_low:  f32x4 = f32x4::new(_mm256_castps256_ps128(other.0));
            let other_high: f32x4 = f32x4::new(_mm256_extractf128_ps(other.0, 1));
            return (self_low == other_low) && (self_high == other_high)
        }
    }
}

impl Eq for f32x8 {}

create_simd_trait!(Add, add, f32x8, 256);
create_simd_trait!(Sub, sub, f32x8, 256);
create_simd_trait!(Mul, mul, f32x8, 256);
create_simd_trait!(Div, div, f32x8, 256);

impl AddAssign<f32x8> for f32 {
    fn add_assign(&mut self, rhs: f32x8) {
        unsafe {
            let low:  f32x4 = f32x4::new(_mm256_castps256_ps128(rhs.0));
            let high: f32x4 = f32x4::new(_mm256_extractf128_ps(rhs.0, 1));
            *self += low;
            *self += high;
        }
    }
}

//type f32x16 = __m512;
//type h32x16 = __m256bh;
//type h32x32 = __m512bh;

/*
Questions:
1. How do I let rust enable different instructions when compiling for different arch?
2. How do I want to switch between different instruction sets?
*/

trait SimdVec { } 

#[derive(Debug)]
struct SimdVecImpl<T: Copy+Default+Sized, const MMBLOCKS: usize> {
    chunks: [T; MMBLOCKS]
}

struct SimdVecImplIterator<'a, T: Copy+Default, const MMBLOCKS: usize> {
    obj: &'a SimdVecImpl<T, MMBLOCKS>,
    cur: usize
}

impl<'a, T: Copy+Default, const MMBLOCKS: usize> Iterator for SimdVecImplIterator<'a, T, MMBLOCKS> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= MMBLOCKS {
            None
        }
        else {
            let result = self.obj.chunks[self.cur];
            self.cur += 1;
            Some(result) 
        }
    }
}

impl<T: Copy+Default, const MMBLOCKS: usize> SimdVec for SimdVecImpl<T, MMBLOCKS> {}

impl<T: Copy+Default, const MMBLOCKS: usize> SimdVecImpl<T, MMBLOCKS> {
    
    fn new() -> Self {
        let mut chunks: [T; MMBLOCKS] = [T::default(); MMBLOCKS];
        SimdVecImpl::<T, MMBLOCKS> {
            chunks 
        }
    }

    fn set_chunk(&mut self, idx: usize, data: T) {
        self.chunks[idx] = data;
    }

    fn iter<'a>(&'a self) -> SimdVecImplIterator<'a, T, MMBLOCKS> {
        SimdVecImplIterator::<T, MMBLOCKS> {
            obj: &self,
            cur: 0
        }
    }
}

/*
impl<T: Copy+Default, const MMBLOCKS: usize> FromIterator<T> for SimdVecImpl<T, MMBLOCKS> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        let mut result = Self::new();
        let mut idx = 0;
        for item in iter {
            result.set_chunk(idx, item);
            idx += 1;
        }
        result
    }
}
*/

/*
impl<T: Copy, const MMBLOCKS: usize> SimdOps for SimdVecImpl<T, MMBLOCKS> {
    type ElementType = T::ElementType;
    type NativeType = T::NativeType;

    fn zero() -> Self {
        SimdVecImpl::<T, MMBLOCKS>::new()
    }

    fn sub(x: &Self, y: &Self) -> Self {
        zip_eq(x.iter(), y.iter()).
        map( |(x, y)| T::sub(&x, &y) ).
        collect::<Self>()
    }

    fn add(x: &Self, y: &Self) -> Self {
        zip_eq(x.iter(), y.iter()).
        map( |(x, y)| T::add(&x, &y) ).
        collect::<Self>()
    }

    fn mul(x: &Self, y: &Self) -> Self {
        zip_eq(x.iter(), y.iter()).
        map( |(x, y)| T::mul(&x, &y) ).
        collect::<Self>()
    }

    fn reduce_sum(z: &Self) -> Self::ElementType {
        z.iter().fold(Self::ElementType::default(), |acc, x| {
            acc + T::reduce_sum(&x)
        })
    }
}
*/

impl<T, const MMBLOCKS: usize> PartialEq for SimdVecImpl<T, MMBLOCKS> 
where T: Copy+Default+PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        zip_eq(self.iter(), other.iter()).
            fold(true, |acc, (x, y)| acc && (x == y))
    }
}

impl<T, const MMBLOCKS: usize> Eq for SimdVecImpl<T, MMBLOCKS> 
where T: Copy+Default+PartialEq
{}

pub struct SimdVector {
    vector: Box<dyn SimdVec>,
    dim: usize
}
struct SimdError(String);

impl SimdVector {
    pub fn new(dim: usize) -> Result<SimdVector, SimdError> {
        let vector = match dim {
            0..=32       => Some(Box::<dyn SimdVec>::new(SimdVecImpl::<f32x4, 4>::new())),
            33..=128     => Some(Box::<dyn SimdVec>::new(SimdVecImpl::<f32x4, 16>::new())),
            129..=512    => Some(Box::<dyn SimdVec>::new(SimdVecImpl::<f32x4, 64>::new())),
            512..=768    => Some(Box::<dyn SimdVec>::new(SimdVecImpl::<f32x4, 96>::new())),
            769..=1024   => Some(Box::<dyn SimdVec>::new(SimdVecImpl::<f32x4, 128>::new())),
            1025..=2048  => Some(Box::<dyn SimdVec>::new(SimdVecImpl::<f32x4, 256>::new())),
            2049..=4096  => Some(Box::<dyn SimdVec>::new(SimdVecImpl::<f32x4, 512>::new())),
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

/*
impl SimdVector {
    pub fn new(
    */


#[cfg(test)]
mod simd_f32x4_tests {
    use super::*;

    #[test]
    fn test_eq_f32x4() {
        let test_f32x4 = f32x4::pack(1.0, 2.0, 3.0, 4.0);
        let equals = f32x4::pack(1.0, 2.0, 3.0, 4.0);
        let not_equals = f32x4::pack(1.0, 2.0, 3.0, 5.0);
        assert_eq!(test_f32x4, equals);
        assert_ne!(test_f32x4, not_equals);
    }

    #[test]
    fn test_add_f32x4() {
        let test_f32x4 = f32x4::pack(1.0, 2.0, 3.0, 4.0);
        let y = f32x4::pack(1.0, 1.0, 1.0, 1.0);
        let result = f32x4::add(test_f32x4,y);
        assert_eq!(result, f32x4::pack(2.0, 3.0, 4.0, 5.0));
    }
    
    #[test]
    fn test_sub_f32x4() {
        let test_f32x4 = f32x4::pack(1.0, 2.0, 3.0, 4.0);
        let y = f32x4::pack(1.0, 1.0, 1.0, 1.0);
        let result = f32x4::sub(test_f32x4,y);
        assert_eq!(result, f32x4::pack(0.0, 1.0, 2.0, 3.0));
    }
    
    #[test]
    fn test_mul_f32x4() {
        let test_f32x4 = f32x4::pack(1.0, 2.0, 3.0, 4.0);
        let y = f32x4::pack(1.5, 0.1, -1.0, 0.0);
        let result = f32x4::mul(test_f32x4,y);
        assert_eq!(result, f32x4::pack(1.5, 0.2, -3.0, 0.0));
    }
    
    #[test]
    fn test_reduce_sum_f32x4() {
        let test_f32x4 = f32x4::pack(1.0, 2.0, 3.0, 4.0);
        let mut result = 0f32; 
        result += test_f32x4;
        let expected_result = 1.0 + 2.0 + 3.0 + 4.0;
        assert_eq!(result, expected_result);
    }
}

#[cfg(test)]
mod simd_f32x8_tests {
    use super::*;

    #[test]
    fn test_eq_f32x8() {
        let test_f32x8 = f32x8::pack(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let equals = f32x8::pack(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let not_equals = f32x8::pack(1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0);
        assert_eq!(test_f32x8, equals);
        assert_ne!(test_f32x8, not_equals);
    }

    #[test]
    fn test_add_f32x8() {
        let test_f32x8 = f32x8::pack(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let y = f32x8::pack(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let result = f32x8::add(test_f32x8,y);
        assert_eq!(result, f32x8::pack(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    }
    
    #[test]
    fn test_sub_f32x8() {
        let test_f32x8 = f32x8::pack(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let y = f32x8::pack(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let result = f32x8::sub(test_f32x8,y);
        assert_eq!(result, f32x8::pack(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0));
    }
    
    #[test]
    fn test_mul_f32x8() {
        let test_f32x8 = f32x8::pack(1.0, 2.0, 3.0, 4.0, 1.0, 6.0, 7.0, 8.0);
        let y = f32x8::pack(1.5, 0.1, -1.0, 0.0, 0.001, 2.0, 0.7, 1.1);
        let result = f32x8::mul(test_f32x8,y);
        assert_eq!(result, f32x8::pack(1.5, 0.2, -3.0, 0.0, 0.001, 12.0, 4.9, 8.8));
    }
    
    #[test]
    fn test_reduce_sum_f32x8() {
        let test_f32x8 = f32x8::pack(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let mut result = 0f32;
        result += test_f32x8;
        let expected_result = 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0;
        assert_eq!(result, expected_result);
    }
}


/*
#[cfg(test)]
mod simd_vector_impl_tests {
    use super::*;

    #[test]
    fn test_simd_vector_impl_add() {
        let mut x = SimdVecImpl::<f32x4, 2>::zero();
        x.set_chunk(0, f32x4::pack(0.0, 1.0, 2.0, 3.0));
        x.set_chunk(1, f32x4::pack(1.0, 2.0, 3.0, 4.0));
        
        let mut y = SimdVecImpl::<f32x4, 2>::zero();
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
