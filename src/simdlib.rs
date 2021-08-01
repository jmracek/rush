#![allow(non_camel_case_types)]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::cmp::{PartialEq, Eq};

pub trait SimdType {}

impl SimdType for __m128 {}
impl SimdType for __m256 {}

pub trait SimdOps {
    type ElementType;
    type NativeType;
    fn zero() -> Self;
    fn sub(x: &Self, y: &Self) -> Self;
    fn add(x: &Self, y: &Self) -> Self;
    fn mul(x: &Self, y: &Self) -> Self;
    fn reduce_sum(z: &Self) -> Self::ElementType;
}

#[derive(Debug)]
struct SimdTypeWrapper<T: SimdType> {
    data: T
}

impl<T: SimdType> SimdTypeWrapper<T> {
    fn new(x: T) -> Self {
        SimdTypeWrapper {
            data: x
        }
    }
}

type f32x4  = SimdTypeWrapper<__m128>;
type f32x8  = SimdTypeWrapper<__m256>;

impl f32x4 {
    fn pack(w: f32, x: f32, y: f32, z: f32) -> Self {
        unsafe {
            f32x4::new(_mm_set_ps(w, x, y, z))
        }
    }
}

impl PartialEq for f32x4 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let elementwise_result = _mm_cmpeq_ps(self.data, other.data);
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

impl f32x8 {
    fn pack(s:f32, t: f32, u: f32, v: f32, w: f32, x: f32, y: f32, z: f32) -> Self {
        unsafe {
            f32x8::new(_mm256_set_ps(s, t, u, v, w, x, y, z))
        }
    }
}

impl PartialEq for f32x8 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let self_low:   f32x4 = f32x4::new(_mm256_castps256_ps128(self.data));
            let self_high:  f32x4 = f32x4::new(_mm256_extractf128_ps(self.data, 1));
            let other_low:  f32x4 = f32x4::new(_mm256_castps256_ps128(other.data));
            let other_high: f32x4 = f32x4::new(_mm256_extractf128_ps(other.data, 1));
            return (self_low == other_low) && (self_high == other_high)
        }
    }
}

impl Eq for f32x8 {}

//type f32x16 = __m512;
//type h32x16 = __m256bh;
//type h32x32 = __m512bh;

impl SimdOps for f32x4 {
    type ElementType = f32;
    type NativeType = __m128;
    fn zero() -> Self {
        unsafe {
            f32x4::new(_mm_setzero_ps())
        }
    }

    fn mul(x: &Self, y: &Self) -> Self {
        unsafe {
            f32x4::new(_mm_mul_ps(x.data, y.data))
        }
    }

    fn add(x: &Self, y: &Self) -> Self {
        unsafe {
            f32x4::new(_mm_add_ps(x.data, y.data))
        }
    }
    
    fn sub(x: &Self, y: &Self) -> Self {
        unsafe {
            f32x4::new(_mm_sub_ps(x.data, y.data))
        }
    }
    // SSE3  
    fn reduce_sum(z: &Self) -> Self::ElementType {
        unsafe {
            // In our notation, z := a_b_c_d
            let b_b_d_d = _mm_movehdup_ps(z.data);
            let ab_2b_cd_2d = _mm_add_ps(z.data, b_b_d_d);
            let cd_2d_d_d = _mm_movehl_ps(b_b_d_d, ab_2b_cd_2d);
            let abcd_rest = _mm_add_ss(ab_2b_cd_2d, cd_2d_d_d);
            _mm_cvtss_f32(abcd_rest)
        }
    }
}

impl SimdOps for f32x8 {
    type ElementType = f32;
    type NativeType = __m256;

    fn zero() -> Self {
        unsafe {
            f32x8::new(_mm256_setzero_ps())
        }
    }

    fn mul(x: &Self, y: &Self) -> Self {
        unsafe {
            f32x8::new(_mm256_mul_ps(x.data, y.data))
        }
    }
    
    fn add(x: &Self, y: &Self) -> Self {
        unsafe {
            f32x8::new(_mm256_add_ps(x.data, y.data))
        }
    }
    
    fn sub(x: &Self, y: &Self) -> Self {
        unsafe {
            f32x8::new(_mm256_sub_ps(x.data, y.data))
        }
    }
   
    fn reduce_sum(z: &Self) -> Self::ElementType {
        unsafe {
            let low:  f32x4 = f32x4::new(_mm256_castps256_ps128(z.data));
            let high: f32x4 = f32x4::new(_mm256_extractf128_ps(z.data, 1));
            f32x4::reduce_sum(&f32x4::add(&low, &high))
        }
    }
}


#[cfg(test)]
mod simd_type_tests {
    use super::*;

    #[test]
    fn test_add_f32x4() {
        let x = f32x4::pack(1.0, 2.0, 3.0, 4.0);
        let y = f32x4::pack(1.0, 1.0, 1.0, 1.0);
        let result = f32x4::add(&x,&y);

        assert_eq!(result, f32x4::pack(2.0, 3.0, 4.0, 5.0));
    }
}


/*
Questions:
1. How do I let rust enable different instructions when compiling for different arch?
2. How to efficiently do dot products?
*/

pub trait Simd {
}

pub struct SimdVectorImpl<T: SimdOps+Copy, const MMBLOCKS: usize> {
    chunks: [T; MMBLOCKS]
}

impl<T, const MMBLOCKS: usize> Simd for SimdVectorImpl<T, MMBLOCKS> 
where 
    T: SimdOps+Copy {}

impl<T: SimdOps+Copy, const MMBLOCKS: usize> SimdVectorImpl<T, MMBLOCKS> 
{
    fn new(size: usize) -> Self {
        let mut chunks: [T; MMBLOCKS] = [T::zero(); MMBLOCKS];
        SimdVectorImpl::<T, MMBLOCKS> {
            chunks 
        }
    }
}



pub struct SimdVector {
    vector: Box<dyn Simd>
}

/*
impl SimdVector {
    pub fn new(
    */
