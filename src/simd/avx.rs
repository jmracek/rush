#![allow(non_camel_case_types)]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::default::Default;
use core::ops::{Add, Sub, Div, Mul, AddAssign};
use std::cmp::{PartialEq, Eq};

use paste::paste;
use crate::simd::base::*;
use crate::simd::sse::*;

impl SimdNativeType for __m256 {}

pub type f32x8  = SimdTypeProxy<__m256>;

impl f32x8 {
    fn from_elts(s:f32, t: f32, u: f32, v: f32, w: f32, x: f32, y: f32, z: f32) -> Self {
        unsafe {
            f32x8::new(_mm256_set_ps(s, t, u, v, w, x, y, z))
        }
    }    
}

impl Default for f32x8 {
    fn default() -> Self {
        unsafe {
            f32x8::new(_mm256_setzero_ps())
        }
    }
}

impl PartialEq for f32x8 {
    #[inline(always)]
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

create_binary_simd_trait!(Add, add, f32x8, 256);
create_binary_simd_trait!(Sub, sub, f32x8, 256);
create_binary_simd_trait!(Mul, mul, f32x8, 256);
create_binary_simd_trait!(Div, div, f32x8, 256);
create_ternary_simd_trait!(FusedMulAdd, fmadd, f32x8, 256);

impl Mul<f32x8> for f32 {
    type Output = f32x8;
    #[inline(always)]
    fn mul(self, rhs: f32x8) -> f32x8 {
        unsafe {
            f32x8::new(_mm256_set1_ps(self)) * rhs
        }
    }
}

impl AddAssign<f32x8> for f32 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32x8) {
        unsafe {
            let low:  f32x4 = f32x4::new(_mm256_castps256_ps128(rhs.0));
            let high: f32x4 = f32x4::new(_mm256_extractf128_ps(rhs.0, 1));
            let combined = low + high;
            *self += combined;
        }
    }
}

impl AddAssign for f32x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32x8) {
        *self = *self + rhs;
    }
}

impl SimdType for f32x8 { 
    type ElementType = f32; 
    const LANES: usize = 8;
    fn pack(elts: &Vec<f32>) -> Self {
        if elts.len() != 8 {
            panic!("Error attempting to pack elts into f32x8")
        }
        f32x8::from_elts(elts[0], elts[1], elts[2], elts[3],
                         elts[4], elts[5], elts[6], elts[7])
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

