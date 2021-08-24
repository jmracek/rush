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

impl SimdNativeType for __m128 {}

pub type f32x4  = SimdTypeProxy<__m128>;

impl f32x4 {
    pub fn from_elts(w: f32, x: f32, y: f32, z: f32) -> Self {
        unsafe {
            f32x4::new(_mm_set_ps(z, y, x, w))
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
    #[inline(always)]
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

create_binary_simd_trait!(Add, add, f32x4);
create_binary_simd_trait!(Sub, sub, f32x4);
create_binary_simd_trait!(Mul, mul, f32x4);
create_binary_simd_trait!(Div, div, f32x4);
create_ternary_simd_trait!(FusedMulAdd, fmadd, f32x4);

impl AddAssign<f32x4> for f32 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32x4) {
        unsafe {
            // In our notation, z := a_b_c_d
            let b_b_d_d = _mm_movehdup_ps(rhs.0);
            let ab_2b_cd_2d = _mm_add_ps(rhs.0, b_b_d_d);
            let cd_2d_d_d = _mm_movehl_ps(b_b_d_d, ab_2b_cd_2d);
            let abcd_rest = _mm_add_ss(ab_2b_cd_2d, cd_2d_d_d);
            let reduction: f32 = _mm_cvtss_f32(abcd_rest);
            *self += reduction;
        }
    }
}

impl Mul<f32x4> for f32 {
    type Output = f32x4;
    #[inline(always)]
    fn mul(self, rhs: f32x4) -> f32x4 {
        unsafe {
            f32x4::new(_mm_set1_ps(self)) * rhs
        }
    }
}

impl AddAssign for f32x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32x4) {
        *self = *self + rhs;
    }
}

impl SimdType for f32x4 { 
    type ElementType = f32; 
    const LANES: usize = 4;
    fn pack(elts: &Vec<f32>) -> Self {
        if elts.len() != 4 {
            panic!("Error attempting to pack elts into f32x4")
        }
        f32x4::from_elts(elts[0], elts[1], elts[2], elts[3])
    }
}

#[cfg(test)]
mod simd_f32x4_tests {
    use super::*;

    #[test]
    fn test_eq_f32x4() {
        let test_f32x4 = f32x4::from_elts(1.0, 2.0, 3.0, 4.0);
        let equals = f32x4::from_elts(1.0, 2.0, 3.0, 4.0);
        let not_equals = f32x4::from_elts(1.0, 2.0, 3.0, 5.0);
        assert_eq!(test_f32x4, equals);
        assert_ne!(test_f32x4, not_equals);
    }

    #[test]
    fn test_add_f32x4() {
        let test_f32x4 = f32x4::from_elts(1.0, 2.0, 3.0, 4.0);
        let y = f32x4::from_elts(1.0, 1.0, 1.0, 1.0);
        let result = f32x4::add(test_f32x4,y);
        assert_eq!(result, f32x4::from_elts(2.0, 3.0, 4.0, 5.0));
    }
    
    #[test]
    fn test_sub_f32x4() {
        let test_f32x4 = f32x4::from_elts(1.0, 2.0, 3.0, 4.0);
        let y = f32x4::from_elts(1.0, 1.0, 1.0, 1.0);
        let result = f32x4::sub(test_f32x4,y);
        assert_eq!(result, f32x4::from_elts(0.0, 1.0, 2.0, 3.0));
    }
    
    #[test]
    fn test_mul_f32x4() {
        let test_f32x4 = f32x4::from_elts(1.0, 2.0, 3.0, 4.0);
        let y = f32x4::from_elts(1.5, 0.1, -1.0, 0.0);
        let result = f32x4::mul(test_f32x4,y);
        assert_eq!(result, f32x4::from_elts(1.5, 0.2, -3.0, 0.0));
    }
    
    #[test]
    fn test_reduce_sum_f32x4() {
        let test_f32x4 = f32x4::from_elts(1.0, 2.0, 3.0, 4.0);
        let mut result = 0f32; 
        result += test_f32x4;
        let expected_result = 1.0 + 2.0 + 3.0 + 4.0;
        assert_eq!(result, expected_result);
    }
}

