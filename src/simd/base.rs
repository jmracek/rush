use core::ops::{Add, Sub, Div, Mul, AddAssign};
use std::default::Default;

pub trait SimdNativeType {}

pub trait FusedMulAdd {
    type Output;
    fn fmadd(self, b: Self, c: Self) -> Self;
}

pub trait SimdType: 
    Copy +
    Sized + 
    Default +
    Add<Output=Self> +
    Sub<Output=Self> +
    Mul<Output=Self> +
    Div<Output=Self> + 
    AddAssign +
    FusedMulAdd 
{
    type ElementType: 
        Mul<Self, Output=Self> +
        AddAssign<Self> +
        Default +
        Clone +
        Copy;
    const LANES: usize;
    fn pack(elts: &Vec<Self::ElementType>) -> Self;
}

#[derive(Debug, Copy, Clone)]
pub struct SimdTypeProxy<T: SimdNativeType>(pub T);

impl<T: SimdNativeType> SimdTypeProxy<T> {
    pub fn new(x: T) -> Self {
        SimdTypeProxy(x)
    }
}

macro_rules! create_binary_simd_trait {
    ($trait:ident, $method:ident, $type:ty) => {
        impl $trait for $type {
            type Output = $type;
            
            #[inline(always)]
            fn $method(self, other: Self) -> Self {
                unsafe {
                    <$type>::new(paste!{[<_mm _$method _ps>]}(self.0, other.0))
                }
            }
        }
    };
    
    ($trait:ident, $method:ident, $type:ty, $bits:literal) => {
        paste! {
            impl $trait for $type {
                type Output = $type;
                
                #[inline(always)]
                fn $method(self, other: Self) -> Self {
                    unsafe {
                        <$type>::new([<_mm $bits _$method _ps>](self.0, other.0))
                    }
                }
            }
        }

    };
    
    ($trait:ident, $method:ident, $type:ty, $bits:literal, $precision:ident) => {
        impl $trait for $type {
            type Output = $type;
            
            #[inline(always)]
            fn $method(self, other: Self) -> Self {
                unsafe {
                    <$type>::new(paste!{[<_mm $bits _$method _$precision>]}(self.0, other.0))
                }
            }
        }
    };
}

macro_rules! create_ternary_simd_trait {
    ($trait:ident, $method:ident, $type:ty) => {
        impl $trait for $type {
            type Output = $type;
            
            #[inline(always)]
            fn $method(self, b: Self, c: Self) -> Self {
                unsafe {
                    <$type>::new(paste!{[<_mm _$method _ps>]}(self.0, b.0, c.0))
                }
            }
        }
    };
    
    ($trait:ident, $method:ident, $type:ty, $bits:literal) => {
        impl $trait for $type {
            type Output = $type;
            
            #[inline(always)]
            fn $method(self, b: Self, c: Self) -> Self {
                unsafe {
                    <$type>::new(paste!{[<_mm $bits _$method _ps>]}(self.0, b.0, c.0))
                }
            }
        }
    };
    
    ($trait:ident, $method:ident, $type:ty, $bits:literal, $precision:ident) => {
        impl $trait for $type {
            type Output = $type;
            
            #[inline(always)]
            fn $method(self, b: Self, c: Self) -> Self {
                unsafe {
                    <$type>::new(paste!{[<_mm $bits _$method _$precision>]}(self.0, b.0, c.0))
                }
            }
        }
    };
}

//type f32x16 = __m512;
//type h32x16 = __m256bh;
//type h32x32 = __m512bh;
