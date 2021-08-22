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

/*
Questions:
1. How do I let rust enable different instructions when compiling for different arch?
2. How do I want to switch between different instruction sets?
*/


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

