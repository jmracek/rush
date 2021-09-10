use core::ops::{Add, Sub, Div, Mul};
use std::iter::{IntoIterator, FromIterator};

pub trait VectorArithmetic:
    Add<Output=Self> + 
    Mul<Output=Self> +
    Div<Output=Self> +
    Sub<Output=Self> +
    Mul<<Self as VectorArithmetic>::DType, Output=Self> +
    Div<<Self as VectorArithmetic>::DType, Output=Self> +
    Default
{
    type DType;
}

pub trait Vector:
    VectorArithmetic<DType=<Self as Vector>::DType> +
    FromIterator<<Self as Vector>::DType>
where
    for<'a> &'a Self: IntoIterator< Item=<Self as Vector>::DType >,
{
    type DType;

    fn distance(&self, other: &Self) -> <Self as Vector>::DType;
    fn dot(&self, other: &Self) -> <Self as Vector>::DType;
    fn dimension(&self) -> usize;
}
