use std::ops::{Add, Mul, Sub, Div};
use itertools::zip_eq;

enum TensorIdx {
    Rank0,
    Rank1(usize)
}

trait Expression {
    type ResultType;
    fn shape<'a>(&'a self) -> &'a Vec<usize>;
    fn eval(&self, idx: &TensorIdx) -> Self::ResultType;
}

enum ExpressionProxy<T: Expression> {
    Expr(T),
    ShapeError()
}

macro_rules! create_binary_expression {
    ($opname:ident, $op:tt, $trait:ident, $method:ident) => {
        struct $opname<L: Expression, R: Expression> {
            left: L, 
            right: R,
            shape: Vec<usize>
        }
        
        impl<L: Expression, R: Expression> Expression for $opname<L, R> where
            <L as Expression>::ResultType: $trait<<R as Expression>::ResultType>
        {
            type ResultType = 
                < <L as Expression>::ResultType as $trait<<R as Expression>::ResultType> >::Output;

            fn shape<'a>(&'a self) -> &'a Vec<usize> { &self.shape }

            fn eval(&self, idx: &TensorIdx) -> Self::ResultType {
                self.left.eval(&idx) $op self.right.eval(&idx)
            }

        }

        impl<L, R> $trait<ExpressionProxy<R>> for ExpressionProxy<L> where
            L: Expression,
            R: Expression,
            <L as Expression>::ResultType: $trait<<R as Expression>::ResultType>
        {
            type Output = ExpressionProxy<$opname<L, R>>;
            fn $method(self, other: ExpressionProxy<R>) -> Self::Output {
                if let (ExpressionProxy::<L>::Expr(e), ExpressionProxy::<R>::Expr(f)) = (self, other) {

                    let compatible_shapes: bool = 
                        (e.shape().len() == f.shape().len()) &&
                        zip_eq(e.shape().iter(), f.shape().iter()).
                            fold(true, |acc, (x, y)| acc && (x == y));

                    if compatible_shapes {
                        let shape = e.shape().clone();
                        ExpressionProxy::<$opname<L, R>>::Expr(
                            $opname::<L, R>{
                                left: e, 
                                right: f, 
                                shape: shape
                            }
                        )
                    }
                    else {
                        ExpressionProxy::<$opname<L, R>>::ShapeError()
                    }
                }
                else {
                    ExpressionProxy::<$opname<L, R>>::ShapeError()
                }
            }
        }
    }
}

create_binary_expression!(Plus, +, Add, add);
create_binary_expression!(Times, *, Mul, mul);
create_binary_expression!(Minus, -, Sub, sub);
create_binary_expression!(Divide, /, Div, div);


/*
struct Square<T: Expression> {
    expr: T,
    shape: Vec<usize>
}

impl<T: Expression> Expression for Square<T> where
    <T as Expression>::ResultType: Mul
{
    type ResultType = T::ResultType;

    fn shape<'a>(&'a self) -> &'a Vec<usize> { &self.shape }

    fn eval(&self, idx: &TensorIdx) -> Self::ResultType {
        let x = self.expr.eval(&idx);
        x * x
    }
}
*/

/*
struct ReduceSum<T: Expression>(T);
impl<T: Expression> Expression for ReduceSum<T> {
    type ResultType = T::ResultType;
    fn eval(&self, idx: TensorIdx) -> Self::ResultType {
        if let TensorIdx::Rank0 = idx {
            let mut acc: f32 = 0.0
            for idx in (0..self.0.len()) {
                acc += self.0.eval(TensorIdx::Rank1(idx)) 
            }
        }

        let x = self.0.eval(idx);
        x * x
    }
}

/*
struct ReduceSum<T: Expression>(T);

impl<T: Expression> Expression for ReduceSum<T> {
    type ResultType = T::ResultType
}
*/

impl<T, const MMBLOCKS: usize> Expression for SimdVecImpl<T, MMBLOCKS> {
    type ResultType = T;
    fn eval(&self, idx: usize) -> T {
        self.chunks[idx]
    }
}

mod api {
    fn square<T: Expression>(expr: T) { ExpressionProxy<Square<T>>(expr) } }
*/

// I want to guarantee that when I have a feature expression and I evaluate it on a SimdVector that 
// the entire chain of operations is executed in sequence. 

// When do I want to force evaluation? What are my options?
//  1. On an explicit call to some function e.g. forward()
//  2. When we attempt to access elements
//  3. On assignment of the expression to a Tensor.

/*

tuple_impl!(usize)
*/

/*
impl<T: Tensor> Expression for ExpressionProxy<T> {
    type ResultType

    fn eval(&self) {

    }
}*/

/*  This is a convenience macro for expressing the type of
    an array of arrays of arrays ... of type T
macro_rules! array_type {
    ($x:ty; $y:literal) => { [$x; $y] };
    ($x:ty; $y:literal, $($z:literal),+) => { [array_type!($x; $($z)+); $y] };
}

macro_rules! init_array {
    ($const:literal; $y:literal) => { [$const; $y] };
    ($const:literal; $y:literal, $($z:literal),+) => { [array_type!($const; $($z)+); $y] };
}
 */
