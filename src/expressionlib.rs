use std::ops::{Add, Mul, Sub, Div, AddAssign};
use itertools::zip_eq;

type TensorIdx = Vec<usize>;
type TensorShape = Vec<usize>;

struct AxisIterator<'a, T: Expression> {
    expr: &'a T,
    cur: TensorIdx,
    axis: usize
}

impl<'a, T: Expression> Iterator for AxisIterator<'a, T> {
    type Item = T::ResultType;
    fn next(&mut self) -> Option<Self::Item> {
        let shape = self.expr.shape();
        let iteration_complete = (shape[self.axis] - 1) == self.cur[self.axis];
        if !iteration_complete {
            let item = self.expr.eval(&self.cur);
            let mut elt = self.cur.get_mut(self.axis)?;
            *elt += 1;
            Some(item)
        }
        else {
            None
        }
    }
}

trait Expression {
    type ResultType;
    fn shape<'a>(&'a self) -> &'a TensorShape;
    fn eval(&self, idx: &TensorIdx) -> Self::ResultType;
    fn iter_stride<'a>(&'a self, sub_idx: &TensorIdx, axis: usize) -> AxisIterator<'a, Self> where Self: Sized {
        let shape = self.shape();
       
        if axis >= shape.len() {
            panic!("Can't iterate over axis outside of shape length");
        }

        let mut cur: Vec<usize> = Vec::new();

        for i in 0..axis {
            cur.push(sub_idx[i]);
        }

        cur.push(0);

        for i in (axis+1)..shape.len() {
            cur.push(sub_idx[i - 1]);
        }
        
        AxisIterator {
            expr: &self,
            cur: cur,
            axis: axis
        }
    }
}

trait Tensor {
    type DType;
    fn from_expr<T: Expression>(proxy: ExpressionProxy<T>) -> Self;
}
   
struct ElementIterator<'a, T: Expression> {
    expr: &'a T,
    cur: TensorIdx
}

impl<'a, T: Expression> Iterator for ElementIterator<'a, T> {
    type Item = TensorIdx;
    fn next(&mut self) -> Option<Self::Item> {
        let shape = self.expr.shape();

        let iteration_complete = 
            zip_eq(shape.iter(), self.cur.iter()).
            fold(true, |acc, (&x, &y)| (x - 1 == y) && acc);
        
        if !iteration_complete {
            let result = self.cur.clone(); 
            for (dim, idx) in self.cur.iter_mut().enumerate().rev() {
                *idx += 1;
                if *idx == shape[dim] {
                    *idx = 0 
                }
                else {
                    break
                }
            }
            Some(result)
        }
        else {
            None
        }
    }
}

enum ExpressionProxy<T: Expression> {
    Expr(T),
    ShapeError()
}

impl<T: Expression> ExpressionProxy<T> {
    fn iter<'a>(&'a self) -> ElementIterator<'a, T> {
        match &*self {
            Self::Expr(expression) => {
                ElementIterator {
                    expr: &expression,
                    cur: vec![0usize; expression.shape().len()]
                }
            }
            _ => panic!("Can't iterate over expression with ShapeError")
        }
    }
}

macro_rules! create_binary_expression {
    ($opname:ident, $op:tt, $trait:ident, $method:ident) => {
        struct $opname<L: Expression, R: Expression> {
            left: L, 
            right: R,
            shape: TensorShape
        }
        
        impl<L: Expression, R: Expression> Expression for $opname<L, R> where
            <L as Expression>::ResultType: $trait<<R as Expression>::ResultType>
        {
            type ResultType = 
                < <L as Expression>::ResultType as $trait<<R as Expression>::ResultType> >::Output;

            fn shape<'a>(&'a self) -> &'a TensorShape { &self.shape }

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

struct Square<T: Expression> {
    expr: T,
    shape: TensorShape
}

impl<T: Expression> Expression for Square<T> where
    <T as Expression>::ResultType: Mul< <T as Expression>::ResultType >+Copy,
{
    type ResultType = 
        < <T as Expression>::ResultType as Mul<<T as Expression>::ResultType> >::Output;

    fn shape<'a>(&'a self) -> &'a TensorShape { &self.shape }

    fn eval(&self, idx: &TensorIdx) -> Self::ResultType {
        let x = self.expr.eval(&idx);
        x * x
    }
}

struct ReduceSum<T: Expression> {
    expr: T,
    shape: TensorShape,
    axis: usize
}

impl<T: Expression> Expression for ReduceSum<T> where 
    f32: AddAssign<<T as Expression>::ResultType>
{
    type ResultType = f32;

    fn shape<'a>(&'a self) -> &'a TensorShape { &self.shape }

    fn eval(&self, idx: &TensorIdx) -> f32 {
        self.expr.
            iter_stride(idx, self.axis).
            fold(0f32, |mut acc, item| { 
                acc += item;
                acc
            })
    }
}

mod api {
    use super::*;

    fn square<T: Expression>(expr_proxy: ExpressionProxy<T>) -> ExpressionProxy<Square<T>>
    where
        <T as Expression>::ResultType: Mul< <T as Expression>::ResultType >+Copy
    {
        if let ExpressionProxy::Expr(expr) = expr_proxy {
            let shape = expr.shape().clone();
            ExpressionProxy::Expr(Square{expr, shape}) 
        }
        else {
            ExpressionProxy::ShapeError()
        }
    } 
    
    fn reduce_sum<T: Expression>(expr_proxy: ExpressionProxy<T>, axis: usize) -> ExpressionProxy<ReduceSum<T>>
    where
        f32: AddAssign<<T as Expression>::ResultType>
    { 
        if let ExpressionProxy::Expr(expr) = expr_proxy {
            ExpressionProxy::Expr(ReduceSum {
                expr: expr, 
                shape: Vec::<usize>::new(), 
                axis: axis
            }) 
        }
        else {
            ExpressionProxy::ShapeError()
        }
    }
}


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
