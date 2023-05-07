use crate::scalar::Scalar;
use ordered_float::NotNan;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg};

pub trait Exp {
    #[must_use]
    fn exp(self) -> Self;
}

impl Exp for NotNan<f64> {
    fn exp(self) -> Self {
        NotNan::new(f64::exp(self.into_inner())).expect("expected a non-NaN")
    }
}

pub trait Sqrt {
    #[must_use]
    fn sqrt(self) -> Self;
}

impl Sqrt for NotNan<f64> {
    fn sqrt(self) -> Self {
        NotNan::new(f64::sqrt(self.into_inner())).expect("expected a non-NaN")
    }
}

pub trait Zero {
    #[must_use]
    fn zero() -> Self;
}

pub trait One {
    #[must_use]
    fn one() -> Self;
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

impl Zero for NotNan<f64> {
    fn zero() -> Self {
        NotNan::new(0.0).unwrap()
    }
}

impl One for NotNan<f64> {
    fn one() -> Self {
        NotNan::new(1.0).unwrap()
    }
}

pub trait NumLike:
    One
    + Zero
    + Exp
    + Add<Output = Self>
    + AddAssign
    + Neg<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Sum
    + Sqrt
    + Clone
    + Sized
    + PartialEq
    + Eq
{
}

impl NumLike for NotNan<f64> {}

impl<A> NumLike for Scalar<A> where A: NumLike {}
