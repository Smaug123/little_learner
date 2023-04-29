use ordered_float::NotNan;
use std::ops::{Add, AddAssign, Div, Mul, Neg};

pub trait Exp {
    fn exp(self) -> Self;
}

impl Exp for NotNan<f64> {
    fn exp(self) -> Self {
        NotNan::new(f64::exp(self.into_inner())).expect("expected a non-NaN")
    }
}

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
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
    + Sized
{
}

impl NumLike for NotNan<f64> {}
