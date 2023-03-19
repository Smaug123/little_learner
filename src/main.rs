use std::iter::Sum;
use std::ops::Mul;

type Point<A, const N: usize> = [A; N];

type Parameters<A, const N: usize, const M: usize> = [Point<A, N>; M];

fn dot_points<A: Mul, const N: usize>(x: &Point<A, N>, y: &Point<A, N>) -> A
where
    A: Sum<<A as Mul>::Output> + Copy,
{
    x.iter().zip(y).map(|(&x, &y)| x * y).sum()
}

fn dot<A, const N: usize, const M: usize>(x: &Point<A, N>, y: &Parameters<A, N, M>) -> Point<A, M>
where
    A: Mul<A> + Sum<<A as Mul>::Output> + Copy + Default,
{
    let mut result = [Default::default(); M];
    for (i, coord) in y.iter().map(|y| dot_points(x, y)).enumerate() {
        result[i] = coord;
    }
    result
}

fn line<A, const N: usize>(x: &Point<A, N>, theta: &Parameters<A, N, 1>) -> Point<A, 1>
where
    A: Mul<A> + Sum<<A as Mul>::Output> + Copy + Default,
{
    dot(x, theta)
}

//fn data_set() -> ([f64; 4], [f64; 4]) {
//    ([2.0, 1.0, 4.0, 3.0], [1.8, 1.2, 4.2, 3.3])
//}

fn linear_params_2d<A>(m: A, c: A) -> Parameters<A, 2, 1> {
    [[c, m]]
}

#[macro_export]
macro_rules! tensor {
    ($x:ty , $i: expr) => {[$x; $i]};
    ($x:ty , $i: expr, $($is:expr),+) => {[tensor!($x, $($is),+); $i]};
}

pub trait Extensible<A> {
    fn apply<F>(&self, other: &Self, op: &F) -> Self
    where
        F: Fn(&A, &A) -> A;
}

impl<A, T, const N: usize> Extensible<A> for [T; N]
where
    T: Extensible<A> + Copy + Default,
{
    fn apply<F>(&self, other: &Self, op: &F) -> Self
    where
        F: Fn(&A, &A) -> A,
    {
        let mut result = [Default::default(); N];
        for (i, coord) in self.iter().enumerate() {
            result[i] = T::apply(coord, &other[i], op);
        }
        result
    }
}

#[macro_export]
macro_rules! extensible {
    ($x: ty) => {
        impl Extensible<$x> for $x {
            fn apply<F>(&self, other: &Self, op: &F) -> Self
            where
                F: Fn(&Self, &Self) -> Self,
            {
                op(self, other)
            }
        }
    };
}

extensible!(u8);
extensible!(f64);

pub fn extension<T, A, F>(t1: &T, t2: &T, op: F) -> T
where
    T: Extensible<A>,
    F: Fn(&A, &A) -> A,
{
    t1.apply::<F>(t2, &op)
}

fn main() {
    let y = line(&[1.0, 7.3], &linear_params_2d(3.0, 1.0));
    println!("{:?}", y);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_type() {
        let _: tensor!(f64, 1, 2, 3) = [[[1.0, 3.0, 6.0], [-1.3, -30.0, -0.0]]];
    }

    #[test]
    fn test_extension() {
        let x: tensor!(u8, 1) = [2];
        let y: tensor!(u8, 1) = [7];
        assert_eq!(extension(&x, &y, |x, y| x + y), [9]);

        let x: tensor!(u8, 3) = [5, 6, 7];
        let y: tensor!(u8, 3) = [2, 0, 1];
        assert_eq!(extension(&x, &y, |x, y| x + y), [7, 6, 8]);

        let x: tensor!(u8, 2, 3) = [[4, 6, 7], [2, 0, 1]];
        let y: tensor!(u8, 2, 3) = [[1, 2, 2], [6, 3, 1]];
        assert_eq!(extension(&x, &y, |x, y| x + y), [[5, 8, 9], [8, 3, 2]]);
    }
}
