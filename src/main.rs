mod auto_diff;
mod expr_syntax_tree;

use std::iter::Sum;
use std::ops::{Mul, Sub};

type Point<A, const N: usize> = [A; N];

type Parameters<A, const N: usize, const M: usize> = [Point<A, N>; M];

#[macro_export]
macro_rules! tensor {
    ($x:ty , $i: expr) => {[$x; $i]};
    ($x:ty , $i: expr, $($is:expr),+) => {[tensor!($x, $($is),+); $i]};
}

pub trait Extensible1<A> {
    fn apply<F>(&self, other: &A, op: &F) -> Self
    where
        F: Fn(&A, &A) -> A;
}

pub trait Extensible2<A> {
    fn apply<F>(&self, other: &Self, op: &F) -> Self
    where
        F: Fn(&A, &A) -> A;
}

impl<A, T, const N: usize> Extensible1<A> for [T; N]
where
    T: Extensible1<A> + Copy + Default,
{
    fn apply<F>(&self, other: &A, op: &F) -> Self
    where
        F: Fn(&A, &A) -> A,
    {
        let mut result = [Default::default(); N];
        for (i, coord) in self.iter().enumerate() {
            result[i] = T::apply(coord, other, op);
        }
        result
    }
}

impl<A, T, const N: usize> Extensible2<A> for [T; N]
where
    T: Extensible2<A> + Copy + Default,
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
macro_rules! extensible1 {
    ($x: ty) => {
        impl Extensible1<$x> for $x {
            fn apply<F>(&self, other: &$x, op: &F) -> Self
            where
                F: Fn(&Self, &Self) -> Self,
            {
                op(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! extensible2 {
    ($x: ty) => {
        impl Extensible2<$x> for $x {
            fn apply<F>(&self, other: &Self, op: &F) -> Self
            where
                F: Fn(&Self, &Self) -> Self,
            {
                op(self, other)
            }
        }
    };
}

extensible1!(u8);
extensible1!(f64);

extensible2!(u8);
extensible2!(f64);

pub fn extension1<T, A, F>(t1: &T, t2: &A, op: F) -> T
where
    T: Extensible1<A>,
    F: Fn(&A, &A) -> A,
{
    t1.apply::<F>(t2, &op)
}

pub fn extension2<T, A, F>(t1: &T, t2: &T, op: F) -> T
where
    T: Extensible2<A>,
    F: Fn(&A, &A) -> A,
{
    t1.apply::<F>(t2, &op)
}

fn dot_points<A: Mul, const N: usize>(x: &Point<A, N>, y: &Point<A, N>) -> A
where
    A: Sum<<A as Mul>::Output> + Copy + Default + Mul<Output = A> + Extensible2<A>,
{
    extension2(x, y, |&x, &y| x * y).into_iter().sum()
}

fn dot<A, const N: usize, const M: usize>(x: &Point<A, N>, y: &Parameters<A, N, M>) -> Point<A, M>
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Copy + Default + Extensible2<A>,
{
    let mut result = [Default::default(); M];
    for (i, coord) in y.iter().map(|y| dot_points(x, y)).enumerate() {
        result[i] = coord;
    }
    result
}

fn sum<A, const N: usize>(x: &tensor!(A, N)) -> A
where
    A: Sum<A> + Copy,
{
    A::sum(x.iter().cloned())
}

fn squared<A, const N: usize>(x: &tensor!(A, N)) -> tensor!(A, N)
where
    A: Mul<Output = A> + Extensible2<A> + Copy + Default,
{
    extension2(x, x, |&a, &b| (a * b))
}

fn l2_norm<A, const N: usize>(prediction: &tensor!(A, N), data: &tensor!(A, N)) -> A
where
    A: Sum<A> + Mul<Output = A> + Extensible2<A> + Copy + Default + Sub<Output = A>,
{
    let diff = extension2(prediction, data, |&x, &y| x - y);
    sum(&squared(&diff))
}

pub fn l2_loss<A, F, Params, const N: usize>(
    target: F,
    data_xs: &tensor!(A, N),
    data_ys: &tensor!(A, N),
    params: &Params,
) -> A
where
    F: Fn(&tensor!(A, N), &Params) -> tensor!(A, N),
    A: Sum<A> + Mul<Output = A> + Extensible2<A> + Copy + Default + Sub<Output = A>,
{
    let pred_ys = target(data_xs, params);
    l2_norm(&pred_ys, data_ys)
}

trait One {
    const ONE: Self;
}

impl One for f64 {
    const ONE: f64 = 1.0;
}

fn predict_line<A, const N: usize>(xs: &tensor!(A, N), theta: &tensor!(A, 2)) -> tensor!(A, N)
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Copy + Default + Extensible2<A> + One,
{
    let mut result: tensor!(A, N) = [Default::default(); N];
    for (i, &x) in xs.iter().enumerate() {
        result[i] = dot(&[x, One::ONE], &[*theta])[0];
    }
    result
}

fn main() {
    let loss = l2_loss(
        predict_line,
        &[2.0, 1.0, 4.0, 3.0],
        &[1.8, 1.2, 4.2, 3.3],
        &[0.0099, 0.0],
    );
    println!("{:?}", loss);
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
        assert_eq!(extension1(&x, &7, |x, y| x + y), [9]);
        let y: tensor!(u8, 1) = [7];
        assert_eq!(extension2(&x, &y, |x, y| x + y), [9]);

        let x: tensor!(u8, 3) = [5, 6, 7];
        assert_eq!(extension1(&x, &2, |x, y| x + y), [7, 8, 9]);
        let y: tensor!(u8, 3) = [2, 0, 1];
        assert_eq!(extension2(&x, &y, |x, y| x + y), [7, 6, 8]);

        let x: tensor!(u8, 2, 3) = [[4, 6, 7], [2, 0, 1]];
        assert_eq!(extension1(&x, &2, |x, y| x + y), [[6, 8, 9], [4, 2, 3]]);
        let y: tensor!(u8, 2, 3) = [[1, 2, 2], [6, 3, 1]];
        assert_eq!(extension2(&x, &y, |x, y| x + y), [[5, 8, 9], [8, 3, 2]]);
    }

    #[test]
    fn test_l2_norm() {
        assert_eq!(
            l2_norm(&[4.0, -3.0, 0.0, -4.0, 3.0], &[0.0, 0.0, 0.0, 0.0, 0.0]),
            50.0
        )
    }

    #[test]
    fn test_l2_loss() {
        let loss = l2_loss(
            predict_line,
            &[2.0, 1.0, 4.0, 3.0],
            &[1.8, 1.2, 4.2, 3.3],
            &[0.0, 0.0],
        );
        assert_eq!(loss, 33.21);

        let loss = l2_loss(
            predict_line,
            &[2.0, 1.0, 4.0, 3.0],
            &[1.8, 1.2, 4.2, 3.3],
            &[0.0099, 0.0],
        );
        assert_eq!((100.0 * loss).round() / 100.0, 32.59);
    }
}
