#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod with_tensor;

use little_learner::auto_diff::{of_scalar, of_slice, to_scalar, Differentiable};
use little_learner::scalar::Scalar;
use little_learner::traits::{One, Zero};
use ordered_float::NotNan;

use std::iter::Sum;
use std::ops::{Add, Mul, Neg};

use crate::with_tensor::{l2_loss, predict_line};

fn dot_2<A, const RANK: usize>(
    x: &Differentiable<A, RANK>,
    y: &Differentiable<A, RANK>,
) -> Differentiable<A, RANK>
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Copy + Default,
{
    Differentiable::map2(x, y, &|x, y| x.clone() * y.clone())
}

fn squared_2<A, const RANK: usize>(x: &Differentiable<A, RANK>) -> Differentiable<A, RANK>
where
    A: Mul<Output = A> + Copy + Default,
{
    Differentiable::map2(x, x, &|x, y| x.clone() * y.clone())
}

fn sum_2<A>(x: Differentiable<A, 1>) -> Scalar<A>
where
    A: Sum<A> + Copy + Add<Output = A> + Zero,
{
    Differentiable::to_vector(x)
        .into_iter()
        .map(to_scalar)
        .sum()
}

fn l2_norm_2<A>(prediction: &Differentiable<A, 1>, data: &Differentiable<A, 1>) -> Scalar<A>
where
    A: Sum<A> + Mul<Output = A> + Copy + Default + Neg<Output = A> + Add<Output = A> + Zero + Neg,
{
    let diff = Differentiable::map2(prediction, data, &|x, y| x.clone() - y.clone());
    sum_2(squared_2(&diff))
}

pub fn l2_loss_2<A, F, Params>(
    target: F,
    data_xs: Differentiable<A, 1>,
    data_ys: Differentiable<A, 1>,
    params: Params,
) -> Scalar<A>
where
    F: Fn(Differentiable<A, 1>, Params) -> Differentiable<A, 1>,
    A: Sum<A> + Mul<Output = A> + Copy + Default + Neg<Output = A> + Add<Output = A> + Zero,
{
    let pred_ys = target(data_xs, params);
    l2_norm_2(&pred_ys, &data_ys)
}

fn predict_line_2<A>(xs: Differentiable<A, 1>, theta: Differentiable<A, 1>) -> Differentiable<A, 1>
where
    A: Mul<Output = A> + Add<Output = A> + Sum<<A as Mul>::Output> + Copy + Default + One + Zero,
{
    let xs = Differentiable::to_vector(xs)
        .into_iter()
        .map(|v| to_scalar(v));
    let mut result = vec![];
    for x in xs {
        let left_arg = Differentiable::of_vector(vec![
            of_scalar(x.clone()),
            of_scalar(<Scalar<A> as One>::one()),
        ]);
        let dotted = Differentiable::to_vector(dot_2(&left_arg, &theta));
        result.push(dotted[0].clone());
    }
    Differentiable::of_vector(result)
}

fn square<A>(x: &A) -> A
where
    A: Mul<Output = A> + Clone,
{
    x.clone() * x.clone()
}

fn main() {
    let loss = l2_loss(
        predict_line,
        &[2.0, 1.0, 4.0, 3.0],
        &[1.8, 1.2, 4.2, 3.3],
        &[0.0099, 0.0],
    );
    println!("{:?}", loss);

    let loss = l2_loss_2(
        predict_line_2,
        of_slice(&[2.0, 1.0, 4.0, 3.0]),
        of_slice(&[1.8, 1.2, 4.2, 3.3]),
        of_slice(&[0.0099, 0.0]),
    );
    println!("{}", loss);

    let input_vec = of_slice(&[NotNan::new(27.0).expect("not nan")]);

    let grad = Differentiable::grad(|x| Differentiable::map(x, &|x| square(&x)), input_vec);
    println!("{}", grad);
}
