use std::{
    iter::Sum,
    ops::{Add, Mul, Neg},
};

use crate::{
    auto_diff::{of_scalar, to_scalar, Differentiable},
    scalar::Scalar,
    traits::{One, Zero},
};

pub fn square<A>(x: &A) -> A
where
    A: Mul<Output = A> + Clone,
{
    x.clone() * x.clone()
}

pub fn dot_2<A, const RANK: usize>(
    x: &Differentiable<A, RANK>,
    y: &Differentiable<A, RANK>,
) -> Differentiable<A, RANK>
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Clone + Default,
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
    A: Sum<A> + Clone + Add<Output = A> + Zero,
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

pub fn predict_line_2<A>(
    xs: Differentiable<A, 1>,
    theta: Differentiable<A, 1>,
) -> Differentiable<A, 1>
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
        let dotted = of_scalar(
            Differentiable::to_vector(dot_2(&left_arg, &theta))
                .iter()
                .map(|x| to_scalar((*x).clone()))
                .sum(),
        );
        result.push(dotted);
    }
    Differentiable::of_vector(result)
}

pub fn predict_quadratic<A>(
    xs: Differentiable<A, 1>,
    theta: Differentiable<A, 1>,
) -> Differentiable<A, 1>
where
    A: Mul<Output = A> + Add<Output = A> + Sum + Default + One + Zero + Clone,
{
    Differentiable::map(xs, &mut |x| {
        let x_powers = vec![Scalar::make(A::one()), x.clone(), square(&x)];
        let x_powers = Differentiable::of_vector(x_powers.into_iter().map(of_scalar).collect());
        sum_2(dot_2(&x_powers, &theta))
    })
}
