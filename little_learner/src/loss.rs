use std::{
    iter::Sum,
    ops::{Add, Mul, Neg},
};

use crate::{
    auto_diff::{Differentiable, RankedDifferentiable},
    scalar::Scalar,
    traits::{One, Zero},
};

pub fn square<A>(x: &A) -> A
where
    A: Mul<Output = A> + Clone,
{
    x.clone() * x.clone()
}

pub fn elementwise_mul<A, const RANK: usize>(
    x: &RankedDifferentiable<A, RANK>,
    y: &RankedDifferentiable<A, RANK>,
) -> RankedDifferentiable<A, RANK>
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Clone + Default,
{
    RankedDifferentiable::map2(x, y, &|x, y| x.clone() * y.clone())
}

pub fn dot_unranked<A>(x: &Differentiable<A>, y: &Differentiable<A>) -> Differentiable<A>
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Clone + Default,
{
    Differentiable::map2(x, y, &|x, y| x.clone() * y.clone())
}

fn squared_2<A, const RANK: usize>(
    x: &RankedDifferentiable<A, RANK>,
) -> RankedDifferentiable<A, RANK>
where
    A: Mul<Output = A> + Copy + Default,
{
    RankedDifferentiable::map2(x, x, &|x, y| x.clone() * y.clone())
}

fn sum_2<A>(x: RankedDifferentiable<A, 1>) -> Scalar<A>
where
    A: Sum<A> + Clone + Add<Output = A> + Zero,
{
    RankedDifferentiable::to_vector(x)
        .into_iter()
        .map(|x| x.to_scalar())
        .sum()
}

fn l2_norm_2<A>(
    prediction: &RankedDifferentiable<A, 1>,
    data: &RankedDifferentiable<A, 1>,
) -> Scalar<A>
where
    A: Sum<A> + Mul<Output = A> + Copy + Default + Neg<Output = A> + Add<Output = A> + Zero + Neg,
{
    let diff = RankedDifferentiable::map2(prediction, data, &|x, y| x.clone() - y.clone());
    sum_2(squared_2(&diff))
}

pub fn l2_loss_2<A, F, Params, const N: usize>(
    target: F,
    data_xs: RankedDifferentiable<A, N>,
    data_ys: RankedDifferentiable<A, 1>,
    params: Params,
) -> Scalar<A>
where
    F: Fn(RankedDifferentiable<A, N>, Params) -> RankedDifferentiable<A, 1>,
    A: Sum<A> + Mul<Output = A> + Copy + Default + Neg<Output = A> + Add<Output = A> + Zero,
{
    let pred_ys = target(data_xs, params);
    l2_norm_2(&pred_ys, &data_ys)
}

pub fn predict_line_2<A>(
    xs: RankedDifferentiable<A, 1>,
    theta: &[RankedDifferentiable<A, 0>; 2],
) -> RankedDifferentiable<A, 1>
where
    A: Mul<Output = A> + Add<Output = A> + Sum<<A as Mul>::Output> + Copy + Default + One + Zero,
{
    let xs = RankedDifferentiable::to_vector(xs)
        .into_iter()
        .map(|v| v.to_scalar());
    let mut result = vec![];
    for x in xs {
        let left_arg = RankedDifferentiable::of_vector(vec![
            RankedDifferentiable::of_scalar(x.clone()),
            RankedDifferentiable::of_scalar(<Scalar<A> as One>::one()),
        ]);
        let dotted = RankedDifferentiable::of_scalar(
            RankedDifferentiable::to_vector(elementwise_mul(
                &left_arg,
                &RankedDifferentiable::of_vector(theta.to_vec()),
            ))
            .iter()
            .map(|x| (*x).clone().to_scalar())
            .sum(),
        );
        result.push(dotted);
    }
    RankedDifferentiable::of_vector(result)
}

pub fn predict_line_2_unranked<A>(
    xs: RankedDifferentiable<A, 1>,
    theta: &[Differentiable<A>; 2],
) -> RankedDifferentiable<A, 1>
where
    A: Mul<Output = A> + Add<Output = A> + Sum<<A as Mul>::Output> + Copy + Default + One + Zero,
{
    let xs = RankedDifferentiable::to_vector(xs)
        .into_iter()
        .map(|v| v.to_scalar());
    let mut result = vec![];
    for x in xs {
        let left_arg = RankedDifferentiable::of_vector(vec![
            RankedDifferentiable::of_scalar(x.clone()),
            RankedDifferentiable::of_scalar(<Scalar<A> as One>::one()),
        ]);
        let dotted = RankedDifferentiable::of_scalar(
            dot_unranked(
                left_arg.to_unranked_borrow(),
                &Differentiable::of_vec(theta.to_vec()),
            )
            .into_vector()
            .into_iter()
            .map(|x| x.into_scalar())
            .sum(),
        );
        result.push(dotted);
    }
    RankedDifferentiable::of_vector(result)
}

pub fn predict_quadratic<A>(
    xs: RankedDifferentiable<A, 1>,
    theta: &[RankedDifferentiable<A, 0>; 3],
) -> RankedDifferentiable<A, 1>
where
    A: Mul<Output = A> + Add<Output = A> + Sum + Default + One + Zero + Clone,
{
    RankedDifferentiable::map(xs, &mut |x| {
        let x_powers = vec![Scalar::make(A::one()), x.clone(), square(&x)];
        let x_powers = RankedDifferentiable::of_vector(
            x_powers
                .into_iter()
                .map(RankedDifferentiable::of_scalar)
                .collect(),
        );
        RankedDifferentiable::to_vector(elementwise_mul(
            &x_powers,
            &RankedDifferentiable::of_vector(theta.to_vec()),
        ))
        .into_iter()
        .map(|x| x.to_scalar())
        .sum()
    })
}

pub fn predict_quadratic_unranked<A>(
    xs: RankedDifferentiable<A, 1>,
    theta: &[Differentiable<A>; 3],
) -> RankedDifferentiable<A, 1>
where
    A: Mul<Output = A> + Add<Output = A> + Sum + Default + One + Zero + Clone,
{
    RankedDifferentiable::map(xs, &mut |x| {
        let x_powers = vec![Scalar::make(A::one()), x.clone(), square(&x)];
        let x_powers = RankedDifferentiable::of_vector(
            x_powers
                .into_iter()
                .map(RankedDifferentiable::of_scalar)
                .collect(),
        );
        dot_unranked(
            x_powers.to_unranked_borrow(),
            &Differentiable::of_vec(theta.to_vec()),
        )
        .attach_rank::<1>()
        .expect("wanted a tensor1")
        .to_vector()
        .into_iter()
        .map(|x| x.to_scalar())
        .sum()
    })
}

// The parameters are: a tensor1 of length 2 (to be dotted with the input), and a scalar (to translate).
pub fn predict_plane<A>(
    xs: RankedDifferentiable<A, 2>,
    theta: &[Differentiable<A>; 2],
) -> RankedDifferentiable<A, 1>
where
    A: Mul<Output = A> + Add<Output = A> + Sum + Default + One + Zero + Clone,
{
    if theta[0].rank() != 1 {
        panic!("theta0 must be of rank 1, got: {}", theta[0].rank())
    }
    let theta0 = RankedDifferentiable::of_vector(
        theta[0]
            .borrow_vector()
            .iter()
            .map(|v| RankedDifferentiable::of_scalar(v.borrow_scalar().clone()))
            .collect::<Vec<_>>(),
    );
    let theta1 = theta[1].borrow_scalar().clone();
    let dotted: Vec<_> = xs
        .to_vector()
        .into_iter()
        .map(|point| sum_2(elementwise_mul(&theta0, &point)))
        .map(|x| RankedDifferentiable::of_scalar(x + theta1.clone()))
        .collect();
    RankedDifferentiable::of_vector(dotted)
}
