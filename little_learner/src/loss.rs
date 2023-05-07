use std::{
    iter::Sum,
    ops::{Add, Mul, Neg},
};

use crate::auto_diff::Differentiable;
use crate::smooth::smooth;
use crate::traits::{NumLike, Sqrt};
use crate::{
    auto_diff::{DifferentiableTagged, RankedDifferentiable},
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
    RankedDifferentiable::map2(x, y, &mut |x, y| x.clone() * y.clone())
}

pub fn dot_unranked_tagged<A, Tag1, Tag2, Tag3, F>(
    x: &DifferentiableTagged<A, Tag1>,
    y: &DifferentiableTagged<A, Tag2>,
    mut combine_tags: F,
) -> DifferentiableTagged<A, Tag3>
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Clone + Default,
    F: FnMut(Tag1, Tag2) -> Tag3,
    Tag1: Clone,
    Tag2: Clone,
{
    DifferentiableTagged::map2_tagged(x, y, &mut |x, tag1, y, tag2| {
        (x.clone() * y.clone(), combine_tags(tag1, tag2))
    })
}

pub fn dot_unranked<A>(x: &Differentiable<A>, y: &Differentiable<A>) -> Differentiable<A>
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Clone + Default,
{
    dot_unranked_tagged(x, y, |(), ()| ())
}

fn squared_2<A, const RANK: usize>(
    x: &RankedDifferentiable<A, RANK>,
) -> RankedDifferentiable<A, RANK>
where
    A: Mul<Output = A> + Copy + Default,
{
    RankedDifferentiable::map2(x, x, &mut |x, y| x.clone() * y.clone())
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
    let diff = RankedDifferentiable::map2(prediction, data, &mut |x, y| x.clone() - y.clone());
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
                &DifferentiableTagged::of_vec(theta.to_vec()),
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
            &DifferentiableTagged::of_vec(theta.to_vec()),
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

pub struct Predictor<F, Inflated, Deflated, Params> {
    pub predict: F,
    pub inflate: fn(Deflated) -> Inflated,
    pub deflate: fn(Inflated) -> Deflated,
    pub update: fn(Inflated, &Deflated, Params) -> Inflated,
}

#[derive(Clone)]
pub struct NakedHypers<A> {
    pub learning_rate: A,
}

pub const fn naked_predictor<F, A>(
    f: F,
) -> Predictor<F, Differentiable<A>, Differentiable<A>, NakedHypers<A>>
where
    A: NumLike,
{
    Predictor {
        predict: f,
        inflate: |x| x,
        deflate: |x| x,

        update: |theta, delta, hyper| {
            let learning_rate = Scalar::make(hyper.learning_rate);
            Differentiable::map2(&theta, delta, &mut |theta, delta| {
                theta.clone() - delta.clone() * learning_rate.clone()
            })
        },
    }
}

#[derive(Clone)]
pub struct RmsHyper<A> {
    pub stabilizer: A,
    pub beta: A,
    pub learning_rate: A,
}

pub const fn rms_predictor<F, A>(
    f: F,
) -> Predictor<F, DifferentiableTagged<A, A>, Differentiable<A>, RmsHyper<A>>
where
    A: NumLike,
{
    Predictor {
        predict: f,
        inflate: |x| x.map_tag(&mut |()| A::zero()),
        deflate: |x| x.map_tag(&mut |_| ()),
        update: |theta, delta, hyper| {
            DifferentiableTagged::map2_tagged(
                &theta,
                delta,
                &mut |theta, smoothed_grad, delta, ()| {
                    let r = smooth(
                        Scalar::make(hyper.beta.clone()),
                        &Differentiable::of_scalar(Scalar::make(smoothed_grad)),
                        &Differentiable::of_scalar(delta.clone() * delta.clone()),
                    )
                    .into_scalar();
                    let learning_rate = Scalar::make(hyper.learning_rate.clone())
                        / (r.sqrt() + Scalar::make(hyper.stabilizer.clone()));
                    (
                        theta.clone()
                            + -(delta.clone() * Scalar::make(hyper.learning_rate.clone())),
                        learning_rate.clone_real_part(),
                    )
                },
            )
        },
    }
}

#[derive(Clone)]
pub struct VelocityHypers<A> {
    pub learning_rate: A,
    pub mu: A,
}

pub const fn velocity_predictor<F, A>(
    f: F,
) -> Predictor<F, DifferentiableTagged<A, A>, Differentiable<A>, VelocityHypers<A>>
where
    A: NumLike,
{
    Predictor {
        predict: f,
        inflate: |x| x.map_tag(&mut |()| A::zero()),
        deflate: |x| x.map_tag(&mut |_| ()),
        update: |theta, delta, hyper| {
            DifferentiableTagged::map2_tagged(&theta, delta, &mut |theta, velocity, delta, ()| {
                let velocity = hyper.mu.clone() * velocity
                    + -(delta.clone_real_part() * hyper.learning_rate.clone());
                (theta.clone() + Scalar::make(velocity.clone()), velocity)
            })
        },
    }
}

#[cfg(test)]
mod test_loss {
    use crate::auto_diff::RankedDifferentiable;
    use crate::loss::{l2_loss_2, predict_line_2};
    use crate::scalar::Scalar;
    use crate::traits::Zero;

    #[test]
    fn loss_example() {
        let xs = [2.0, 1.0, 4.0, 3.0];
        let ys = [1.8, 1.2, 4.2, 3.3];
        let loss = l2_loss_2(
            predict_line_2,
            RankedDifferentiable::of_slice(&xs),
            RankedDifferentiable::of_slice(&ys),
            &[
                RankedDifferentiable::of_scalar(Scalar::zero()),
                RankedDifferentiable::of_scalar(Scalar::zero()),
            ],
        );

        assert_eq!(*loss.real_part(), 33.21);
    }
}
