use std::{
    iter::Sum,
    ops::{Add, Mul, Neg},
};

use crate::auto_diff::{Differentiable, RankedDifferentiableTagged};
use crate::ext::{sum, sum_1};
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

pub fn dot<A, Tag1, Tag2>(
    x: &RankedDifferentiableTagged<A, Tag1, 1>,
    y: &RankedDifferentiableTagged<A, Tag2, 1>,
) -> Scalar<A>
where
    A: Mul<Output = A> + Sum + Clone + Add<Output = A> + Zero,
{
    // Much sadness - find a way to get rid of these clones
    let x = x.map_tag(&mut |_| ());
    let y = y.map_tag(&mut |_| ());
    x.to_vector()
        .iter()
        .zip(y.to_vector().iter())
        .map(|(x, y)| x.clone().to_scalar() * y.clone().to_scalar())
        .sum()
}

fn squared_2<A, const RANK: usize>(
    x: &RankedDifferentiable<A, RANK>,
) -> RankedDifferentiable<A, RANK>
where
    A: Mul<Output = A> + Copy + Default,
{
    RankedDifferentiable::map2(x, x, &mut |x, y| x.clone() * y.clone())
}

fn l2_norm_2<A>(
    prediction: &RankedDifferentiable<A, 1>,
    data: &RankedDifferentiable<A, 1>,
) -> Scalar<A>
where
    A: Sum<A> + Mul<Output = A> + Copy + Default + Neg<Output = A> + Add<Output = A> + Zero + Neg,
{
    let diff = RankedDifferentiable::map2(prediction, data, &mut |x, y| x.clone() - y.clone());
    sum_1(squared_2(&diff)).into_scalar()
}

pub fn l2_loss_2<A, F, Params>(
    target: &mut F,
    data_xs: &Differentiable<A>,
    data_ys: RankedDifferentiable<A, 1>,
    params: Params,
) -> Scalar<A>
where
    F: FnMut(&Differentiable<A>, Params) -> RankedDifferentiable<A, 1>,
    A: Sum<A> + Mul<Output = A> + Copy + Default + Neg<Output = A> + Add<Output = A> + Zero,
{
    let pred_ys = target(data_xs, params);
    l2_norm_2(&pred_ys, &data_ys)
}

pub fn predict_line_2<A>(
    xs: &RankedDifferentiable<A, 1>,
    theta: [RankedDifferentiable<A, 0>; 2],
) -> RankedDifferentiable<A, 1>
where
    A: Mul<Output = A> + Add<Output = A> + Sum<<A as Mul>::Output> + Copy + Default + One + Zero,
{
    let xs = xs.to_unranked_borrow().borrow_vector()
        .into_iter()
        .map(|v| v.borrow_scalar());
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

/// The parameters are: a tensor1 of length 2 (to be dotted with the input), and a scalar (to translate).
///
/// # Panics
/// Panics if the input `theta` is not of rank 1 consisting of a tensor1 and a scalar.
pub fn predict_plane<A>(
    xs: &Differentiable<A>,
    theta: &[Differentiable<A>; 2],
) -> RankedDifferentiable<A, 1>
where
    A: Mul<Output = A> + Add<Output = A> + Sum + Default + One + Zero + Clone,
{
    assert_eq!(
        theta[0].rank(),
        1,
        "theta0 must be of rank 1, got: {}",
        theta[0].rank()
    );
    let theta0 = RankedDifferentiable::of_vector(
        theta[0]
            .borrow_vector()
            .iter()
            .map(|v| RankedDifferentiable::of_scalar(v.borrow_scalar().clone()))
            .collect::<Vec<_>>(),
    );
    let theta1 = theta[1].clone().attach_rank::<0>().unwrap();
    let dotted: Vec<_> = xs
        .borrow_vector()
        .into_iter()
        .map(|point| {
            sum(elementwise_mul(&theta0, &point.clone().attach_rank::<1>().unwrap()).to_unranked_borrow())
                .attach_rank::<0>()
                .unwrap()
        })
        .map(|x| x.map2(&theta1, &mut |x, theta| x.clone() + theta.clone()))
        .collect();
    RankedDifferentiable::of_vector(dotted)
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
            &mut |x, y| { predict_line_2(&x.attach_rank::<1>().unwrap(), y) },
            RankedDifferentiable::of_slice(&xs).to_unranked_borrow(),
            RankedDifferentiable::of_slice(&ys),
            [
                RankedDifferentiable::of_scalar(Scalar::zero()),
                RankedDifferentiable::of_scalar(Scalar::zero()),
            ],
        );

        assert_eq!(*loss.real_part(), 33.21);
    }
}
