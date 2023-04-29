#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod sample;
mod with_tensor;

use core::hash::Hash;
use rand::Rng;

use little_learner::auto_diff::{grad, Differentiable, RankedDifferentiable};

use crate::sample::sample2;
use little_learner::loss::{l2_loss_2, predict_plane};
use little_learner::scalar::Scalar;
use little_learner::traits::{NumLike, Zero};
use ordered_float::NotNan;

fn iterate<A, F>(mut f: F, start: A, n: u32) -> A
where
    F: FnMut(A) -> A,
{
    let mut v = start;
    for _ in 0..n {
        v = f(v);
    }
    v
}

struct GradientDescentHyper<A, R: Rng> {
    learning_rate: A,
    iterations: u32,
    sampling: Option<(R, usize)>,
}

fn gradient_descent_step<A, F, const RANK: usize, const PARAM_NUM: usize>(
    f: &mut F,
    theta: [Differentiable<A>; PARAM_NUM],
    learning_rate: A,
) -> [Differentiable<A>; PARAM_NUM]
where
    A: Clone + NumLike + Hash + Eq,
    F: FnMut(&[Differentiable<A>; PARAM_NUM]) -> RankedDifferentiable<A, RANK>,
{
    let delta = grad(f, &theta);
    let mut i = 0;
    theta.map(|theta| {
        let delta = &delta[i];
        i += 1;
        // For speed, you might want to truncate_dual this.
        let learning_rate = Scalar::make(learning_rate.clone());
        Differentiable::map2(
            &theta,
            &delta.map(&mut |s| s * learning_rate.clone()),
            &mut |theta, delta| (*theta).clone() - (*delta).clone(),
        )
    })
}

fn gradient_descent<'a, T, R: Rng, Point, F, G, const IN_SIZE: usize, const PARAM_NUM: usize>(
    mut hyper: GradientDescentHyper<T, R>,
    xs: &'a [Point],
    to_ranked_differentiable: G,
    ys: &[T],
    zero_params: [Differentiable<T>; PARAM_NUM],
    predict: F,
) -> [Differentiable<T>; PARAM_NUM]
where
    T: NumLike + Clone + Copy + Eq + std::iter::Sum + Default + Hash,
    Point: 'a + Copy,
    F: Fn(
        RankedDifferentiable<T, IN_SIZE>,
        &[Differentiable<T>; PARAM_NUM],
    ) -> RankedDifferentiable<T, 1>,
    G: for<'b> Fn(&'b [Point]) -> RankedDifferentiable<T, IN_SIZE>,
{
    let iterations = hyper.iterations;
    iterate(
        |theta| {
            gradient_descent_step(
                &mut |x| match hyper.sampling.as_mut() {
                    None => RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(
                        l2_loss_2(
                            &predict,
                            to_ranked_differentiable(xs),
                            RankedDifferentiable::of_slice(ys),
                            x,
                        ),
                    )]),
                    Some((rng, batch_size)) => {
                        let (sampled_xs, sampled_ys) = sample2(rng, *batch_size, xs, ys);
                        RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(
                            l2_loss_2(
                                &predict,
                                to_ranked_differentiable(&sampled_xs),
                                RankedDifferentiable::of_slice(&sampled_ys),
                                x,
                            ),
                        )])
                    }
                },
                theta,
                hyper.learning_rate,
            )
        },
        zero_params,
        iterations,
    )
}

fn to_not_nan_1<T, const N: usize>(xs: [T; N]) -> [NotNan<T>; N]
where
    T: ordered_float::Float,
{
    xs.map(|x| NotNan::new(x).expect("not nan"))
}

fn to_not_nan_2<T, const N: usize, const M: usize>(xs: [[T; N]; M]) -> [[NotNan<T>; N]; M]
where
    T: ordered_float::Float,
{
    xs.map(to_not_nan_1)
}

fn collect_vec<T>(input: RankedDifferentiable<NotNan<T>, 1>) -> Vec<T>
where
    T: Copy,
{
    input
        .to_vector()
        .into_iter()
        .map(|x| x.to_scalar().real_part().into_inner())
        .collect::<Vec<_>>()
}

fn main() {
    let plane_xs = [
        [1.0, 2.05],
        [1.0, 3.0],
        [2.0, 2.0],
        [2.0, 3.91],
        [3.0, 6.13],
        [4.0, 8.09],
    ];
    let plane_ys = [13.99, 15.99, 18.0, 22.4, 30.2, 37.94];

    let hyper = GradientDescentHyper {
        learning_rate: NotNan::new(0.001).expect("not nan"),
        iterations: 1000,
        sampling: None::<(rand::rngs::StdRng, _)>,
    };

    let iterated = {
        let xs = to_not_nan_2(plane_xs);
        let ys = to_not_nan_1(plane_ys);
        let zero_params = [
            RankedDifferentiable::of_slice(&[NotNan::<f64>::zero(), NotNan::<f64>::zero()])
                .to_unranked(),
            Differentiable::of_scalar(Scalar::zero()),
        ];

        gradient_descent(
            hyper,
            &xs,
            RankedDifferentiable::of_slice_2::<_, 2>,
            &ys,
            zero_params,
            predict_plane,
        )
    };

    let [theta0, theta1] = iterated;

    let theta0 = theta0.attach_rank::<1>().expect("rank 1 tensor");
    let theta1 = theta1.attach_rank::<0>().expect("rank 0 tensor");

    assert_eq!(collect_vec(theta0), [3.97757644609063, 2.0496557321494446]);
    assert_eq!(
        theta1.to_scalar().real_part().into_inner(),
        5.786758464448078
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use little_learner::{
        auto_diff::grad,
        loss::{l2_loss_2, predict_line_2, predict_line_2_unranked, predict_quadratic_unranked},
    };

    use crate::with_tensor::{l2_loss, predict_line};

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

    #[test]
    fn l2_loss_non_autodiff_example() {
        let xs = [2.0, 1.0, 4.0, 3.0];
        let ys = [1.8, 1.2, 4.2, 3.3];
        let loss = l2_loss(predict_line, &xs, &ys, &[0.0099, 0.0]);
        assert_eq!(loss, 32.5892403);
    }

    #[test]
    fn grad_example() {
        let input_vec = [Differentiable::of_scalar(Scalar::make(
            NotNan::new(27.0).expect("not nan"),
        ))];

        let grad: Vec<_> = grad(
            |x| {
                RankedDifferentiable::of_scalar(
                    x[0].borrow_scalar().clone() * x[0].borrow_scalar().clone(),
                )
            },
            &input_vec,
        )
        .into_iter()
        .map(|x| x.into_scalar().real_part().into_inner())
        .collect();
        assert_eq!(grad, [54.0]);
    }

    #[test]
    fn loss_gradient() {
        let zero = Scalar::<NotNan<f64>>::zero();
        let input_vec = [
            RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
            RankedDifferentiable::of_scalar(zero).to_unranked(),
        ];
        let xs = to_not_nan_1([2.0, 1.0, 4.0, 3.0]);
        let ys = to_not_nan_1([1.8, 1.2, 4.2, 3.3]);
        let grad = grad(
            |x| {
                RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(l2_loss_2(
                    predict_line_2_unranked,
                    RankedDifferentiable::of_slice(&xs),
                    RankedDifferentiable::of_slice(&ys),
                    x,
                ))])
            },
            &input_vec,
        );

        assert_eq!(
            grad.into_iter()
                .map(|x| *(x.into_scalar().real_part()))
                .collect::<Vec<_>>(),
            [-63.0, -21.0]
        );
    }

    #[test]
    fn test_iterate() {
        let f = |t: [i32; 3]| t.map(|i| i - 3);
        assert_eq!(iterate(f, [1, 2, 3], 5u32), [-14, -13, -12]);
    }

    #[test]
    fn first_optimisation_test() {
        let xs = [2.0, 1.0, 4.0, 3.0];
        let ys = [1.8, 1.2, 4.2, 3.3];

        let zero = Scalar::<NotNan<f64>>::zero();

        let hyper = GradientDescentHyper {
            learning_rate: NotNan::new(0.01).expect("not nan"),
            iterations: 1000,
            sampling: None::<(rand::rngs::StdRng, _)>,
        };
        let iterated = {
            let xs = to_not_nan_1(xs);
            let ys = to_not_nan_1(ys);
            let zero_params = [
                RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
                RankedDifferentiable::of_scalar(zero).to_unranked(),
            ];
            gradient_descent(
                hyper,
                &xs,
                |b| RankedDifferentiable::of_slice(b),
                &ys,
                zero_params,
                predict_line_2_unranked,
            )
        };
        let iterated = iterated
            .into_iter()
            .map(|x| x.into_scalar().real_part().into_inner())
            .collect::<Vec<_>>();

        assert_eq!(iterated, vec![1.0499993623489503, 0.0000018747718457656533]);
    }

    #[test]
    fn optimise_quadratic() {
        let xs = [-1.0, 0.0, 1.0, 2.0, 3.0];
        let ys = [2.55, 2.1, 4.35, 10.2, 18.25];

        let zero = Scalar::<NotNan<f64>>::zero();

        let hyper = GradientDescentHyper {
            learning_rate: NotNan::new(0.001).expect("not nan"),
            iterations: 1000,
            sampling: None::<(rand::rngs::StdRng, _)>,
        };

        let iterated = {
            let xs = to_not_nan_1(xs);
            let ys = to_not_nan_1(ys);
            let zero_params = [
                RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
                RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
                RankedDifferentiable::of_scalar(zero).to_unranked(),
            ];
            gradient_descent(
                hyper,
                &xs,
                |b| RankedDifferentiable::of_slice(b),
                &ys,
                zero_params,
                predict_quadratic_unranked,
            )
        };
        let iterated = iterated
            .into_iter()
            .map(|x| x.into_scalar().real_part().into_inner())
            .collect::<Vec<_>>();

        assert_eq!(
            iterated,
            [2.0546423148479684, 0.9928606519360353, 1.4787394427094362]
        );
    }

    #[test]
    fn optimise_plane() {
        let plane_xs = [
            [1.0, 2.05],
            [1.0, 3.0],
            [2.0, 2.0],
            [2.0, 3.91],
            [3.0, 6.13],
            [4.0, 8.09],
        ];
        let plane_ys = [13.99, 15.99, 18.0, 22.4, 30.2, 37.94];

        let hyper = GradientDescentHyper {
            learning_rate: NotNan::new(0.001).expect("not nan"),
            iterations: 1000,
            sampling: None::<(rand::rngs::StdRng, _)>,
        };

        let iterated = {
            let xs = to_not_nan_2(plane_xs);
            let ys = to_not_nan_1(plane_ys);
            let zero_params = [
                RankedDifferentiable::of_slice(&[NotNan::zero(), NotNan::zero()]).to_unranked(),
                Differentiable::of_scalar(Scalar::zero()),
            ];
            gradient_descent(
                hyper,
                &xs,
                RankedDifferentiable::of_slice_2::<_, 2>,
                &ys,
                zero_params,
                predict_plane,
            )
        };

        let [theta0, theta1] = iterated;

        let theta0 = theta0.attach_rank::<1>().expect("rank 1 tensor");
        let theta1 = theta1.attach_rank::<0>().expect("rank 0 tensor");

        assert_eq!(collect_vec(theta0), [3.97757644609063, 2.0496557321494446]);
        assert_eq!(
            theta1.to_scalar().real_part().into_inner(),
            5.786758464448078
        );
    }
}
