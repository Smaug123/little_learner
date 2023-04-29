#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod with_tensor;

use core::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, Neg};

use little_learner::auto_diff::{grad, Differentiable, RankedDifferentiable};

use little_learner::loss::{l2_loss_2, predict_plane};
use little_learner::scalar::Scalar;
use little_learner::traits::{Exp, One, Zero};
use ordered_float::NotNan;

fn iterate<A, F>(f: &F, start: A, n: u32) -> A
where
    F: Fn(A) -> A,
{
    let mut v = start;
    for _ in 0..n {
        v = f(v);
    }
    v
}

struct GradientDescentHyper<A> {
    learning_rate: A,
    iterations: u32,
}

fn gradient_descent_step<A, F, const RANK: usize, const PARAM_NUM: usize>(
    f: &F,
    theta: [Differentiable<A>; PARAM_NUM],
    params: &GradientDescentHyper<A>,
) -> [Differentiable<A>; PARAM_NUM]
where
    A: Clone
        + Mul<Output = A>
        + Neg<Output = A>
        + Add<Output = A>
        + Hash
        + AddAssign
        + Div<Output = A>
        + Zero
        + One
        + Eq
        + Exp,
    F: Fn(&[Differentiable<A>; PARAM_NUM]) -> RankedDifferentiable<A, RANK>,
{
    let delta = grad(f, &theta);
    let mut i = 0;
    theta.map(|theta| {
        let delta = &delta[i];
        i += 1;
        // For speed, you might want to truncate_dual this.
        let learning_rate = Scalar::make((params.learning_rate).clone());
        Differentiable::map2(
            &theta,
            &delta.map(&mut |s| s * learning_rate.clone()),
            &|theta, delta| (*theta).clone() - (*delta).clone(),
        )
    })
}

fn gradient_descent<T>(
    hyper: GradientDescentHyper<T>,
    xs: &[[T; 2]],
    ys: &[T],
) -> [Differentiable<T>; 2]
where
    T: Zero
        + Clone
        + Copy
        + One
        + Exp
        + Div<Output = T>
        + Eq
        + std::iter::Sum
        + Default
        + Add<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>
        + AddAssign
        + std::hash::Hash,
{
    iterate(
        &|theta| {
            gradient_descent_step(
                &|x| {
                    RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(
                        l2_loss_2(
                            predict_plane,
                            RankedDifferentiable::of_slice_2::<_, 2>(xs),
                            RankedDifferentiable::of_slice(ys),
                            x,
                        ),
                    )])
                },
                theta,
                &hyper,
            )
        },
        [
            RankedDifferentiable::of_slice([T::zero(), T::zero()]).to_unranked(),
            Differentiable::of_scalar(Scalar::zero()),
        ],
        hyper.iterations,
    )
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
    };

    let iterated = {
        let xs = plane_xs.map(|x| {
            [
                NotNan::new(x[0]).expect("not nan"),
                NotNan::new(x[1]).expect("not nan"),
            ]
        });
        let ys = plane_ys.map(|x| NotNan::new(x).expect("not nan"));
        gradient_descent(hyper, &xs, &ys)
    };

    let [theta0, theta1] = iterated;

    let theta0 = theta0.attach_rank::<1>().expect("rank 1 tensor");
    let theta1 = theta1.attach_rank::<0>().expect("rank 0 tensor");

    assert_eq!(
        theta0
            .to_vector()
            .into_iter()
            .map(|x| x.to_scalar().real_part().into_inner())
            .collect::<Vec<_>>(),
        [3.97757644609063, 2.0496557321494446]
    );
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
        let xs = [2.0, 1.0, 4.0, 3.0].map(|x| NotNan::new(x).expect("not nan"));
        let ys = [1.8, 1.2, 4.2, 3.3].map(|x| NotNan::new(x).expect("not nan"));
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
        assert_eq!(iterate(&f, [1, 2, 3], 5u32), [-14, -13, -12]);
    }

    #[test]
    fn first_optimisation_test() {
        let xs = [2.0, 1.0, 4.0, 3.0];
        let ys = [1.8, 1.2, 4.2, 3.3];

        let zero = Scalar::<NotNan<f64>>::zero();

        let hyper = GradientDescentHyper {
            learning_rate: NotNan::new(0.01).expect("not nan"),
            iterations: 1000,
        };
        let iterated = {
            let xs = xs.map(|x| NotNan::new(x).expect("not nan"));
            let ys = ys.map(|x| NotNan::new(x).expect("not nan"));
            iterate(
                &|theta| {
                    gradient_descent_step(
                        &|x| {
                            RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(
                                l2_loss_2(
                                    predict_line_2_unranked,
                                    RankedDifferentiable::of_slice(&xs),
                                    RankedDifferentiable::of_slice(&ys),
                                    x,
                                ),
                            )])
                        },
                        theta,
                        &hyper,
                    )
                },
                [
                    RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
                    RankedDifferentiable::of_scalar(zero).to_unranked(),
                ],
                hyper.iterations,
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
        };

        let iterated = {
            let xs = xs.map(|x| NotNan::new(x).expect("not nan"));
            let ys = ys.map(|x| NotNan::new(x).expect("not nan"));
            iterate(
                &|theta| {
                    gradient_descent_step(
                        &|x| {
                            RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(
                                l2_loss_2(
                                    predict_quadratic_unranked,
                                    RankedDifferentiable::of_slice(&xs),
                                    RankedDifferentiable::of_slice(&ys),
                                    x,
                                ),
                            )])
                        },
                        theta,
                        &hyper,
                    )
                },
                [
                    RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
                    RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
                    RankedDifferentiable::of_scalar(zero).to_unranked(),
                ],
                hyper.iterations,
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
        };

        let iterated = {
            let xs = plane_xs.map(|x| {
                [
                    NotNan::new(x[0]).expect("not nan"),
                    NotNan::new(x[1]).expect("not nan"),
                ]
            });
            let ys = plane_ys.map(|x| NotNan::new(x).expect("not nan"));
            iterate(
                &|theta| {
                    gradient_descent_step(
                        &|x| {
                            RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(
                                l2_loss_2(
                                    predict_plane,
                                    RankedDifferentiable::of_slice_2::<_, 2>(&xs),
                                    RankedDifferentiable::of_slice(ys),
                                    x,
                                ),
                            )])
                        },
                        theta,
                        &hyper,
                    )
                },
                [
                    RankedDifferentiable::of_slice([NotNan::zero(), NotNan::zero()]).to_unranked(),
                    Differentiable::of_scalar(Scalar::zero()),
                ],
                hyper.iterations,
            )
        };

        let [theta0, theta1] = iterated;

        let theta0 = theta0.attach_rank::<1>().expect("rank 1 tensor");
        let theta1 = theta1.attach_rank::<0>().expect("rank 0 tensor");

        assert_eq!(
            theta0
                .to_vector()
                .into_iter()
                .map(|x| x.to_scalar().real_part().into_inner())
                .collect::<Vec<_>>(),
            [3.97757644609063, 2.0496557321494446]
        );
        assert_eq!(
            theta1.to_scalar().real_part().into_inner(),
            5.786758464448078
        );
    }
}
