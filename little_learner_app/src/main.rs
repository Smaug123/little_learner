#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod sample;
mod with_tensor;

use core::hash::Hash;
use rand::Rng;

use little_learner::auto_diff::{grad, Differentiable, RankedDifferentiable};

use crate::sample::sample2;
use little_learner::loss::{l2_loss_2, plane_predictor, Predictor};
use little_learner::not_nan::{to_not_nan_1, to_not_nan_2};
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

struct GradientDescentHyperImmut<A> {
    learning_rate: A,
    iterations: u32,
    mu: A,
}

struct GradientDescentHyper<A, R: Rng> {
    sampling: Option<(R, usize)>,
    params: GradientDescentHyperImmut<A>,
}

impl<A> GradientDescentHyper<A, rand::rngs::StdRng> {
    fn new(learning_rate: A, iterations: u32, mu: A) -> Self {
        GradientDescentHyper {
            params: GradientDescentHyperImmut {
                learning_rate,
                iterations,
                mu,
            },
            sampling: None,
        }
    }
}

fn naked_gradient_descent<A>(
    hyper: &GradientDescentHyperImmut<A>,
    theta: Differentiable<A>,
    delta: &Differentiable<A>,
) -> Differentiable<A>
    where
        A: NumLike,
{
    let learning_rate = Scalar::make(hyper.learning_rate.clone());
    general_gradient_descent(
        theta,
        |theta| theta,
        |theta, delta| theta.clone() - delta.clone() * learning_rate.clone(),
        delta,
    )
}

fn velocity_gradient_descent<A>(
    hyper: &GradientDescentHyperImmut<A>,
    theta: (Differentiable<A>, Scalar<A>),
    delta: &Differentiable<A>,
) -> Differentiable<A>
    where
        A: NumLike,
{
    let learning_rate = Scalar::make(hyper.learning_rate.clone());
    let velocity = theta.1.clone();
    general_gradient_descent(
        theta,
        |(theta, _)| theta,
        |theta, delta| {
            theta.clone() + Scalar::make(hyper.mu.clone()) * velocity.clone()
                - delta.clone() * learning_rate.clone()
        },
        delta,
    )
}

fn general_gradient_descent<A, Inflated, Deflate, Adjust>(
    theta: Inflated,
    mut deflate: Deflate,
    mut adjust: Adjust,
    delta: &Differentiable<A>,
) -> Differentiable<A>
    where
        A: NumLike,
        Deflate: FnMut(Inflated) -> Differentiable<A>,
        Adjust: FnMut(&Scalar<A>, &Scalar<A>) -> Scalar<A>,
{
    Differentiable::map2(&deflate(theta), delta, &mut adjust)
}

/// `adjust` takes the previous value and a delta, and returns a deflated new value.
fn general_gradient_descent_step<
    A,
    F,
    Inflated,
    Deflate,
    Adjust,
    const RANK: usize,
    const PARAM_NUM: usize,
>(
    f: &mut F,
    theta: [Inflated; PARAM_NUM],
    deflate: Deflate,
    mut adjust: Adjust,
) -> [Differentiable<A>; PARAM_NUM]
where
    A: Clone + NumLike + Hash + Eq,
    F: FnMut(&[Differentiable<A>; PARAM_NUM]) -> RankedDifferentiable<A, RANK>,
    Deflate: FnMut(Inflated) -> Differentiable<A>,
    Inflated: Clone,
    Adjust: FnMut(Inflated, &Differentiable<A>) -> Differentiable<A>,
{
    let deflated = theta.clone().map(deflate);
    let delta = grad(f, &deflated);
    let mut i = 0;
    theta.map(|inflated| {
        let delta = &delta[i];
        i += 1;
        adjust(inflated, delta)
    })
}

fn naked_gradient_descent_step<A, F, const RANK: usize, const PARAM_NUM: usize>(
    f: &mut F,
    theta: [Differentiable<A>; PARAM_NUM],
    hyper: &GradientDescentHyperImmut<A>,
) -> [Differentiable<A>; PARAM_NUM]
where
    A: Clone + NumLike + Hash + Eq,
    F: FnMut(&[Differentiable<A>; PARAM_NUM]) -> RankedDifferentiable<A, RANK>,
{
    general_gradient_descent_step(
        f,
        theta,
        |x| x,
        |theta, delta| naked_gradient_descent(&hyper, theta, delta),
    )
}

/// Each input `theta` is expected to be enriched with its "velocity" as well:
/// how much that parameter changed on the previous tick.
fn velocity_gradient_descent_step<A, F, const RANK: usize, const PARAM_NUM: usize>(
    f: &mut F,
    theta: [(Differentiable<A>, Scalar<A>); PARAM_NUM],
    hyper: &GradientDescentHyperImmut<A>,
) -> [Differentiable<A>; PARAM_NUM]
where
    A: Clone + NumLike + Hash + Eq,
    F: FnMut(&[Differentiable<A>; PARAM_NUM]) -> RankedDifferentiable<A, RANK>,
{
    general_gradient_descent_step(
        f,
        theta,
        |(x, _)| x,
        |theta, delta| velocity_gradient_descent(&hyper, theta, delta),
    )
}

fn gradient_descent<'a, T, R: Rng, Point, F, G, const IN_SIZE: usize, const PARAM_NUM: usize>(
    hyper: &mut GradientDescentHyper<T, R>,
    xs: &'a [Point],
    to_ranked_differentiable: G,
    ys: &[T],
    zero_params: [Differentiable<T>; PARAM_NUM],
    mut predictor: Predictor<F, Scalar<T>, Scalar<T>>,
) -> [Differentiable<T>; PARAM_NUM]
where
    T: NumLike + Hash + Copy + Default,
    Point: 'a + Copy,
    F: Fn(
        RankedDifferentiable<T, IN_SIZE>,
        &[Differentiable<T>; PARAM_NUM],
    ) -> RankedDifferentiable<T, 1>,
    G: for<'b> Fn(&'b [Point]) -> RankedDifferentiable<T, IN_SIZE>,
{
    let iterations = hyper.params.iterations;
    let out = iterate(
        |theta| {
            naked_gradient_descent_step(
                &mut |x| match hyper.sampling.as_mut() {
                    None => RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(
                        l2_loss_2(
                            &predictor.predict,
                            to_ranked_differentiable(xs),
                            RankedDifferentiable::of_slice(ys),
                            x,
                        ),
                    )]),
                    Some((rng, batch_size)) => {
                        let (sampled_xs, sampled_ys) = sample2(rng, *batch_size, xs, ys);
                        RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(
                            l2_loss_2(
                                &predictor.predict,
                                to_ranked_differentiable(&sampled_xs),
                                RankedDifferentiable::of_slice(&sampled_ys),
                                x,
                            ),
                        )])
                    }
                },
                theta,
                &hyper.params,
            )
        },
        zero_params.map(|x| x.map(&mut predictor.inflate)),
        iterations,
    );
    out.map(|x| x.map(&mut predictor.deflate))
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

    let mut hyper = GradientDescentHyper::new(
        NotNan::new(0.001).expect("not nan"),
        1000,
        NotNan::new(0.0).expect("not nan"),
    );

    let iterated = {
        let xs = to_not_nan_2(plane_xs);
        let ys = to_not_nan_1(plane_ys);
        let zero_params = [
            RankedDifferentiable::of_slice(&[NotNan::<f64>::zero(), NotNan::<f64>::zero()])
                .to_unranked(),
            Differentiable::of_scalar(Scalar::zero()),
        ];

        gradient_descent(
            &mut hyper,
            &xs,
            RankedDifferentiable::of_slice_2::<_, 2>,
            &ys,
            zero_params,
            plane_predictor(),
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
    use little_learner::loss::{line_unranked_predictor, quadratic_unranked_predictor};
    use rand::SeedableRng;

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

        let mut hyper = GradientDescentHyper::new(
            NotNan::new(0.01).expect("not nan"),
            1000,
            NotNan::new(0.0).expect("not nan"),
        );
        let iterated = {
            let xs = to_not_nan_1(xs);
            let ys = to_not_nan_1(ys);
            let zero_params = [
                RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
                RankedDifferentiable::of_scalar(zero).to_unranked(),
            ];
            gradient_descent(
                &mut hyper,
                &xs,
                |b| RankedDifferentiable::of_slice(b),
                &ys,
                zero_params,
                line_unranked_predictor(),
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

        let mut hyper = GradientDescentHyper::new(
            NotNan::new(0.001).expect("not nan"),
            1000,
            NotNan::new(0.0).expect("not nan"),
        );

        let iterated = {
            let xs = to_not_nan_1(xs);
            let ys = to_not_nan_1(ys);
            let zero_params = [
                RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
                RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
                RankedDifferentiable::of_scalar(zero).to_unranked(),
            ];
            gradient_descent(
                &mut hyper,
                &xs,
                |b| RankedDifferentiable::of_slice(b),
                &ys,
                zero_params,
                quadratic_unranked_predictor(),
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

    const PLANE_XS: [[f64; 2]; 6] = [
        [1.0, 2.05],
        [1.0, 3.0],
        [2.0, 2.0],
        [2.0, 3.91],
        [3.0, 6.13],
        [4.0, 8.09],
    ];
    const PLANE_YS: [f64; 6] = [13.99, 15.99, 18.0, 22.4, 30.2, 37.94];

    #[test]
    fn optimise_plane() {
        let mut hyper = GradientDescentHyper::new(
            NotNan::new(0.001).expect("not nan"),
            1000,
            NotNan::new(0.0).expect("not nan"),
        );

        let iterated = {
            let xs = to_not_nan_2(PLANE_XS);
            let ys = to_not_nan_1(PLANE_YS);
            let zero_params = [
                RankedDifferentiable::of_slice(&[NotNan::zero(), NotNan::zero()]).to_unranked(),
                Differentiable::of_scalar(Scalar::zero()),
            ];
            gradient_descent(
                &mut hyper,
                &xs,
                RankedDifferentiable::of_slice_2::<_, 2>,
                &ys,
                zero_params,
                plane_predictor(),
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

    #[test]
    fn optimise_plane_with_sampling() {
        let rng = rand::rngs::StdRng::seed_from_u64(314159);
        let mut hyper = GradientDescentHyper {
            params: GradientDescentHyperImmut {
                learning_rate: NotNan::new(0.001).expect("not nan"),
                iterations: 1000,
                mu: NotNan::new(0.0).expect("not nan"),
            },
            sampling: Some((rng, 4)),
        };

        let iterated = {
            let xs = to_not_nan_2(PLANE_XS);
            let ys = to_not_nan_1(PLANE_YS);
            let zero_params = [
                RankedDifferentiable::of_slice(&[NotNan::zero(), NotNan::zero()]).to_unranked(),
                Differentiable::of_scalar(Scalar::zero()),
            ];
            gradient_descent(
                &mut hyper,
                &xs,
                RankedDifferentiable::of_slice_2::<_, 2>,
                &ys,
                zero_params,
                plane_predictor(),
            )
        };

        let [theta0, theta1] = iterated;

        let theta0 = collect_vec(theta0.attach_rank::<1>().expect("rank 1 tensor"));
        let theta1 = theta1
            .attach_rank::<0>()
            .expect("rank 0 tensor")
            .to_scalar()
            .real_part()
            .into_inner();

        /*
           Mathematica code to verify by eye that the optimisation gave a reasonable result:

        xs = {{1.0, 2.05}, {1.0, 3.0}, {2.0, 2.0}, {2.0, 3.91}, {3.0,
            6.13}, {4.0, 8.09}};
        ys = {13.99, 15.99, 18.0, 22.4, 30.2, 37.94};
        points = ListPointPlot3D[Append @@@ Transpose[{xs, ys}]];

        withoutBatching0 = {3.97757644609063, 2.0496557321494446};
        withoutBatching1 = 5.2839863438547159;
        withoutBatching =
            Plot3D[{x, y} . withoutBatching0 + withoutBatching1, {x, 0, 4}, {y,
            0, 8}];

        withBatching0 = {3.8581694055684781, 2.2166222673968554};
        withBatching1 = 5.2399202468216668;
        withBatching =
            Plot3D[{x, y} . withBatching0 + withBatching1, {x, 0, 4}, {y, 0, 8}];

        Show[points, withoutBatching]

        Show[points, withBatching]
         */

        assert_eq!(theta0, [3.8581694055684781, 2.2166222673968554]);
        assert_eq!(theta1, 5.2839863438547159);
    }
}
