use crate::auto_diff::{grad, Differentiable, RankedDifferentiable};
use crate::hyper;
use crate::loss::l2_loss_2;
use crate::predictor::Predictor;
use crate::sample::sample2;
use crate::traits::NumLike;
use rand::Rng;
use std::hash::Hash;

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

/// `adjust` takes the previous value and a delta, and returns a deflated new value.
fn general_gradient_descent_step<
    A,
    F,
    Inflated,
    Deflate,
    Adjust,
    Hyper,
    const RANK: usize,
    const PARAM_NUM: usize,
>(
    f: &mut F,
    theta: [Inflated; PARAM_NUM],
    deflate: Deflate,
    hyper: Hyper,
    mut adjust: Adjust,
) -> [Inflated; PARAM_NUM]
where
    A: Clone + NumLike + Hash + Eq,
    F: FnMut(&[Differentiable<A>; PARAM_NUM]) -> RankedDifferentiable<A, RANK>,
    Deflate: FnMut(Inflated) -> Differentiable<A>,
    Inflated: Clone,
    Hyper: Clone,
    Adjust: FnMut(Inflated, &Differentiable<A>, Hyper) -> Inflated,
{
    let deflated = theta.clone().map(deflate);
    let delta = grad(f, &deflated);
    let mut i = 0;
    theta.map(|inflated| {
        let delta = &delta[i];
        i += 1;
        adjust(inflated, delta, hyper.clone())
    })
}

pub fn gradient_descent<
    'a,
    T,
    R,
    Point,
    F,
    G,
    H,
    Inflated,
    Hyper,
    ImmutableHyper,
    const IN_SIZE: usize,
    const PARAM_NUM: usize,
>(
    hyper: Hyper,
    xs: &'a [Point],
    to_ranked_differentiable: G,
    ys: &[T],
    zero_params: [Differentiable<T>; PARAM_NUM],
    mut predictor: Predictor<F, Inflated, Differentiable<T>, ImmutableHyper>,
    to_immutable: H,
) -> [Differentiable<T>; PARAM_NUM]
where
    T: NumLike + Hash + Copy + Default,
    Point: 'a + Copy,
    F: Fn(
        RankedDifferentiable<T, IN_SIZE>,
        &[Differentiable<T>; PARAM_NUM],
    ) -> RankedDifferentiable<T, 1>,
    G: for<'b> Fn(&'b [Point]) -> RankedDifferentiable<T, IN_SIZE>,
    Inflated: Clone,
    ImmutableHyper: Clone,
    Hyper: Into<hyper::BaseGradientDescent<R>>,
    H: FnOnce(&Hyper) -> ImmutableHyper,
    R: Rng,
{
    let sub_hypers = to_immutable(&hyper);
    let mut gradient_hyper: hyper::BaseGradientDescent<R> = hyper.into();
    let iterations = gradient_hyper.iterations;
    let out = iterate(
        |theta| {
            general_gradient_descent_step(
                &mut |x| match gradient_hyper.sampling.as_mut() {
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
                predictor.deflate,
                sub_hypers.clone(),
                predictor.update,
            )
        },
        zero_params.map(predictor.inflate),
        iterations,
    );
    out.map(&mut predictor.deflate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto_diff::RankedDifferentiableTagged;
    use crate::hyper;
    use crate::loss::{predict_line_2_unranked, predict_plane, predict_quadratic_unranked};
    use crate::not_nan::{to_not_nan_1, to_not_nan_2};
    use crate::predictor;
    use crate::scalar::Scalar;
    use crate::traits::Zero;
    use ordered_float::NotNan;
    use rand::rngs::StdRng;
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

        let hyper = hyper::NakedGradientDescent::new(NotNan::new(0.01).expect("not nan"), 1000);
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
                predictor::naked(predict_line_2_unranked),
                hyper::NakedGradientDescent::to_immutable,
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

        let hyper = hyper::NakedGradientDescent::new(NotNan::new(0.001).expect("not nan"), 1000);

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
                predictor::naked(predict_quadratic_unranked),
                hyper::NakedGradientDescent::to_immutable,
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
        let hyper = hyper::NakedGradientDescent::new(NotNan::new(0.001).expect("not nan"), 1000);

        let iterated = {
            let xs = to_not_nan_2(PLANE_XS);
            let ys = to_not_nan_1(PLANE_YS);
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
                predictor::naked(predict_plane),
                hyper::NakedGradientDescent::to_immutable,
            )
        };

        let [theta0, theta1] = iterated;

        let theta0 = theta0.attach_rank::<1>().expect("rank 1 tensor");
        let theta1 = theta1.attach_rank::<0>().expect("rank 0 tensor");

        assert_eq!(theta0.collect(), [3.97757644609063, 2.0496557321494446]);
        assert_eq!(
            theta1.to_scalar().real_part().into_inner(),
            5.786758464448078
        );
    }

    #[test]
    fn optimise_plane_with_sampling() {
        let rng = StdRng::seed_from_u64(314159);
        let hyper = hyper::NakedGradientDescent::new(NotNan::new(0.001).expect("not nan"), 1000)
            .with_rng(rng, 4);

        let iterated = {
            let xs = to_not_nan_2(PLANE_XS);
            let ys = to_not_nan_1(PLANE_YS);
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
                predictor::naked(predict_plane),
                hyper::NakedGradientDescent::to_immutable,
            )
        };

        let [theta0, theta1] = iterated;

        let theta0 = theta0.attach_rank::<1>().expect("rank 1 tensor").collect();
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

        assert_eq!(theta0, [3.858_169_405_568_478, 2.2166222673968554]);
        assert_eq!(theta1, 5.283_986_343_854_716);
    }

    #[test]
    fn test_with_velocity() {
        let hyper = hyper::VelocityGradientDescent::zero_momentum(
            NotNan::new(0.001).expect("not nan"),
            1000,
        )
        .with_mu(NotNan::new(0.9).expect("not nan"));

        let iterated = {
            let xs = to_not_nan_2(PLANE_XS);
            let ys = to_not_nan_1(PLANE_YS);
            let zero_params = [
                RankedDifferentiable::of_slice(&[NotNan::<f64>::zero(), NotNan::<f64>::zero()])
                    .to_unranked(),
                Differentiable::of_scalar(Scalar::zero()),
            ];

            gradient_descent(
                hyper,
                &xs,
                RankedDifferentiableTagged::of_slice_2::<_, 2>,
                &ys,
                zero_params,
                predictor::velocity(predict_plane),
                hyper::VelocityGradientDescent::to_immutable,
            )
        };

        let [theta0, theta1] = iterated;

        let theta0 = theta0.attach_rank::<1>().expect("rank 1 tensor");
        let theta1 = theta1.attach_rank::<0>().expect("rank 0 tensor");

        assert_eq!(theta0.collect(), [3.979645447136021, 1.976454920954754]);
        assert_eq!(
            theta1.to_scalar().real_part().into_inner(),
            6.169579045974949
        );
    }

    #[test]
    fn test_with_rms() {
        let beta = NotNan::new(0.9).expect("not nan");
        let stabilizer = NotNan::new(0.00000001).expect("not nan");
        let hyper = hyper::RmsGradientDescent::default(NotNan::new(0.001).expect("not nan"), 3000)
            .with_stabilizer(stabilizer)
            .with_beta(beta);

        let iterated = {
            let xs = to_not_nan_2(PLANE_XS);
            let ys = to_not_nan_1(PLANE_YS);
            let zero_params = [
                RankedDifferentiable::of_slice(&[NotNan::<f64>::zero(), NotNan::<f64>::zero()])
                    .to_unranked(),
                Differentiable::of_scalar(Scalar::zero()),
            ];

            gradient_descent(
                hyper,
                &xs,
                RankedDifferentiableTagged::of_slice_2::<_, 2>,
                &ys,
                zero_params,
                predictor::rms(predict_plane),
                hyper::RmsGradientDescent::to_immutable,
            )
        };

        let [theta0, theta1] = iterated;

        let theta0 = theta0.attach_rank::<1>().expect("rank 1 tensor");
        let theta1 = theta1.attach_rank::<0>().expect("rank 0 tensor");

        let fitted_theta0 = theta0
            .collect()
            .iter()
            .map(|x| x.into_inner())
            .collect::<Vec<_>>();
        let fitted_theta1 = theta1.to_scalar().real_part().into_inner();
        assert_eq!(fitted_theta0, [3.985_350_099_342_649, 1.9745945728216352]);
        assert_eq!(fitted_theta1, 6.164_222_983_181_168);
    }
}
