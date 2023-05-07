#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod with_tensor;

use little_learner::auto_diff::{Differentiable, RankedDifferentiable, RankedDifferentiableTagged};

use little_learner::gradient_descent::gradient_descent;
use little_learner::hyper::VelocityGradientDescentHyper;
use little_learner::loss::{predict_plane, velocity_predictor};
use little_learner::not_nan::{to_not_nan_1, to_not_nan_2};
use little_learner::scalar::Scalar;
use little_learner::traits::Zero;
use ordered_float::NotNan;

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

    let hyper =
        VelocityGradientDescentHyper::zero_momentum(NotNan::new(0.001).expect("not nan"), 1000)
            .with_mu(NotNan::new(0.9).expect("not nan"));

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
            RankedDifferentiableTagged::of_slice_2::<_, 2>,
            &ys,
            zero_params,
            velocity_predictor(predict_plane),
            VelocityGradientDescentHyper::to_immutable,
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

#[cfg(test)]
mod tests {}
