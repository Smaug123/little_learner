#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod with_tensor;

use little_learner::auto_diff::{Differentiable, RankedDifferentiable, RankedDifferentiableTagged};

use little_learner::gradient_descent::gradient_descent;
use little_learner::hyper::RmsGradientDescentHyper;
use little_learner::loss::predict_plane;
use little_learner::predictor::rms_predictor;
use little_learner::not_nan::{to_not_nan_1, to_not_nan_2};
use little_learner::scalar::Scalar;
use little_learner::traits::Zero;
use ordered_float::NotNan;

const PLANE_XS: [[f64; 2]; 6] = [
    [1.0, 2.05],
    [1.0, 3.0],
    [2.0, 2.0],
    [2.0, 3.91],
    [3.0, 6.13],
    [4.0, 8.09],
];
const PLANE_YS: [f64; 6] = [13.99, 15.99, 18.0, 22.4, 30.2, 37.94];

fn main() {
    let beta = NotNan::new(0.9).expect("not nan");
    let stabilizer = NotNan::new(0.00000001).expect("not nan");
    let hyper = RmsGradientDescentHyper::default(NotNan::new(0.001).expect("not nan"), 3000)
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
            rms_predictor(predict_plane),
            RmsGradientDescentHyper::to_immutable,
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
    assert_eq!(fitted_theta0, [3.9853500993426492, 1.9745945728216352]);
    assert_eq!(fitted_theta1, 6.1642229831811681);
}

#[cfg(test)]
mod tests {}
