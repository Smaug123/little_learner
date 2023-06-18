#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(closure_lifetime_binder)]

use crate::rms_example::rms_example;
use little_learner::auto_diff::{Differentiable, RankedDifferentiable};
use little_learner::gradient_descent::gradient_descent;
use little_learner::predictor::naked;
use little_learner::scalar::Scalar;
use little_learner::{block, hyper};
use ordered_float::NotNan;
use rand::thread_rng;

mod iris;
mod rms_example;

fn to_diff<A, const N: usize>(x: &[[A; N]]) -> Differentiable<A>
where
    A: Clone,
{
    Differentiable::of_vec(
        x.iter()
            .map(|x| {
                Differentiable::of_vec(
                    x.iter().map(|i| Differentiable::of_scalar(Scalar::make(i.clone())))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn to_diff_out<A, const N: usize>(x: &[[A; N]]) -> RankedDifferentiable<A, 1>
    where
        A: Clone,
{
    todo!()
}

fn main() {
    //rms_example();

    let irises = iris::import::<f64, _>()
        .iter()
        .map(|x| x.map(|z| NotNan::new(z).unwrap()))
        .collect::<Vec<_>>();
    let (training_xs, training_ys, test_xs, test_ys) = iris::partition(&irises);
    let mut network = block::compose_mut(
        block::dense_mut::<NotNan<f64>, ()>(6, 3),
        block::dense_mut(4, 6),
        2,
    );
    let mut rng = thread_rng();
    let second_layer_weights = block::dense_initial_weights::<f64, _>(&mut rng, 6, 3);
    let first_layer_weights = block::dense_initial_weights::<f64, _>(&mut rng, 4, 6);
    let second_layer_biases = block::dense_initial_biases::<NotNan<f64>>(6);
    let first_layer_biases = block::dense_initial_biases::<NotNan<f64>>(4);

    let all_weights = [
        first_layer_weights,
        first_layer_biases,
        second_layer_weights,
        second_layer_biases,
    ];

    let hyper = hyper::NakedGradientDescent::new(NotNan::new(0.0002).expect("not nan"), 2000)
        .with_rng(rng, 8);

    let predictor = naked(
        for<'a, 'b> |x: &'a Differentiable<NotNan<f64>>,
                 y: &'b [Differentiable<NotNan<f64>>; 4]|
                 -> RankedDifferentiable<NotNan<f64>, 1> {
            let x = x.clone();
            let y = y.clone();
            (network.f)(&x, &y).attach_rank().unwrap()
        },
    );

    let _iterated = gradient_descent(
        hyper,
        &training_xs,
        &mut to_diff,
        &mut to_diff_out,
        &training_ys,
        all_weights,
        predictor,
        hyper::NakedGradientDescent::to_immutable,
    );
}
