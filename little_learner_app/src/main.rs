#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(closure_lifetime_binder)]

use crate::rms_example::rms_example;
use little_learner::argmax::one_hot_class_eq;
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
                    x.iter()
                        .map(|i| Differentiable::of_scalar(Scalar::make(i.clone())))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn to_diff_out<A, const N: usize>(x: &[[A; N]]) -> RankedDifferentiable<A, 2>
where
    A: Clone,
{
    RankedDifferentiable::of_vector(
        x.iter()
            .map(|x| {
                RankedDifferentiable::of_vector(
                    x.iter()
                        .map(|i| RankedDifferentiable::of_scalar(Scalar::make(i.clone())))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn main() {
    rms_example();

    let irises = iris::import::<f64, _>()
        .iter()
        .map(|x| x.map(|z| NotNan::new(z).unwrap()))
        .collect::<Vec<_>>();

    let mut rng = thread_rng();

    let data = iris::partition(&mut rng, &irises);
    let mut network = block::compose_mut(
        block::dense_mut::<NotNan<f64>, ()>(6, 3),
        block::dense_mut(4, 6),
        2,
    );
    let second_layer_weights = block::dense_initial_weights::<f64, _>(&mut rng, 6, 3);
    let first_layer_weights = block::dense_initial_weights::<f64, _>(&mut rng, 4, 6);
    let second_layer_biases = block::dense_initial_biases::<NotNan<f64>>(3);
    let first_layer_biases = block::dense_initial_biases::<NotNan<f64>>(6);

    let all_weights = [
        first_layer_weights,
        first_layer_biases,
        second_layer_weights,
        second_layer_biases,
    ];

    let hyper = hyper::NakedGradientDescent::new(NotNan::new(0.0002).expect("not nan"), 2000)
        .with_rng(rng, 8);

    let mut predictor = naked(
        for<'a, 'b> |x: &'a Differentiable<NotNan<f64>>,
                     params: &'b [Differentiable<NotNan<f64>>; 4]|
                     -> RankedDifferentiable<NotNan<f64>, 2> {
            let params = params.clone();
            let x = x.clone();
            (network.f)(&x, &params).attach_rank().unwrap()
        },
    );

    let params = gradient_descent(
        hyper,
        &data.training_xs,
        &mut to_diff,
        &mut to_diff_out,
        &data.training_ys,
        all_weights,
        &mut predictor,
        hyper::NakedGradientDescent::to_immutable,
    );

    println!("First layer weights: {}", &params[0]);
    println!("First layer biases: {}", &params[1]);
    println!("Second layer weights: {}", &params[2]);
    println!("Second layer biases: {}", &params[3]);

    let mut good_count = 0;
    let mut bad_count = 0;
    for (test_x, expected) in data.test_xs.iter().zip(data.test_ys.iter()) {
        let actual = (predictor.predict)(&to_diff(&[*test_x]), &params);
        println!("Test: {:?}. Actual: {}", test_x, actual);
        // We made a single prediction so this is safe:
        let actual = &actual.to_vector()[0];
        let expected = &to_diff_out(&[*expected]).to_vector()[0];
        if !one_hot_class_eq(expected, actual) {
            println!("Bad prediction! Actual: {}; expected: {}", actual, expected);
            bad_count += 1;
        } else {
            good_count += 1;
            println!(
                "Good prediction! Actual: {}; expected: {}",
                actual, expected
            )
        }
    }

    println!("Bad: {}; good: {}", bad_count, good_count);
}
