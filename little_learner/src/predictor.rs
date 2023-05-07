use crate::auto_diff::{Differentiable, DifferentiableTagged};
use crate::scalar::Scalar;
use crate::smooth::smooth;
use crate::traits::{NumLike, Sqrt};

/// A Predictor is a function (`predict`) we're optimising, an `inflate` which adds any metadata
/// that the prediction engine might require, a corresponding `deflate` which removes the metadata,
/// and an `update` which computes the next guess based on the previous guess.
pub struct Predictor<F, Inflated, Deflated, Params> {
    /// The function we're trying to optimise.
    pub predict: F,
    /// Attach prediction metadata to an input to the function we're trying to optimise.
    pub inflate: fn(Deflated) -> Inflated,
    /// Remove prediction metadata.
    pub deflate: fn(Inflated) -> Deflated,
    /// Given a guess at an optimum, the gradient at that point, and any hyperparameters,
    /// compute the next guess at the optimum.
    pub update: fn(Inflated, &Deflated, Params) -> Inflated,
}

/// Hyperparameters applying to the most basic way to calculate the next step.
#[derive(Clone)]
pub struct NakedHypers<A> {
    pub learning_rate: A,
}

pub const fn naked<F, A>(f: F) -> Predictor<F, Differentiable<A>, Differentiable<A>, NakedHypers<A>>
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
                (theta.clone() - delta.clone() * learning_rate.clone()).truncate_dual(None)
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

pub const fn rms<F, A>(
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
                        (theta.clone()
                            + -(delta.clone() * Scalar::make(hyper.learning_rate.clone())))
                        .truncate_dual(None),
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

pub const fn velocity<F, A>(
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
