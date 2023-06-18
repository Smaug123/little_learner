use crate::auto_diff::{Differentiable, DifferentiableTagged};
use crate::ext::k_relu;
use crate::scalar::Scalar;
use crate::traits::NumLike;
use num::Float;
use ordered_float::NotNan;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

pub struct Block<F: ?Sized, const N: usize> {
    pub f: Box<F>,
    ranks: [usize; N],
}

/// Does the second argument first, so compose(b1, b2) performs b2 on its input, and then b1.
pub fn compose_once<'a, 'c, 'd, A, T, B, C, F, G, const N: usize, const M: usize>(
    b1: Block<F, N>,
    b2: Block<G, M>,
    j: usize,
) -> Block<impl FnOnce(&'a A, &'d [T]) -> C, { N + M }>
where
    F: FnOnce(&'a A, &'d [T]) -> B,
    G: for<'b> FnOnce(&'b B, &'d [T]) -> C,
    A: 'a,
    T: 'd,
{
    let mut ranks = [0usize; N + M];
    ranks[..N].copy_from_slice(&b1.ranks);
    ranks[N..(M + N)].copy_from_slice(&b2.ranks);
    Block {
        f: Box::new(move |t, theta| {
            let intermediate = (b1.f)(t, theta);
            (b2.f)(&intermediate, &theta[j..])
        }),
        ranks,
    }
}

/// Does the second argument first, so compose(b1, b2) performs b2 on its input, and then b1.
pub fn compose_mut<A, T, B, C, F, G, const N: usize, const M: usize>(
    mut b1: Block<F, N>,
    mut b2: Block<G, M>,
    j: usize,
) -> Block<dyn for<'a, 'b> FnMut(&'a A, &'b [T]) -> C, { N + M }>
where
    F: for<'a, 'd> FnMut(&'a A, &'d [T]) -> B + 'static,
    G: for<'b, 'd> FnMut(&'b B, &'d [T]) -> C + 'static,
{
    let mut ranks = [0usize; N + M];
    ranks[..N].copy_from_slice(&b1.ranks);
    ranks[N..(M + N)].copy_from_slice(&b2.ranks);
    Block {
        f: Box::new(move |t, theta| {
            let intermediate = (b1.f)(t, theta);
            (b2.f)(&intermediate, &theta[j..])
        }),
        ranks,
    }
}

#[must_use]
pub fn dense_mut<A, Tag>(
    input_len: usize,
    neuron_count: usize,
) -> Block<
    impl for<'a, 'b> FnMut(
        &'a DifferentiableTagged<A, Tag>,
        &'b [Differentiable<A>],
    ) -> Differentiable<A>,
    2,
>
where
    Tag: Clone,
    A: NumLike + PartialOrd + Default + std::fmt::Display,
{
    Block {
        f: Box::new(
            for<'a, 'b> |t: &'a DifferentiableTagged<A, Tag>,
                         theta: &'b [Differentiable<A>]|
                         -> Differentiable<A> { k_relu(t, &theta[0..2]) },
        ),
        ranks: [input_len, neuron_count],
    }
}

pub fn dense_initial_weights<A, R>(
    rng: &mut R,
    input_len: usize,
    neuron_count: usize,
) -> Differentiable<NotNan<A>>
where
    R: Rng,
    A: Float + num::One,
    Standard: Distribution<A>,
{
    let mut rows = Vec::with_capacity(neuron_count);
    // Variance of 2/n, mean of 0, suggests uniform distribution on [-sqrt(6/n), sqrt(6/n)].
    // The Rust standard distribution on floats is uniform between 0 and 0.5.
    let n = A::from(neuron_count).unwrap();
    let four = A::from(4).unwrap();
    let six = A::from(6).unwrap();
    let dist = Standard.map(|x| ((four * x) - A::one()) * (six * n.recip()).sqrt());

    for _ in 0..neuron_count {
        let mut row = Vec::with_capacity(input_len);
        for _ in 0..input_len {
            row.push(Differentiable::of_scalar(Scalar::make(
                NotNan::new(dist.sample(rng)).unwrap(),
            )));
        }
        rows.push(Differentiable::of_vec(row));
    }
    Differentiable::of_vec(rows)
}

pub fn dense_initial_biases<A>(neuron_count: usize) -> Differentiable<A>
where
    A: crate::traits::Zero + Clone,
{
    Differentiable::of_vec(vec![
        Differentiable::of_scalar(Scalar::make(A::zero()));
        neuron_count
    ])
}
