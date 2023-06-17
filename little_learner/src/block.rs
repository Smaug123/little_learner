use crate::auto_diff::{Differentiable, RankedDifferentiableTagged};
use crate::ext::relu;
use crate::traits::NumLike;

pub struct Block<F, const N: usize> {
    f: F,
    ranks: [usize; N],
}

pub fn compose<'a, A, T, B, C, F, G, const N: usize, const M: usize>(
    b1: Block<F, N>,
    b2: Block<G, M>,
    j: usize,
) -> Block<impl FnOnce(A, &'a [T]) -> C, { N + M }>
where
    F: FnOnce(A, &'a [T]) -> B,
    G: FnOnce(B, &'a [T]) -> C,
    T: 'a,
{
    let mut ranks = [0usize; N + M];
    ranks.copy_from_slice(&b1.ranks[..N]);
    ranks[N..(M + N)].copy_from_slice(&b2.ranks[..M]);
    Block {
        f: move |t, theta| (b2.f)((b1.f)(t, theta), &theta[j..]),
        ranks,
    }
}

pub fn dense<'a, 'b, A, Tag>(
    input_len: usize,
    neuron_count: usize,
) -> Block<
    impl FnOnce(&'a RankedDifferentiableTagged<A, Tag, 1>, &'b [Differentiable<A>]) -> Differentiable<A>,
    2,
>
where
    Tag: Clone,
    A: NumLike + PartialOrd + Default,
{
    Block {
        f: |t, theta: &'b [Differentiable<A>]| -> Differentiable<A> {
            relu(
                t,
                &(theta[0].clone().attach_rank().unwrap()),
                &(theta[1].clone().attach_rank().unwrap()),
            )
        },
        ranks: [input_len, neuron_count],
    }
}
