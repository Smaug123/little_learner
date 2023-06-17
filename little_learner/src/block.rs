use crate::auto_diff::{Differentiable, RankedDifferentiable, RankedDifferentiableTagged};
use crate::ext::relu;
use crate::traits::NumLike;

pub struct Block<F, const N: usize> {
    f: F,
    ranks: [usize; N],
}

/// Does the second argument first, so compose(b1, b2) performs b2 on its input, and then b1.
pub fn compose<'a, 'c, 'd, A, T, B, C, F, G, const N: usize, const M: usize>(
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
        f: move |t, theta| {
            let intermediate = (b1.f)(t, theta);
            (b2.f)(&intermediate, &theta[j..])
        },
        ranks,
    }
}

#[must_use]
pub fn dense<'b, A, Tag>(
    input_len: usize,
    neuron_count: usize,
) -> Block<
    impl for<'a> FnOnce(
        &'a RankedDifferentiableTagged<A, Tag, 1>,
        &'b [Differentiable<A>],
    ) -> RankedDifferentiable<A, 1>,
    2,
>
where
    Tag: Clone,
    A: NumLike + PartialOrd + Default,
{
    Block {
        f: for<'a> |t: &'a RankedDifferentiableTagged<A, Tag, 1>,
                    theta: &'b [Differentiable<A>]|
                 -> RankedDifferentiable<A, 1> {
            relu(
                t,
                &(theta[0].clone().attach_rank().unwrap()),
                &(theta[1].clone().attach_rank().unwrap()),
            )
            .attach_rank()
            .unwrap()
        },
        ranks: [input_len, neuron_count],
    }
}
