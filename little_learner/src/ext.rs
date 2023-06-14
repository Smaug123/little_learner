use crate::auto_diff::DifferentiableTagged;

pub fn ext1<A, B, Tag, Tag2, F>(
    n: usize,
    f: &mut F,
    t: &DifferentiableTagged<A, Tag>,
) -> DifferentiableTagged<B, Tag2>
where
    F: FnMut(&DifferentiableTagged<A, Tag>) -> DifferentiableTagged<B, Tag2>,
{
    if t.rank() == n {
        f(t)
    } else {
        t.map_once_tagged(|x| ext1(n, f, x))
    }
}

#[cfg(test)]
mod tests {
    use crate::auto_diff::{Differentiable, RankedDifferentiable};
    use crate::ext::ext1;
    use crate::not_nan::to_not_nan_2;
    use crate::scalar::Scalar;
    use crate::traits::Zero;

    fn zeros_redefined<A>(t: &Differentiable<A>) -> Differentiable<A>
    where
        A: Zero,
    {
        ext1(
            0,
            &mut |_| Differentiable::of_scalar(Scalar::make(A::zero())),
            t,
        )
    }

    #[test]
    fn define_zeros() {
        let shape = RankedDifferentiable::of_slice_2::<_, 2>(&to_not_nan_2([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]));
        let zeros = zeros_redefined(&shape.to_unranked());
        let to_zeros = zeros
            .attach_rank::<2>()
            .unwrap()
            .to_vector()
            .iter()
            .map(|x| {
                (*x).clone()
                    .to_vector()
                    .iter()
                    .map(|x| (*x).clone().to_scalar().clone_real_part().into_inner())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(to_zeros, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    }
}
