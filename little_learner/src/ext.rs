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
    use ordered_float::NotNan;

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

    fn flatten_2<A>(t: RankedDifferentiable<A, 2>) -> RankedDifferentiable<A, 1>
    where
        A: Clone,
    {
        let mut result = Vec::new();
        for v in t.to_unranked_borrow().borrow_vector() {
            result.extend((*v.borrow_vector()).clone())
        }
        Differentiable::of_vec(result).attach_rank::<1>().unwrap()
    }

    #[test]
    fn test_flatten_2() {
        let input = RankedDifferentiable::of_slice_2::<_, 2>(&to_not_nan_2([
            [1.0, 0.5],
            [3.1, 2.2],
            [7.3, 2.1],
        ]));
        let flattened = flatten_2(input);
        let reshaped = flattened
            .to_vector()
            .iter()
            .map(|x| (*x).clone().to_scalar().clone_real_part().into_inner())
            .collect::<Vec<_>>();
        assert_eq!(reshaped, [1.0, 0.5, 3.1, 2.2, 7.3, 2.1])
    }

    #[test]
    fn test_flatten() {
        let flatten = |t: &Differentiable<NotNan<f64>>| {
            ext1(
                2,
                &mut |t| flatten_2((*t).clone().attach_rank::<2>().unwrap()).to_unranked(),
                t,
            )
        };
        let input = RankedDifferentiable::of_vector(vec![
            RankedDifferentiable::of_slice_2::<_, 2>(&to_not_nan_2([
                [1.0, 0.5],
                [3.1, 2.2],
                [7.3, 2.1],
            ])),
            RankedDifferentiable::of_slice_2::<_, 2>(&to_not_nan_2([
                [2.9, 3.5],
                [0.7, 1.5],
                [2.5, 6.4],
            ])),
        ]);

        let flattened = flatten(&input.to_unranked())
            .attach_rank::<2>()
            .unwrap()
            .to_vector()
            .iter()
            .map(|i| {
                i.to_unranked_borrow()
                    .borrow_vector()
                    .iter()
                    .map(|j| j.borrow_scalar().clone_real_part().into_inner())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            flattened,
            [
                [1.0, 0.5, 3.1, 2.2, 7.3, 2.1],
                [2.9, 3.5, 0.7, 1.5, 2.5, 6.4]
            ]
        )
    }
}
