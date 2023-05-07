use crate::auto_diff::{Differentiable, DifferentiableTagged};
use crate::scalar::Scalar;
use crate::traits::One;
use std::ops::{Add, Mul, Neg};

/// Combine `old_value` and `new_value`, weighting the combination towards `new_value` by a factor
/// of `decay`.
pub fn smooth_tagged<A, F, Tag1, Tag2, Tag3>(
    decay: Scalar<A>,
    old_value: &DifferentiableTagged<A, Tag1>,
    new_value: &DifferentiableTagged<A, Tag2>,
    mut tags: F,
) -> DifferentiableTagged<A, Tag3>
where
    A: One + Clone + Mul<Output = A> + Neg<Output = A> + Add<Output = A>,
    F: FnMut(Tag1, Tag2) -> Tag3,
    Tag1: Clone,
    Tag2: Clone,
{
    DifferentiableTagged::map2_tagged(old_value, new_value, &mut |old, tag1, new, tag2| {
        (
            (old.clone() * decay.clone()) + (new.clone() * (Scalar::<A>::one() + -decay.clone())),
            tags(tag1, tag2),
        )
    })
}

/// Combine `old_value` and `new_value`, weighting the combination towards `new_value` by a factor
/// of `decay`.
pub fn smooth<A>(
    decay: Scalar<A>,
    old_value: &Differentiable<A>,
    new_value: &Differentiable<A>,
) -> Differentiable<A>
where
    A: One + Clone + Mul<Output = A> + Neg<Output = A> + Add<Output = A>,
{
    smooth_tagged(decay, old_value, new_value, |(), ()| ())
}

#[cfg(test)]
mod test_smooth {
    use crate::auto_diff::Differentiable;
    use crate::scalar::Scalar;
    use crate::smooth::smooth;
    use crate::traits::Zero;
    use ordered_float::NotNan;

    #[test]
    fn one_dimension() {
        let decay = Scalar::make(NotNan::new(0.9).expect("not nan"));
        let smoothed = smooth(
            decay.clone(),
            &Differentiable::of_scalar(Scalar::<NotNan<f64>>::zero()),
            &Differentiable::of_scalar(Scalar::make(NotNan::new(50.3).expect("not nan"))),
        );
        assert_eq!(
            smoothed.into_scalar().real_part().into_inner(),
            5.0299999999999985
        );

        let numbers = vec![50.3, 22.7, 4.3, 2.7, 1.8, 2.2, 0.6];
        let mut output = Vec::with_capacity(numbers.len());
        let mut acc = Scalar::<NotNan<f64>>::zero();
        for number in numbers {
            let number =
                Differentiable::of_scalar(Scalar::make(NotNan::new(number).expect("not nan")));
            let next = smooth(decay.clone(), &Differentiable::of_scalar(acc), &number);
            output.push(next.clone().into_scalar().clone_real_part().into_inner());
            acc = next.into_scalar();
        }

        // Note that the original sequence from the book has been heavily affected by rounding.
        // By zero-indexed element 4, the sequence is different in the first significant digit!
        assert_eq!(
            output,
            vec![
                5.0299999999999985,
                6.796_999_999_999_998,
                6.547_299_999_999_998,
                6.162_569_999_999_998,
                5.7263129999999975,
                5.373_681_699_999_998,
                4.896_313_529_999_998
            ]
        );
    }

    fn hydrate(v: &[f64]) -> Differentiable<NotNan<f64>> {
        Differentiable::of_vec(
            v.iter()
                .cloned()
                .map(|v| Differentiable::of_scalar(Scalar::make(NotNan::new(v).expect("not nan"))))
                .collect(),
        )
    }

    #[test]
    fn more_dimension() {
        let decay = Scalar::make(NotNan::new(0.9).expect("not nan"));

        let inputs = [
            vec![1.0, 1.1, 3.0],
            vec![13.4, 18.2, 41.4],
            vec![1.1, 0.3, 67.3],
        ]
        .map(|x| hydrate(&x));

        let mut current = hydrate(&vec![0.8, 3.1, 2.2]);
        let mut output = Vec::with_capacity(inputs.len());
        for input in inputs {
            current = smooth(decay.clone(), &current, &input);
            output.push(current.clone().attach_rank::<1>().unwrap().collect());
        }

        assert_eq!(
            output,
            vec![
                vec![0.820_000_000_000_000_1, 2.9, 2.2800000000000002],
                vec![2.078, 4.43, 6.191_999_999_999_999],
                vec![1.9802, 4.0169999999999995, 12.302799999999998]
            ]
        );
    }
}
