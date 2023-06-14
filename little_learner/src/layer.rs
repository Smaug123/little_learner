use crate::auto_diff::{Differentiable, RankedDifferentiable, RankedDifferentiableTagged};
use crate::decider::relu;
use crate::traits::NumLike;

/// Returns a tensor1.
/// Theta has two components: a tensor2 of weights and a tensor1 of bias.
pub fn layer<T>(
    theta: Differentiable<T>,
    t: RankedDifferentiable<T, 1>,
) -> RankedDifferentiable<T, 1>
where
    T: NumLike + PartialOrd,
{
    let mut theta = theta.into_vector();
    assert_eq!(theta.len(), 2, "Needed weights and a bias");
    let b = theta.pop().unwrap().attach_rank::<1>().unwrap();
    let w = theta.pop().unwrap().attach_rank::<2>().unwrap();

    RankedDifferentiableTagged::map2_once(
        &w,
        &b,
        &mut |w: &RankedDifferentiable<_, 1>, b: &RankedDifferentiable<_, 0>| {
            RankedDifferentiableTagged::of_scalar(relu(&t, w, b.clone().to_scalar()))
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::auto_diff::{Differentiable, RankedDifferentiable};
    use crate::layer::layer;
    use crate::not_nan::{to_not_nan_1, to_not_nan_2};

    #[test]
    fn test_single_layer() {
        let b = RankedDifferentiable::of_slice(&to_not_nan_1([1.0, 2.0]));
        let w = RankedDifferentiable::of_slice_2::<_, 2>(&to_not_nan_2([
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ]));
        let theta = Differentiable::of_vec(vec![w.to_unranked(), b.to_unranked()]);

        /*
        Two neurons:
        w =
          (3 4 5
           6 7 8)
        b = (1, 2)

        Three inputs:
        t = (9, 10, 11)

        Output has two elements, one per neuron.
        Neuron 1 has weights (3,4,5) and bias 1;
        Neuron 2 has weights (6,7,8) and bias 2.

        Neuron 1 is relu(t, (3,4,5), 1), which is (9, 10, 11).(3, 4, 5) + 1.
        Neuron 2 is relu(t, (6,7,8), 2), which is (9, 10, 11).(6, 7, 8) + 2.
         */

        let t = RankedDifferentiable::of_slice(&to_not_nan_1([9.0, 10.0, 11.0]));
        let mut output = layer(theta, t)
            .to_vector()
            .iter()
            .map(|t| (*t).clone().to_scalar().clone_real_part().into_inner())
            .collect::<Vec<_>>();

        assert_eq!(output.len(), 2);
        let result_2 = output.pop().unwrap();
        let result_1 = output.pop().unwrap();

        assert_eq!(result_1, (9 * 3 + 10 * 4 + 11 * 5 + 1) as f64);
        assert_eq!(result_2, (9 * 6 + 10 * 7 + 11 * 8 + 2) as f64);
    }
}
