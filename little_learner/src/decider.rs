use crate::auto_diff::RankedDifferentiableTagged;
use crate::loss::dot;
use crate::scalar::Scalar;
use crate::traits::{NumLike, Zero};

fn rectify<A>(x: A) -> A
where
    A: Zero + PartialOrd,
{
    if x < A::zero() {
        A::zero()
    } else {
        x
    }
}

fn linear<A, Tag1, Tag2>(
    t: &RankedDifferentiableTagged<A, Tag1, 1>,
    theta0: &RankedDifferentiableTagged<A, Tag2, 1>,
    theta1: Scalar<A>,
) -> Scalar<A>
where
    A: NumLike,
{
    dot(theta0, t) + theta1
}

pub fn relu<A, Tag1, Tag2>(
    t: &RankedDifferentiableTagged<A, Tag1, 1>,
    theta0: &RankedDifferentiableTagged<A, Tag2, 1>,
    theta1: Scalar<A>,
) -> Scalar<A>
where
    A: NumLike + PartialOrd,
{
    rectify(linear(t, theta0, theta1))
}

#[cfg(test)]
mod test_decider {
    use crate::auto_diff::RankedDifferentiable;
    use crate::decider::{linear, relu};
    use crate::not_nan::to_not_nan_1;
    use crate::scalar::Scalar;
    use ordered_float::NotNan;

    #[test]
    fn test_linear() {
        let theta0 = RankedDifferentiable::of_slice(&to_not_nan_1([7.1, 4.3, -6.4]));
        let theta1 = Scalar::make(NotNan::new(0.6).expect("not nan"));
        let t = RankedDifferentiable::of_slice(&to_not_nan_1([2.0, 1.0, 3.0]));

        let result = linear(&t, &theta0, theta1).real_part().into_inner();

        assert!((result + 0.1).abs() < 0.000_000_01);
    }

    #[test]
    fn test_relu() {
        let theta0 = RankedDifferentiable::of_slice(&to_not_nan_1([7.1, 4.3, -6.4]));
        let theta1 = Scalar::make(NotNan::new(0.6).expect("not nan"));
        let t = RankedDifferentiable::of_slice(&to_not_nan_1([2.0, 1.0, 3.0]));

        let result = relu(&t, &theta0, theta1).real_part().into_inner();

        assert_eq!(result, 0.0);
    }
}
