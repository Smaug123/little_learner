use crate::auto_diff::RankedDifferentiableTagged;
use crate::scalar::Scalar;
use crate::traits::Zero;

fn rectify<A>(x: A) -> A where A: Zero + PartialOrd {
    if x < A::zero() {
        A::zero()
    } else {
        x
    }
}

fn linear<A, Tag1, Tag2>(t: RankedDifferentiableTagged<A, Tag1, 1>, theta0: RankedDifferentiableTagged<A, Tag2, 1>, theta1: Scalar<A>) -> Scalar<A> {
    dot_product(theta0, t) + theta1
}