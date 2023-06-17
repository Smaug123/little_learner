use crate::auto_diff::{
    Differentiable, DifferentiableTagged, RankedDifferentiable, RankedDifferentiableTagged,
};
use crate::decider::rectify;
use crate::scalar::Scalar;
use crate::traits::{NumLike, Zero};
use std::iter::Sum;
use std::ops::{Add, Mul};

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

pub fn ext2<A, B, C, Tag, Tag2, Tag3, F>(
    n: usize,
    m: usize,
    f: &mut F,
    t: &DifferentiableTagged<A, Tag>,
    u: &DifferentiableTagged<B, Tag2>,
) -> DifferentiableTagged<C, Tag3>
where
    F: FnMut(
        &DifferentiableTagged<A, Tag>,
        &DifferentiableTagged<B, Tag2>,
    ) -> DifferentiableTagged<C, Tag3>,
    A: Clone,
    Tag: Clone,
    B: Clone,
    Tag2: Clone,
{
    if t.rank() == n && u.rank() == m {
        f(t, u)
    } else if t.rank() == n {
        u.map_once_tagged(|eu| ext2(n, m, f, t, eu))
    } else if u.rank() == m {
        t.map_once_tagged(|et| ext2(n, m, f, et, u))
    } else if t.rank() == u.rank() {
        t.map2_once_tagged(u, |t, u| ext2(n, m, f, t, u))
    } else if t.rank() > u.rank() {
        t.map_once_tagged(|et| ext2(n, m, f, et, u))
    } else {
        u.map_once_tagged(|eu| ext2(n, m, f, t, eu))
    }
}

pub fn elementwise_mul_via_ext<A, Tag, Tag2, const RANK1: usize, const RANK2: usize>(
    x: &RankedDifferentiableTagged<A, Tag, RANK1>,
    y: &RankedDifferentiableTagged<A, Tag2, RANK2>,
) -> RankedDifferentiable<A, RANK1>
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Clone + Default,
    Tag: Clone,
    Tag2: Clone,
{
    ext2(
        0,
        0,
        &mut |x, y| {
            DifferentiableTagged::of_scalar(x.borrow_scalar().clone() * y.borrow_scalar().clone())
        },
        x.to_unranked_borrow(),
        y.to_unranked_borrow(),
    )
    .attach_rank::<RANK1>()
    .unwrap()
}

/// Produce the element-wise multiplication of the inputs, threading where necessary until the
/// first argument has rank 2 and the second argument has rank 1.
/// This is essentially "matrix-multiply a matrix by a vector, but don't do the sum; instead
/// leave the components to be summed in a vector".
pub fn star_2_1<T, Tag, Tag2>(
    x: &DifferentiableTagged<T, Tag>,
    y: &DifferentiableTagged<T, Tag2>,
) -> Differentiable<T>
where
    T: Clone + Sum + Mul<Output = T> + Default,
    Tag: Clone,
    Tag2: Clone,
{
    ext2(
        2,
        1,
        &mut |x, y| {
            elementwise_mul_via_ext(
                &x.clone().attach_rank::<2>().unwrap(),
                &y.clone().attach_rank::<1>().unwrap(),
            )
            .to_unranked()
        },
        x,
        y,
    )
}

fn sum_1_scalar<A, Tag>(x: RankedDifferentiableTagged<A, Tag, 1>) -> Scalar<A>
where
    A: Sum<A> + Clone + Add<Output = A> + Zero,
{
    RankedDifferentiableTagged::to_vector(x)
        .into_iter()
        .map(|x| x.to_scalar())
        .sum()
}

pub fn sum_1<A, Tag>(x: RankedDifferentiableTagged<A, Tag, 1>) -> Differentiable<A>
where
    A: Sum<A> + Clone + Add<Output = A> + Zero,
{
    DifferentiableTagged::of_scalar(sum_1_scalar(x))
}

pub fn sum<T>(x: &Differentiable<T>) -> Differentiable<T>
where
    T: Sum<T> + Clone + Add<Output = T> + Zero,
{
    ext1(1, &mut |y| sum_1(y.clone().attach_rank::<1>().unwrap()), x)
}

/// Matrix-multiply W with T, threading where necessary until the first argument has rank 2 and the
/// second argument has rank 1.
pub fn dot_2_1<A, Tag, Tag2>(
    w: &DifferentiableTagged<A, Tag>,
    t: &DifferentiableTagged<A, Tag2>,
) -> Differentiable<A>
where
    A: NumLike + Default,
    Tag: Clone,
    Tag2: Clone,
{
    assert!(
        w.rank() >= 2,
        "w needed to have rank 2 or more, was {}",
        w.rank()
    );
    assert!(
        t.rank() >= 1,
        "t needed to have rank 1 or more, was {}",
        t.rank()
    );
    sum(&star_2_1(w, t))
}

pub fn linear<A, Tag1, Tag2, Tag3>(
    theta0: &DifferentiableTagged<A, Tag1>,
    theta1: &DifferentiableTagged<A, Tag2>,
    t: &DifferentiableTagged<A, Tag3>,
) -> DifferentiableTagged<A, ()>
where
    A: NumLike + Default,
    Tag1: Clone,
    Tag2: Clone,
    Tag3: Clone,
{
    dot_2_1(theta0, t).map2_tagged(theta1, &mut |x, _, y, _| (x.clone() + y.clone(), ()))
}

pub fn relu<A, Tag1, Tag2, Tag3>(
    t: &RankedDifferentiableTagged<A, Tag1, 1>,
    theta0: &RankedDifferentiableTagged<A, Tag2, 2>,
    theta1: &RankedDifferentiableTagged<A, Tag3, 1>,
) -> Differentiable<A>
where
    A: NumLike + PartialOrd + Default,
    Tag1: Clone,
    Tag2: Clone,
    Tag3: Clone,
{
    linear(
        theta0.to_unranked_borrow(),
        theta1.to_unranked_borrow(),
        t.to_unranked_borrow(),
    )
    .map(&mut rectify)
}

#[cfg(test)]
mod tests {
    use crate::auto_diff::{Differentiable, RankedDifferentiable};
    use crate::ext::{dot_2_1, ext1, relu, star_2_1};
    use crate::not_nan::{to_not_nan_1, to_not_nan_2};
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

    #[test]
    fn test_star_2_1_a() {
        let input1 = RankedDifferentiable::of_slice_2::<_, 2>(&to_not_nan_2([
            [3.0, 4.0, 5.0],
            [7.0, 8.0, 9.0],
        ]));
        let input2 = RankedDifferentiable::of_slice(&to_not_nan_1([2.0, 4.0, 3.0]));

        let output = star_2_1(input1.to_unranked_borrow(), input2.to_unranked_borrow())
            .into_vector()
            .iter()
            .map(|x| {
                x.clone()
                    .into_vector()
                    .iter()
                    .map(|i| i.clone().into_scalar().clone_real_part().into_inner())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(output, [[6.0, 16.0, 15.0], [14.0, 32.0, 27.0]])
    }

    #[test]
    fn test_star_2_1_b() {
        let input1 = RankedDifferentiable::of_slice_2::<_, 2>(&to_not_nan_2([
            [8.0, 1.0],
            [7.0, 3.0],
            [5.0, 4.0],
        ]));
        let input2 = RankedDifferentiable::of_slice_2::<_, 2>(&to_not_nan_2([
            [6.0, 2.0],
            [4.0, 9.0],
            [3.0, 8.0],
        ]));

        let output = star_2_1(input1.to_unranked_borrow(), input2.to_unranked_borrow())
            .into_vector()
            .iter()
            .map(|x| {
                x.clone()
                    .into_vector()
                    .iter()
                    .map(|i| {
                        i.clone()
                            .into_vector()
                            .iter()
                            .map(|i| i.borrow_scalar().clone_real_part().into_inner())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            output,
            [
                [[48.0, 2.0], [42.0, 6.0], [30.0, 8.0]],
                [[32.0, 9.0], [28.0, 27.0], [20.0, 36.0]],
                [[24.0, 8.0], [21.0, 24.0], [15.0, 32.0]]
            ]
        )
    }

    #[test]
    fn test_dot_2_1() {
        let w = RankedDifferentiable::of_slice_2::<_, 2>(&to_not_nan_2([
            [2.0, 1.0, 3.1],
            [3.7, 4.0, 6.1],
        ]));
        let t = RankedDifferentiable::of_slice(&to_not_nan_1([1.3, 0.4, 3.3]));

        let result = dot_2_1(w.to_unranked_borrow(), t.to_unranked_borrow())
            .attach_rank::<1>()
            .unwrap()
            .to_vector()
            .iter()
            .map(|x| x.clone().to_scalar().clone_real_part().into_inner())
            .collect::<Vec<_>>();
        assert_eq!(result, [13.23, 26.54])
    }

    #[test]
    fn test_relu() {
        let weights = to_not_nan_2([
            [7.1, 4.3, -6.4],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-1.3, -2.4, -3.6],
        ]);
        let biases = to_not_nan_1([10.2, 11.3, 12.4, 13.5]);
        let inputs = to_not_nan_1([7.0, 8.0, 9.0]);
        let theta0 = RankedDifferentiable::of_slice_2::<_, 2>(&weights);
        let theta1 = RankedDifferentiable::of_slice(&biases);
        let t = RankedDifferentiable::of_slice(&inputs);

        let result = relu(&t, &theta0, &theta1)
            .into_vector()
            .iter()
            .map(|x| x.borrow_scalar().clone_real_part().into_inner())
            .collect::<Vec<_>>();

        let mut expected = Vec::new();
        for (weights, bias) in weights.iter().zip(biases.iter()) {
            expected.push(
                crate::decider::relu(
                    &t,
                    &RankedDifferentiable::of_slice(weights),
                    Scalar::make(bias.clone()),
                )
                .clone_real_part()
                .into_inner(),
            );
        }

        assert_eq!(result, expected);
    }
}
