use crate::auto_diff::{Differentiable, DifferentiableTagged, RankedDifferentiable};
use std::iter::Sum;
use std::ops::Mul;

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

pub fn elementwise_mul_via_ext<A, const RANK1: usize, const RANK2: usize>(
    x: &RankedDifferentiable<A, RANK1>,
    y: &RankedDifferentiable<A, RANK2>,
) -> RankedDifferentiable<A, RANK1>
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Clone + Default,
{
    ext2(
        0,
        0,
        &mut |x, y| Differentiable::of_scalar(x.clone().into_scalar() * y.clone().into_scalar()),
        x.to_unranked_borrow(),
        y.to_unranked_borrow(),
    )
    .attach_rank::<RANK1>()
    .unwrap()
}

/// Produce the matrix multiplication of the inputs, threading where necessary until the
/// first argument has rank 2 and the second argument has rank 1.
pub fn star_2_1<T>(x: &Differentiable<T>, y: &Differentiable<T>) -> Differentiable<T>
where
    T: Clone + Sum + Mul<Output = T> + Default,
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

#[cfg(test)]
mod tests {
    use crate::auto_diff::{Differentiable, RankedDifferentiable};
    use crate::ext::{elementwise_mul_via_ext, ext1, ext2, star_2_1};
    use crate::not_nan::{to_not_nan_1, to_not_nan_2};
    use crate::scalar::Scalar;
    use crate::traits::Zero;
    use ordered_float::NotNan;
    use std::iter::Sum;
    use std::ops::Mul;

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
}
