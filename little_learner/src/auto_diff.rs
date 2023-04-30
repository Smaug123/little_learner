use crate::scalar::Scalar;
use crate::traits::{Exp, One, Zero};
use core::hash::Hash;
use std::collections::HashMap;
use std::{
    fmt::{Display, Write},
    ops::{AddAssign, Div, Mul, Neg},
};

impl<A> Zero for Differentiable<A>
where
    A: Zero,
{
    fn zero() -> Differentiable<A> {
        Differentiable {
            contents: DifferentiableContents::Scalar(Scalar::Number(A::zero(), None)),
        }
    }
}

impl<A> One for Scalar<A>
where
    A: One,
{
    fn one() -> Scalar<A> {
        Scalar::Number(A::one(), None)
    }
}

impl<A> One for Differentiable<A>
where
    A: One,
{
    fn one() -> Differentiable<A> {
        Differentiable {
            contents: DifferentiableContents::Scalar(Scalar::one()),
        }
    }
}

impl<A> Clone for DifferentiableContents<A>
where
    A: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Scalar(arg0) => Self::Scalar(arg0.clone()),
            Self::Vector(arg0, rank) => Self::Vector(arg0.clone(), *rank),
        }
    }
}

impl<A> Clone for Differentiable<A>
where
    A: Clone,
{
    fn clone(&self) -> Self {
        Differentiable {
            contents: self.contents.clone(),
        }
    }
}

#[derive(Debug)]
enum DifferentiableContents<A> {
    Scalar(Scalar<A>),
    // Contains the rank.
    Vector(Vec<Differentiable<A>>, usize),
}

#[derive(Debug)]
pub struct Differentiable<A> {
    contents: DifferentiableContents<A>,
}

impl<A> Display for DifferentiableContents<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DifferentiableContents::Scalar(s) => f.write_fmt(format_args!("{}", s)),
            DifferentiableContents::Vector(v, _rank) => {
                f.write_char('[')?;
                for v in v.iter() {
                    f.write_fmt(format_args!("{}", v))?;
                    f.write_char(',')?;
                }
                f.write_char(']')
            }
        }
    }
}

impl<A> Display for Differentiable<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.contents))
    }
}

impl<A> DifferentiableContents<A> {
    fn map<B, F>(&self, f: &mut F) -> DifferentiableContents<B>
    where
        F: FnMut(Scalar<A>) -> Scalar<B>,
        A: Clone,
    {
        match self {
            DifferentiableContents::Scalar(a) => DifferentiableContents::Scalar(f(a.clone())),
            DifferentiableContents::Vector(slice, rank) => {
                DifferentiableContents::Vector(slice.iter().map(|x| x.map(f)).collect(), *rank)
            }
        }
    }

    fn map2<B, C, F>(
        &self,
        other: &DifferentiableContents<B>,
        f: &mut F,
    ) -> DifferentiableContents<C>
    where
        F: FnMut(&Scalar<A>, &Scalar<B>) -> Scalar<C>,
        A: Clone,
        B: Clone,
    {
        match (self, other) {
            (DifferentiableContents::Scalar(a), DifferentiableContents::Scalar(b)) => {
                DifferentiableContents::Scalar(f(a, b))
            }
            (
                DifferentiableContents::Vector(slice_a, rank_a),
                DifferentiableContents::Vector(slice_b, rank_b),
            ) => {
                if rank_a != rank_b {
                    panic!("Unexpectedly different ranks in map2");
                }
                DifferentiableContents::Vector(
                    slice_a
                        .iter()
                        .zip(slice_b.iter())
                        .map(|(a, b)| a.map2(b, f))
                        .collect(),
                    *rank_a,
                )
            }
            _ => panic!("Wrong shapes!"),
        }
    }

    fn of_slice<'a, T, I>(input: I) -> DifferentiableContents<T>
    where
        T: Clone + 'a,
        I: IntoIterator<Item = &'a T>,
    {
        DifferentiableContents::Vector(
            input
                .into_iter()
                .map(|v| Differentiable {
                    contents: DifferentiableContents::Scalar(Scalar::Number(v.clone(), None)),
                })
                .collect(),
            1,
        )
    }

    fn rank(&self) -> usize {
        match self {
            DifferentiableContents::Scalar(_) => 0,
            DifferentiableContents::Vector(_, rank) => *rank,
        }
    }
}

impl<A> Differentiable<A> {
    pub fn map<B, F>(&self, f: &mut F) -> Differentiable<B>
    where
        A: Clone,
        F: FnMut(Scalar<A>) -> Scalar<B>,
    {
        Differentiable {
            contents: self.contents.map(f),
        }
    }

    pub fn map2<B, C, F>(&self, other: &Differentiable<B>, f: &mut F) -> Differentiable<C>
    where
        F: FnMut(&Scalar<A>, &Scalar<B>) -> Scalar<C>,
        A: Clone,
        B: Clone,
    {
        Differentiable {
            contents: self.contents.map2(&other.contents, f),
        }
    }

    pub fn attach_rank<const RANK: usize>(
        self: Differentiable<A>,
    ) -> Option<RankedDifferentiable<A, RANK>> {
        if self.contents.rank() == RANK {
            Some(RankedDifferentiable { contents: self })
        } else {
            None
        }
    }

    pub fn of_scalar(s: Scalar<A>) -> Differentiable<A> {
        Differentiable {
            contents: DifferentiableContents::Scalar(s),
        }
    }
}

impl<A> DifferentiableContents<A> {
    fn into_scalar(self) -> Scalar<A> {
        match self {
            DifferentiableContents::Scalar(s) => s,
            DifferentiableContents::Vector(_, _) => panic!("not a scalar"),
        }
    }

    fn into_vector(self) -> Vec<Differentiable<A>> {
        match self {
            DifferentiableContents::Scalar(_) => panic!("not a vector"),
            DifferentiableContents::Vector(v, _) => v,
        }
    }

    fn borrow_scalar(&self) -> &Scalar<A> {
        match self {
            DifferentiableContents::Scalar(s) => s,
            DifferentiableContents::Vector(_, _) => panic!("not a scalar"),
        }
    }

    fn borrow_vector(&self) -> &Vec<Differentiable<A>> {
        match self {
            DifferentiableContents::Scalar(_) => panic!("not a vector"),
            DifferentiableContents::Vector(v, _) => v,
        }
    }
}

impl<A> Differentiable<A> {
    pub fn into_scalar(self) -> Scalar<A> {
        self.contents.into_scalar()
    }

    pub fn into_vector(self) -> Vec<Differentiable<A>> {
        self.contents.into_vector()
    }

    pub fn borrow_scalar(&self) -> &Scalar<A> {
        self.contents.borrow_scalar()
    }

    pub fn borrow_vector(&self) -> &Vec<Differentiable<A>> {
        self.contents.borrow_vector()
    }

    fn of_slice<'a, T>(input: T) -> Differentiable<A>
    where
        A: Clone + 'a,
        T: IntoIterator<Item = &'a A>,
    {
        Differentiable {
            contents: DifferentiableContents::<A>::of_slice(input),
        }
    }

    pub fn of_vec(input: Vec<Differentiable<A>>) -> Differentiable<A> {
        if input.is_empty() {
            panic!("Can't make an empty tensor");
        }
        let rank = input[0].rank();
        Differentiable {
            contents: DifferentiableContents::Vector(input, 1 + rank),
        }
    }

    pub fn rank(&self) -> usize {
        self.contents.rank()
    }
}

impl<A> DifferentiableContents<A>
where
    A: Clone
        + Eq
        + Hash
        + AddAssign
        + Mul<Output = A>
        + Exp
        + Div<Output = A>
        + Zero
        + One
        + Neg<Output = A>,
{
    fn accumulate_gradients_vec(v: &[Differentiable<A>], acc: &mut HashMap<Scalar<A>, A>) {
        for v in v.iter().rev() {
            v.contents.accumulate_gradients(acc);
        }
    }

    fn accumulate_gradients(&self, acc: &mut HashMap<Scalar<A>, A>) {
        match self {
            DifferentiableContents::Scalar(y) => {
                let k = y.clone_link();
                k.invoke(y, A::one(), acc);
            }
            DifferentiableContents::Vector(y, _rank) => {
                DifferentiableContents::accumulate_gradients_vec(y, acc)
            }
        }
    }

    fn grad_once<const PARAM_NUM: usize>(
        self,
        wrt: [Differentiable<A>; PARAM_NUM],
    ) -> [Differentiable<A>; PARAM_NUM] {
        let mut acc = HashMap::new();
        self.accumulate_gradients(&mut acc);

        wrt.map(|wrt| {
            wrt.map(&mut |d| match acc.get(&d) {
                None => Scalar::Number(A::zero(), None),
                Some(x) => Scalar::Number(x.clone(), None),
            })
        })
    }
}

#[derive(Clone, Debug)]
pub struct RankedDifferentiable<A, const RANK: usize> {
    contents: Differentiable<A>,
}

impl<A, const RANK: usize> Display for RankedDifferentiable<A, RANK>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.contents, f)
    }
}

impl<A> RankedDifferentiable<A, 0> {
    pub fn to_scalar(self) -> Scalar<A> {
        self.contents.contents.into_scalar()
    }

    pub fn of_scalar(s: Scalar<A>) -> RankedDifferentiable<A, 0> {
        RankedDifferentiable {
            contents: Differentiable::of_scalar(s),
        }
    }
}

impl<A> RankedDifferentiable<A, 1> {
    pub fn of_slice<'a, T>(input: T) -> RankedDifferentiable<A, 1>
    where
        A: Clone + 'a,
        T: IntoIterator<Item = &'a A>,
    {
        RankedDifferentiable {
            contents: Differentiable {
                contents: DifferentiableContents::<A>::of_slice(input),
            },
        }
    }
}

impl<A> RankedDifferentiable<A, 2> {
    pub fn of_slice_2<T, const N: usize>(input: &[T]) -> RankedDifferentiable<A, 2>
    where
        A: Clone,
        T: AsRef<[A]>,
    {
        let v = input
            .iter()
            .map(|x| Differentiable::of_slice(x.as_ref()))
            .collect::<Vec<_>>();
        RankedDifferentiable {
            contents: Differentiable::of_vec(v),
        }
    }
}

impl<A, const RANK: usize> RankedDifferentiable<A, RANK> {
    pub fn to_unranked(self) -> Differentiable<A> {
        self.contents
    }

    pub fn to_unranked_borrow(&self) -> &Differentiable<A> {
        &self.contents
    }

    pub fn of_vector(
        s: Vec<RankedDifferentiable<A, RANK>>,
    ) -> RankedDifferentiable<A, { RANK + 1 }> {
        RankedDifferentiable {
            contents: Differentiable::of_vec(s.into_iter().map(|v| v.contents).collect()),
        }
    }

    pub fn map<B, F>(
        self: RankedDifferentiable<A, RANK>,
        f: &mut F,
    ) -> RankedDifferentiable<B, RANK>
    where
        F: FnMut(Scalar<A>) -> Scalar<B>,
        A: Clone,
    {
        RankedDifferentiable {
            contents: Differentiable::map(&self.contents, f),
        }
    }

    pub fn map2<B, C, F>(
        self: &RankedDifferentiable<A, RANK>,
        other: &RankedDifferentiable<B, RANK>,
        f: &mut F,
    ) -> RankedDifferentiable<C, RANK>
    where
        F: FnMut(&Scalar<A>, &Scalar<B>) -> Scalar<C>,
        A: Clone,
        B: Clone,
    {
        RankedDifferentiable {
            contents: Differentiable::map2(&self.contents, &other.contents, f),
        }
    }

    pub fn to_vector(
        self: RankedDifferentiable<A, RANK>,
    ) -> Vec<RankedDifferentiable<A, { RANK - 1 }>> {
        self.contents
            .into_vector()
            .into_iter()
            .map(|v| RankedDifferentiable { contents: v })
            .collect()
    }
}

pub fn grad<A, F, const RANK: usize, const PARAM_RANK: usize>(
    mut f: F,
    theta: &[Differentiable<A>; PARAM_RANK],
) -> [Differentiable<A>; PARAM_RANK]
where
    F: FnMut(&[Differentiable<A>; PARAM_RANK]) -> RankedDifferentiable<A, RANK>,
    A: ?Sized
        + Clone
        + Hash
        + AddAssign
        + Mul<Output = A>
        + Exp
        + Div<Output = A>
        + Zero
        + One
        + Neg<Output = A>
        + Eq,
{
    let mut i = 0usize;
    let wrt = theta.each_ref().map(|theta| {
        theta.map(&mut |x| {
            let result = Scalar::truncate_dual(x, Some(i));
            i += 1;
            result
        })
    });
    let after_f = f(&wrt);
    DifferentiableContents::grad_once(after_f.contents.contents, wrt)
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use crate::loss::{l2_loss_2, predict_line_2_unranked};
    use crate::not_nan::to_not_nan_1;

    use super::*;

    fn extract_scalar<A>(d: &Differentiable<A>) -> &A {
        d.borrow_scalar().real_part()
    }

    #[test]
    fn test_map() {
        let v = Differentiable::of_vec(
            vec![
                Differentiable::of_scalar(Scalar::Number(
                    NotNan::new(3.0).expect("3 is not NaN"),
                    Some(0usize),
                )),
                Differentiable::of_scalar(Scalar::Number(
                    NotNan::new(4.0).expect("4 is not NaN"),
                    Some(1usize),
                )),
            ]
            .into(),
        );
        let mapped = v.map(&mut |x: Scalar<NotNan<f64>>| match x {
            Scalar::Number(i, n) => Scalar::Number(i + NotNan::new(1.0).expect("1 is not NaN"), n),
            Scalar::Dual(_, _) => panic!("Not hit"),
        });

        let v = mapped
            .into_vector()
            .iter()
            .map(|d| extract_scalar(d).clone())
            .collect::<Vec<_>>();

        assert_eq!(v, [4.0, 5.0]);
    }

    #[test]
    fn test_autodiff() {
        let input_vec = [
            RankedDifferentiable::of_scalar(Scalar::<NotNan<f64>>::zero()).contents,
            RankedDifferentiable::of_scalar(Scalar::<NotNan<f64>>::zero()).contents,
        ];
        let xs = [2.0, 1.0, 4.0, 3.0].map(|x| NotNan::new(x).expect("not nan"));
        let ys = [1.8, 1.2, 4.2, 3.3].map(|x| NotNan::new(x).expect("not nan"));
        let grad = grad(
            |x| {
                RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(l2_loss_2(
                    predict_line_2_unranked,
                    RankedDifferentiable::of_slice(xs.iter()),
                    RankedDifferentiable::of_slice(ys.iter()),
                    x,
                ))])
            },
            &input_vec,
        );

        let grad_vec = grad
            .map(Differentiable::into_scalar)
            .map(|x| f64::from(*x.real_part()));
        assert_eq!(grad_vec, [-63.0, -21.0]);
    }

    #[test]
    fn grad_example() {
        let input_vec = [Differentiable::of_scalar(Scalar::make(
            NotNan::new(27.0).expect("not nan"),
        ))];

        let grad: Vec<_> = grad(
            |x| {
                RankedDifferentiable::of_scalar(
                    x[0].borrow_scalar().clone() * x[0].borrow_scalar().clone(),
                )
            },
            &input_vec,
        )
        .into_iter()
        .map(|x| x.into_scalar().real_part().into_inner())
        .collect();
        assert_eq!(grad, [54.0]);
    }

    #[test]
    fn loss_gradient() {
        let zero = Scalar::<NotNan<f64>>::zero();
        let input_vec = [
            RankedDifferentiable::of_scalar(zero.clone()).to_unranked(),
            RankedDifferentiable::of_scalar(zero).to_unranked(),
        ];
        let xs = to_not_nan_1([2.0, 1.0, 4.0, 3.0]);
        let ys = to_not_nan_1([1.8, 1.2, 4.2, 3.3]);
        let grad = grad(
            |x| {
                RankedDifferentiable::of_vector(vec![RankedDifferentiable::of_scalar(l2_loss_2(
                    predict_line_2_unranked,
                    RankedDifferentiable::of_slice(&xs),
                    RankedDifferentiable::of_slice(&ys),
                    x,
                ))])
            },
            &input_vec,
        );

        assert_eq!(
            grad.into_iter()
                .map(|x| *(x.into_scalar().real_part()))
                .collect::<Vec<_>>(),
            [-63.0, -21.0]
        );
    }
}
