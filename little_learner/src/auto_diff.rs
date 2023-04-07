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
        Differentiable::Scalar(Scalar::Number(A::zero(), None))
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
        Differentiable::Scalar(Scalar::one())
    }
}

impl<A> Clone for Differentiable<A>
where
    A: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Scalar(arg0) => Self::Scalar(arg0.clone()),
            Self::Vector(arg0) => Self::Vector(arg0.clone()),
        }
    }
}

#[derive(Debug)]
pub enum Differentiable<A> {
    Scalar(Scalar<A>),
    Vector(Vec<Differentiable<A>>),
}

impl<A> Display for Differentiable<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Differentiable::Scalar(s) => f.write_fmt(format_args!("{}", s)),
            Differentiable::Vector(v) => {
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

impl<A> Differentiable<A> {
    pub fn map<B, F>(&self, f: &mut F) -> Differentiable<B>
    where
        F: FnMut(Scalar<A>) -> Scalar<B>,
        A: Clone,
    {
        match self {
            Differentiable::Scalar(a) => Differentiable::Scalar(f(a.clone())),
            Differentiable::Vector(slice) => {
                Differentiable::Vector(slice.iter().map(|x| x.map(f)).collect())
            }
        }
    }

    pub fn map2<B, C, F>(&self, other: &Differentiable<B>, f: &F) -> Differentiable<C>
    where
        F: Fn(&Scalar<A>, &Scalar<B>) -> Scalar<C>,
        A: Clone,
        B: Clone,
    {
        match (self, other) {
            (Differentiable::Scalar(a), Differentiable::Scalar(b)) => {
                Differentiable::Scalar(f(a, b))
            }
            (Differentiable::Vector(slice_a), Differentiable::Vector(slice_b)) => {
                Differentiable::Vector(
                    slice_a
                        .iter()
                        .zip(slice_b.iter())
                        .map(|(a, b)| a.map2(b, f))
                        .collect(),
                )
            }
            _ => panic!("Wrong shapes!"),
        }
    }

    fn of_slice<T>(input: T) -> Differentiable<A>
    where
        A: Clone,
        T: AsRef<[A]>,
    {
        Differentiable::Vector(
            input
                .as_ref()
                .iter()
                .map(|v| Differentiable::Scalar(Scalar::Number((*v).clone(), None)))
                .collect(),
        )
    }

    pub fn rank(&self) -> usize {
        match self {
            Differentiable::Scalar(_) => 0,
            Differentiable::Vector(v) => v[0].rank() + 1,
        }
    }

    pub fn attach_rank<const RANK: usize>(self: Differentiable<A>) -> Option<RankedDifferentiable<A, RANK>> {
        if self.rank() == RANK {
            Some(RankedDifferentiable { contents: self })
        } else {
            None
        }
    }
}

impl<A> Differentiable<A> {
    pub fn into_scalar(self) -> Scalar<A> {
        match self {
            Differentiable::Scalar(s) => s,
            Differentiable::Vector(_) => panic!("not a scalar"),
        }
    }

    pub fn into_vector(self) -> Vec<Differentiable<A>> {
        match self {
            Differentiable::Scalar(_) => panic!("not a vector"),
            Differentiable::Vector(v) => v,
        }
    }

    pub fn borrow_scalar(&self) -> &Scalar<A> {
        match self {
            Differentiable::Scalar(s) => s,
            Differentiable::Vector(_) => panic!("not a scalar"),
        }
    }

    pub fn borrow_vector(&self) -> &Vec<Differentiable<A>> {
        match self {
            Differentiable::Scalar(_) => panic!("not a vector"),
            Differentiable::Vector(v) => v,
        }
    }
}

impl<A> Differentiable<A>
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
            v.accumulate_gradients(acc);
        }
    }

    fn accumulate_gradients(&self, acc: &mut HashMap<Scalar<A>, A>) {
        match self {
            Differentiable::Scalar(y) => {
                let k = y.clone_link();
                k.invoke(y, A::one(), acc);
            }
            Differentiable::Vector(y) => Differentiable::accumulate_gradients_vec(y, acc),
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
        match self.contents {
            Differentiable::Scalar(s) => s,
            Differentiable::Vector(_) => panic!("not a scalar despite teq that we're a scalar"),
        }
    }

    pub fn of_scalar(s: Scalar<A>) -> RankedDifferentiable<A, 0> {
        RankedDifferentiable {
            contents: Differentiable::Scalar(s),
        }
    }
}

impl<A> RankedDifferentiable<A, 1> {
    pub fn of_slice<T>(input: T) -> RankedDifferentiable<A, 1>
    where
        A: Clone,
        T: AsRef<[A]>,
    {
        RankedDifferentiable {
            contents: Differentiable::of_slice(input),
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
            .map(|x| Differentiable::of_slice(x))
            .collect::<Vec<_>>();
        RankedDifferentiable {
            contents: Differentiable::Vector(v),
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
            contents: Differentiable::Vector(s.into_iter().map(|v| v.contents).collect()),
        }
    }

    pub fn map<B, F>(self: RankedDifferentiable<A, RANK>, f: &mut F) -> RankedDifferentiable<B, RANK>
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
        f: &F,
    ) -> RankedDifferentiable<C, RANK>
    where
        F: Fn(&Scalar<A>, &Scalar<B>) -> Scalar<C>,
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
        match self.contents {
            Differentiable::Scalar(_) => panic!("not a scalar"),
            Differentiable::Vector(v) => v
                .into_iter()
                .map(|v| RankedDifferentiable { contents: v })
                .collect(),
        }
    }
}

pub fn grad<A, F, const RANK: usize, const PARAM_RANK: usize>(
    f: F,
    theta: &[Differentiable<A>; PARAM_RANK],
) -> [Differentiable<A>; PARAM_RANK]
where
    F: Fn(&[Differentiable<A>; PARAM_RANK]) -> RankedDifferentiable<A, RANK>,
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
    Differentiable::grad_once(after_f.contents, wrt)
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use crate::loss::{l2_loss_2, predict_line_2_unranked};

    use super::*;

    fn extract_scalar<'a, A>(d: &'a Differentiable<A>) -> &'a A {
        match d {
            Differentiable::Scalar(a) => &(a.real_part()),
            Differentiable::Vector(_) => panic!("not a scalar"),
        }
    }

    #[test]
    fn test_map() {
        let v = Differentiable::Vector(
            vec![
                Differentiable::Scalar(Scalar::Number(
                    NotNan::new(3.0).expect("3 is not NaN"),
                    Some(0usize),
                )),
                Differentiable::Scalar(Scalar::Number(
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

        let v = match mapped {
            Differentiable::Scalar(_) => panic!("Not a scalar"),
            Differentiable::Vector(v) => v
                .iter()
                .map(|d| extract_scalar(d).clone())
                .collect::<Vec<_>>(),
        };

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
                    RankedDifferentiable::of_slice(&xs),
                    RankedDifferentiable::of_slice(&ys),
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
}
