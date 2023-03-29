use crate::scalar::Scalar;
use crate::traits::{Exp, One, Zero};
use core::hash::Hash;
use std::collections::HashMap;
use std::{
    fmt::{Display, Write},
    ops::{AddAssign, Div, Mul, Neg},
};

impl<A> Zero for DifferentiableHidden<A>
where
    A: Zero,
{
    fn zero() -> DifferentiableHidden<A> {
        DifferentiableHidden::Scalar(Scalar::Number(A::zero()))
    }
}

impl<A> One for Scalar<A>
where
    A: One,
{
    fn one() -> Scalar<A> {
        Scalar::Number(A::one())
    }
}

impl<A> One for DifferentiableHidden<A>
where
    A: One,
{
    fn one() -> DifferentiableHidden<A> {
        DifferentiableHidden::Scalar(Scalar::one())
    }
}

impl<A> Clone for DifferentiableHidden<A>
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

enum DifferentiableHidden<A> {
    Scalar(Scalar<A>),
    Vector(Vec<DifferentiableHidden<A>>),
}

impl<A> Display for DifferentiableHidden<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DifferentiableHidden::Scalar(s) => f.write_fmt(format_args!("{}", s)),
            DifferentiableHidden::Vector(v) => {
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

impl<A> DifferentiableHidden<A> {
    fn map<B, F>(&self, f: &F) -> DifferentiableHidden<B>
    where
        F: Fn(Scalar<A>) -> Scalar<B>,
        A: Clone,
    {
        match self {
            DifferentiableHidden::Scalar(a) => DifferentiableHidden::Scalar(f(a.clone())),
            DifferentiableHidden::Vector(slice) => {
                DifferentiableHidden::Vector(slice.iter().map(|x| x.map(f)).collect())
            }
        }
    }

    fn map2<B, C, F>(&self, other: &DifferentiableHidden<B>, f: &F) -> DifferentiableHidden<C>
    where
        F: Fn(&Scalar<A>, &Scalar<B>) -> Scalar<C>,
        A: Clone,
        B: Clone,
    {
        match (self, other) {
            (DifferentiableHidden::Scalar(a), DifferentiableHidden::Scalar(b)) => {
                DifferentiableHidden::Scalar(f(a, b))
            }
            (DifferentiableHidden::Vector(slice_a), DifferentiableHidden::Vector(slice_b)) => {
                DifferentiableHidden::Vector(
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

    fn of_slice(input: &[A]) -> DifferentiableHidden<A>
    where
        A: Clone,
    {
        DifferentiableHidden::Vector(
            input
                .iter()
                .map(|v| DifferentiableHidden::Scalar(Scalar::Number((*v).clone())))
                .collect(),
        )
    }
}

impl<A> DifferentiableHidden<A>
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
    fn accumulate_gradients_vec(v: &[DifferentiableHidden<A>], acc: &mut HashMap<Scalar<A>, A>) {
        for v in v.iter().rev() {
            v.accumulate_gradients(acc);
        }
    }

    fn accumulate_gradients(&self, acc: &mut HashMap<Scalar<A>, A>) {
        match self {
            DifferentiableHidden::Scalar(y) => {
                let k = y.clone_link();
                k.invoke(y, A::one(), acc);
            }
            DifferentiableHidden::Vector(y) => {
                DifferentiableHidden::accumulate_gradients_vec(y, acc)
            }
        }
    }

    fn grad_once(self, wrt: &DifferentiableHidden<A>) -> DifferentiableHidden<A> {
        let mut acc = HashMap::new();
        self.accumulate_gradients(&mut acc);

        wrt.map(&|d| match acc.get(&d) {
            None => Scalar::Number(A::zero()),
            Some(x) => Scalar::Number(x.clone()),
        })
    }
}

#[derive(Clone)]
pub struct Differentiable<A, const RANK: usize> {
    contents: DifferentiableHidden<A>,
}

impl<A, const RANK: usize> Display for Differentiable<A, RANK>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.contents, f)
    }
}

pub fn of_scalar<A>(s: Scalar<A>) -> Differentiable<A, 0> {
    Differentiable {
        contents: DifferentiableHidden::Scalar(s),
    }
}

pub fn to_scalar<A>(s: Differentiable<A, 0>) -> Scalar<A> {
    match s.contents {
        DifferentiableHidden::Scalar(s) => s,
        DifferentiableHidden::Vector(_) => panic!("not a vector"),
    }
}

pub fn of_slice<A>(input: &[A]) -> Differentiable<A, 1>
where
    A: Clone,
{
    Differentiable {
        contents: DifferentiableHidden::of_slice(input),
    }
}

impl<A, const RANK: usize> Differentiable<A, RANK> {
    pub fn of_vector(s: Vec<Differentiable<A, RANK>>) -> Differentiable<A, { RANK + 1 }> {
        Differentiable {
            contents: DifferentiableHidden::Vector(s.into_iter().map(|v| v.contents).collect()),
        }
    }

    pub fn map<B, F>(s: Differentiable<A, RANK>, f: &F) -> Differentiable<B, RANK>
    where
        F: Fn(Scalar<A>) -> Scalar<B>,
        A: Clone,
    {
        Differentiable {
            contents: DifferentiableHidden::map(&s.contents, f),
        }
    }

    pub fn map2<B, C, F>(
        self: &Differentiable<A, RANK>,
        other: &Differentiable<B, RANK>,
        f: &F,
    ) -> Differentiable<C, RANK>
    where
        F: Fn(&Scalar<A>, &Scalar<B>) -> Scalar<C>,
        A: Clone,
        B: Clone,
    {
        Differentiable {
            contents: DifferentiableHidden::map2(&self.contents, &other.contents, f),
        }
    }

    pub fn to_vector(s: Differentiable<A, { RANK + 1 }>) -> Vec<Differentiable<A, RANK>> {
        match s.contents {
            DifferentiableHidden::Scalar(_) => panic!("not a scalar"),
            DifferentiableHidden::Vector(v) => v
                .into_iter()
                .map(|v| Differentiable { contents: v })
                .collect(),
        }
    }

    pub fn grad<F>(f: F, theta: Differentiable<A, RANK>) -> Differentiable<A, RANK>
    where
        F: Fn(Differentiable<A, RANK>) -> Differentiable<A, RANK>,
        A: Clone
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
        let wrt = theta.contents.map(&Scalar::truncate_dual);
        let after_f = f(Differentiable {
            contents: wrt.clone(),
        });
        Differentiable {
            contents: DifferentiableHidden::grad_once(after_f.contents, &wrt),
        }
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use super::*;

    fn extract_scalar<'a, A>(d: &'a DifferentiableHidden<A>) -> &'a A {
        match d {
            DifferentiableHidden::Scalar(a) => &(a.real_part()),
            DifferentiableHidden::Vector(_) => panic!("not a scalar"),
        }
    }

    #[test]
    fn test_map() {
        let v = DifferentiableHidden::Vector(
            vec![
                DifferentiableHidden::Scalar(Scalar::Number(
                    NotNan::new(3.0).expect("3 is not NaN"),
                )),
                DifferentiableHidden::Scalar(Scalar::Number(
                    NotNan::new(4.0).expect("4 is not NaN"),
                )),
            ]
            .into(),
        );
        let mapped = v.map(&|x: Scalar<NotNan<f64>>| match x {
            Scalar::Number(i) => Scalar::Number(i + NotNan::new(1.0).expect("1 is not NaN")),
            Scalar::Dual(_, _) => panic!("Not hit"),
        });

        let v = match mapped {
            DifferentiableHidden::Scalar(_) => panic!("Not a scalar"),
            DifferentiableHidden::Vector(v) => v
                .iter()
                .map(|d| extract_scalar(d).clone())
                .collect::<Vec<_>>(),
        };

        assert_eq!(v, [4.0, 5.0]);
    }
}
