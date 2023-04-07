use crate::traits::{Exp, One, Zero};
use core::hash::Hash;
use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum LinkData<A> {
    Addition(Box<Scalar<A>>, Box<Scalar<A>>),
    Neg(Box<Scalar<A>>),
    Mul(Box<Scalar<A>>, Box<Scalar<A>>),
    Exponent(Box<Scalar<A>>),
    Log(Box<Scalar<A>>),
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Link<A> {
    EndOfLink(Option<usize>),
    Link(LinkData<A>),
}

impl<A> Display for Link<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Link::EndOfLink(Some(i)) => f.write_fmt(format_args!("<end {}>", *i)),
            Link::EndOfLink(None) => f.write_str("<end>"),
            Link::Link(LinkData::Addition(left, right)) => {
                f.write_fmt(format_args!("({} + {})", left.as_ref(), right.as_ref()))
            }
            Link::Link(LinkData::Neg(arg)) => f.write_fmt(format_args!("(-{})", arg.as_ref())),
            Link::Link(LinkData::Mul(left, right)) => {
                f.write_fmt(format_args!("({} * {})", left.as_ref(), right.as_ref()))
            }
            Link::Link(LinkData::Exponent(arg)) => {
                f.write_fmt(format_args!("exp({})", arg.as_ref()))
            }
            Link::Link(LinkData::Log(arg)) => f.write_fmt(format_args!("log({})", arg.as_ref())),
        }
    }
}

impl<A> Link<A> {
    pub fn invoke(self, d: &Scalar<A>, z: A, acc: &mut HashMap<Scalar<A>, A>)
    where
        A: Eq
            + Hash
            + AddAssign
            + Clone
            + Exp
            + Mul<Output = A>
            + Div<Output = A>
            + Neg<Output = A>
            + Zero
            + One,
    {
        match self {
            Link::EndOfLink(_) => match acc.entry(d.clone()) {
                Entry::Occupied(mut o) => {
                    let entry = o.get_mut();
                    *entry += z;
                }
                Entry::Vacant(v) => {
                    v.insert(z);
                }
            },
            Link::Link(data) => {
                match data {
                    LinkData::Addition(left, right) => {
                        // The `z` here reflects the fact that dx/dx = 1, so it's 1 * z.
                        left.as_ref().clone_link().invoke(&left, z.clone(), acc);
                        right.as_ref().clone_link().invoke(&right, z, acc);
                    }
                    LinkData::Exponent(arg) => {
                        // d/dx (e^x) = exp x, so exp z * z.
                        arg.as_ref().clone_link().invoke(
                            &arg,
                            z * arg.clone_real_part().exp(),
                            acc,
                        );
                    }
                    LinkData::Mul(left, right) => {
                        // d/dx(f g) = f dg/dx + g df/dx
                        left.as_ref().clone_link().invoke(
                            &left,
                            right.clone_real_part() * z.clone(),
                            acc,
                        );
                        right
                            .as_ref()
                            .clone_link()
                            .invoke(&right, left.clone_real_part() * z, acc);
                    }
                    LinkData::Log(arg) => {
                        // d/dx(log y) = 1/y dy/dx
                        arg.as_ref().clone_link().invoke(
                            &arg,
                            A::one() / arg.clone_real_part() * z,
                            acc,
                        );
                    }
                    LinkData::Neg(arg) => {
                        // d/dx(-y) = - dy/dx
                        arg.as_ref().clone_link().invoke(&arg, -z, acc);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum Scalar<A> {
    Number(A, Option<usize>),
    // The value, and the link.
    Dual(A, Link<A>),
}

impl<A> Zero for Scalar<A>
where
    A: Zero,
{
    fn zero() -> Self {
        Scalar::Number(A::zero(), None)
    }
}

impl<A> Add for Scalar<A>
where
    A: Add<Output = A> + Clone,
{
    type Output = Scalar<A>;

    fn add(self, rhs: Self) -> Self::Output {
        Scalar::Dual(
            self.clone_real_part() + rhs.clone_real_part(),
            Link::Link(LinkData::Addition(Box::new(self), Box::new(rhs))),
        )
    }
}

impl<A> Neg for Scalar<A>
where
    A: Neg<Output = A> + Clone,
{
    type Output = Scalar<A>;

    fn neg(self) -> Self::Output {
        Scalar::Dual(
            -self.clone_real_part(),
            Link::Link(LinkData::Neg(Box::new(self))),
        )
    }
}

impl<A> Sub for Scalar<A>
where
    A: Add<Output = A> + Neg<Output = A> + Clone,
{
    type Output = Scalar<A>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<A> Mul for Scalar<A>
where
    A: Mul<Output = A> + Clone,
{
    type Output = Scalar<A>;

    fn mul(self, rhs: Self) -> Self::Output {
        Scalar::Dual(
            self.clone_real_part() * rhs.clone_real_part(),
            Link::Link(LinkData::Mul(Box::new(self), Box::new(rhs))),
        )
    }
}

impl<A> Sum for Scalar<A>
where
    A: Zero + Add<Output = A> + Clone,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut answer = Zero::zero();
        for i in iter {
            answer = answer + i;
        }
        answer
    }
}

impl<A> Scalar<A> {
    pub fn real_part(&self) -> &A {
        match self {
            Scalar::Number(a, _) => a,
            Scalar::Dual(a, _) => a,
        }
    }

    pub fn clone_real_part(&self) -> A
    where
        A: Clone,
    {
        match self {
            Scalar::Number(a, _) => (*a).clone(),
            Scalar::Dual(a, _) => (*a).clone(),
        }
    }

    pub fn link(self) -> Link<A> {
        match self {
            Scalar::Dual(_, link) => link,
            Scalar::Number(_, i) => Link::EndOfLink(i),
        }
    }

    pub fn clone_link(&self) -> Link<A>
    where
        A: Clone,
    {
        match self {
            Scalar::Dual(_, data) => data.clone(),
            Scalar::Number(_, i) => Link::EndOfLink(*i),
        }
    }

    pub fn truncate_dual(self, index: Option<usize>) -> Scalar<A>
    where
        A: Clone,
    {
        Scalar::Dual(self.clone_real_part(), Link::EndOfLink(index))
    }

    pub fn make(x: A) -> Scalar<A> {
        Scalar::Number(x, None)
    }
}

impl<A> Display for Scalar<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar::Number(n, Some(index)) => f.write_fmt(format_args!("{}_{}", n, index)),
            Scalar::Number(n, None) => f.write_fmt(format_args!("{}", n)),
            Scalar::Dual(n, link) => f.write_fmt(format_args!("<{}, link: {}>", n, link)),
        }
    }
}
