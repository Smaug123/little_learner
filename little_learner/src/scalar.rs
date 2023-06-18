use crate::traits::{Exp, One, Sqrt, Zero};
use core::hash::Hash;
use std::cmp::Ordering;
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
    Div(Box<Scalar<A>>, Box<Scalar<A>>),
    Sqrt(Box<Scalar<A>>),
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
            Link::Link(LinkData::Sqrt(arg)) => f.write_fmt(format_args!("sqrt({})", arg.as_ref())),
            Link::Link(LinkData::Div(left, right)) => {
                f.write_fmt(format_args!("({} / {})", left.as_ref(), right.as_ref()))
            }
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
            + Add<Output = A>
            + Mul<Output = A>
            + Div<Output = A>
            + Neg<Output = A>
            + Sqrt
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
                    LinkData::Div(left, right) => {
                        // d/dx(f / g) = f d(1/g)/dx + (df/dx) / g
                        //             = -f (dg/dx)/g^2 + (df/dx) / g
                        left.as_ref().clone_link().invoke(
                            &left,
                            z.clone() / right.clone_real_part(),
                            acc,
                        );
                        right.as_ref().clone_link().invoke(
                            &right,
                            -left.clone_real_part() * z
                                / (right.clone_real_part() * right.clone_real_part()),
                            acc,
                        );
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
                    LinkData::Sqrt(arg) => {
                        // d/dx(y^(1/2)) = 1/2 y^(-1/2) dy/dx
                        let two = A::one() + A::one();
                        arg.as_ref().clone_link().invoke(
                            &arg,
                            A::one() / (two * arg.as_ref().clone_real_part().sqrt()) * z,
                            acc,
                        );
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

impl<A> AddAssign for Scalar<A>
where
    A: Add<Output = A> + Clone,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
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
            answer += i;
        }
        answer
    }
}

impl<A> PartialOrd for Scalar<A>
where
    A: PartialOrd + Clone,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.real_part().partial_cmp(other.real_part())
    }
}

impl<A> Exp for Scalar<A>
where
    A: Exp + Clone,
{
    fn exp(self) -> Self {
        Self::Dual(
            self.clone_real_part().exp(),
            Link::Link(LinkData::Exponent(Box::new(self))),
        )
    }
}

impl<A> Div for Scalar<A>
where
    A: Div<Output = A> + Clone,
{
    type Output = Scalar<A>;

    fn div(self, rhs: Self) -> Self::Output {
        Self::Dual(
            self.clone_real_part() / rhs.clone_real_part(),
            Link::Link(LinkData::Div(Box::new(self), Box::new(rhs))),
        )
    }
}

impl<A> Sqrt for Scalar<A>
where
    A: Sqrt + Clone,
{
    fn sqrt(self) -> Self {
        Self::Dual(
            self.clone_real_part().sqrt(),
            Link::Link(LinkData::Sqrt(Box::new(self))),
        )
    }
}

impl<A> Default for Scalar<A>
where
    A: Default,
{
    fn default() -> Self {
        Scalar::Number(A::default(), None)
    }
}

impl<A> Scalar<A> {
    pub fn real_part(&self) -> &A {
        match self {
            Scalar::Number(a, _) | Scalar::Dual(a, _) => a,
        }
    }

    pub fn clone_real_part(&self) -> A
    where
        A: Clone,
    {
        match self {
            Scalar::Number(a, _) | Scalar::Dual(a, _) => (*a).clone(),
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

    #[must_use]
    pub fn truncate_dual(self, index: Option<usize>) -> Scalar<A>
    where
        A: Clone,
    {
        Scalar::Dual(self.clone_real_part(), Link::EndOfLink(index))
    }

    #[must_use]
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
            Scalar::Number(n, Some(index)) => f.write_fmt(format_args!("{n}_{index}")),
            Scalar::Number(n, None) => f.write_fmt(format_args!("{n}")),
            Scalar::Dual(n, _link) => f.write_fmt(format_args!("{n}")),
        }
    }
}

#[cfg(test)]
mod test_loss {
    use crate::auto_diff::{grad, Differentiable, RankedDifferentiable};
    use crate::scalar::Scalar;
    use crate::traits::Sqrt;
    use ordered_float::NotNan;
    use std::collections::HashMap;

    #[test]
    fn div_gradient() {
        let left = Scalar::make(NotNan::new(3.0).expect("not nan"));
        let right = Scalar::make(NotNan::new(5.0).expect("not nan"));
        let divided = left / right;
        assert_eq!(divided.clone_real_part().into_inner(), 3.0 / 5.0);
        let mut acc = HashMap::new();
        divided
            .clone_link()
            .invoke(&divided, NotNan::new(1.0).expect("not nan"), &mut acc);

        // Derivative of x/5 with respect to x is the constant 1/5
        // Derivative of 3/x with respect to x is -3/x^2, so at the value 5 is -3/25
        assert_eq!(acc.len(), 2);
        for (key, value) in acc {
            let key = key.real_part().into_inner();
            let value = value.into_inner();
            if key < 4.0 {
                // This is the numerator.
                assert_eq!(key, 3.0);
                assert_eq!(value, 1.0 / 5.0);
            } else {
                // This is the denominator.
                assert_eq!(key, 5.0);
                assert_eq!(value, -3.0 / 25.0);
            }
        }
    }

    #[test]
    fn sqrt_gradient() {
        let nine = Differentiable::of_scalar(Scalar::make(NotNan::new(9.0).expect("not nan")));
        let graded: [Differentiable<NotNan<f64>>; 1] = grad(
            |x| Differentiable::of_scalar(x[0].clone().into_scalar().sqrt()),
            &[nine],
        );
        let graded = graded.map(|x| x.into_scalar().clone_real_part().into_inner())[0];

        // Derivative of sqrt(x) with respect to x at 3 is 1/6
        assert_eq!(graded, 1.0 / 6.0);
    }
}
