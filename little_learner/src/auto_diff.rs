use core::hash::Hash;
use ordered_float::NotNan;
use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::{Display, Write},
    ops::{Add, AddAssign, Div, Mul},
};

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

impl Zero for NotNan<f64> {
    fn zero() -> Self {
        NotNan::new(0.0).unwrap()
    }
}

impl One for NotNan<f64> {
    fn one() -> Self {
        NotNan::new(1.0).unwrap()
    }
}

impl<A> Zero for Differentiable<A>
where
    A: Zero,
{
    fn zero() -> Differentiable<A> {
        Differentiable::Scalar(Scalar::Number(A::zero()))
    }
}

impl<A> One for Differentiable<A>
where
    A: One,
{
    fn one() -> Differentiable<A> {
        Differentiable::Scalar(Scalar::Number(A::one()))
    }
}

pub trait Exp {
    fn exp(self) -> Self;
}

impl Exp for NotNan<f64> {
    fn exp(self) -> Self {
        NotNan::new(f64::exp(self.into_inner())).expect("expected a non-NaN")
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum LinkData<A> {
    Addition(Box<Scalar<A>>, Box<Scalar<A>>),
    Mul(Box<Scalar<A>>, Box<Scalar<A>>),
    Exponent(Box<Scalar<A>>),
    Log(Box<Scalar<A>>),
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum Link<A> {
    EndOfLink,
    Link(LinkData<A>),
}

impl<A> Display for Link<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Link::EndOfLink => f.write_str("<end>"),
            Link::Link(LinkData::Addition(left, right)) => {
                f.write_fmt(format_args!("({} + {})", left.as_ref(), right.as_ref()))
            }
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
    fn invoke(self, d: &Scalar<A>, z: A, acc: &mut HashMap<Scalar<A>, A>)
    where
        A: Eq + Hash + AddAssign + Clone + Exp + Mul<Output = A> + Div<Output = A> + Zero + One,
    {
        match self {
            Link::EndOfLink => match acc.entry(d.clone()) {
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
                }
            }
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum Scalar<A> {
    Number(A),
    // The value, and the link.
    Dual(A, Link<A>),
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

impl<A> Scalar<A> {
    pub fn real_part(&self) -> &A {
        match self {
            Scalar::Number(a) => a,
            Scalar::Dual(a, _) => a,
        }
    }

    fn clone_real_part(&self) -> A
    where
        A: Clone,
    {
        match self {
            Scalar::Number(a) => (*a).clone(),
            Scalar::Dual(a, _) => (*a).clone(),
        }
    }

    pub fn link(self) -> Link<A> {
        match self {
            Scalar::Dual(_, link) => link,
            Scalar::Number(_) => Link::EndOfLink,
        }
    }

    fn clone_link(&self) -> Link<A>
    where
        A: Clone,
    {
        match self {
            Scalar::Dual(_, data) => data.clone(),
            Scalar::Number(_) => Link::EndOfLink,
        }
    }

    fn truncate_dual(self) -> Scalar<A>
    where
        A: Clone,
    {
        Scalar::Dual(self.clone_real_part(), Link::EndOfLink)
    }
}

impl<A> Display for Scalar<A>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar::Number(n) => f.write_fmt(format_args!("{}", n)),
            Scalar::Dual(n, link) => f.write_fmt(format_args!("{}, link: {}", n, link)),
        }
    }
}

pub enum Differentiable<A> {
    Scalar(Scalar<A>),
    Vector(Box<[Differentiable<A>]>),
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
    pub fn map<B, F>(&self, f: &F) -> Differentiable<B>
    where
        F: Fn(Scalar<A>) -> Scalar<B>,
        A: Clone,
    {
        match self {
            Differentiable::Scalar(a) => Differentiable::Scalar(f(a.clone())),
            Differentiable::Vector(slice) => {
                Differentiable::Vector(slice.iter().map(|x| x.map(f)).collect())
            }
        }
    }
}

impl<A> Differentiable<A>
where
    A: Clone + Eq + Hash + AddAssign + Mul<Output = A> + Exp + Div<Output = A> + Zero + One,
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

    fn grad_once(self, wrt: Differentiable<A>) -> Differentiable<A> {
        let mut acc = HashMap::new();
        self.accumulate_gradients(&mut acc);

        wrt.map(&|d| match acc.get(&d) {
            None => Scalar::Number(A::zero()),
            Some(x) => Scalar::Number(x.clone()),
        })
    }

    pub fn grad<F>(f: F, theta: Differentiable<A>) -> Differentiable<A>
    where
        F: Fn(&Differentiable<A>) -> Differentiable<A>,
    {
        let wrt = theta.map(&Scalar::truncate_dual);
        let after_f = f(&wrt);
        Differentiable::grad_once(after_f, wrt)
    }
}

#[cfg(test)]
mod tests {
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
                Differentiable::Scalar(Scalar::Number(NotNan::new(3.0).expect("3 is not NaN"))),
                Differentiable::Scalar(Scalar::Number(NotNan::new(4.0).expect("4 is not NaN"))),
            ]
            .into(),
        );
        let mapped = v.map(&|x: Scalar<NotNan<f64>>| match x {
            Scalar::Number(i) => Scalar::Number(i + NotNan::new(1.0).expect("1 is not NaN")),
            Scalar::Dual(_, _) => panic!("Not hit"),
        });

        let v = match mapped {
            Differentiable::Scalar(_) => panic!("Not a scalar"),
            Differentiable::Vector(v) => v
                .as_ref()
                .iter()
                .map(|d| extract_scalar(d).clone())
                .collect::<Vec<_>>(),
        };

        assert_eq!(v, [4.0, 5.0]);
    }
}
