#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum Scalar<A> {
    Number(A),
    // The value, and the link.
    Dual(A, Option<A>),
}

#[allow(dead_code)]
impl<A> Scalar<A> {
    fn real_part(&self) -> &A {
        match self {
            Scalar::Number(a) => a,
            Scalar::Dual(a, _) => a,
        }
    }

    fn link(self) -> Option<A> {
        match self {
            Scalar::Dual(_, link) => link,
            Scalar::Number(_) => None,
        }
    }

    fn truncate_dual(self) -> Scalar<A>
    where
        A: Clone,
    {
        Scalar::Dual(self.real_part().clone(), None)
    }
}

#[allow(dead_code)]
pub enum Differentiable<A> {
    Scalar(Scalar<A>),
    Vector(Box<[Differentiable<A>]>),
}

#[allow(dead_code)]
impl<A> Differentiable<A> {
    fn map<B, F>(&self, f: &F) -> Differentiable<B>
    where
        F: Fn(Scalar<A>) -> Scalar<B>,
        A: Clone,
    {
        match self {
            Differentiable::Scalar(a) => Differentiable::Scalar(f((*a).clone())),
            Differentiable::Vector(slice) => {
                Differentiable::Vector(slice.iter().map(|x| x.map(f)).collect())
            }
        }
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
                Differentiable::Scalar(Scalar::Number(3)),
                Differentiable::Scalar(Scalar::Number(4)),
            ]
            .into(),
        );
        let mapped = v.map(&|x| match x {
            Scalar::Number(i) => Scalar::Number(i + 1),
            Scalar::Dual(_, _) => panic!("Not hit"),
        });

        let v = match mapped {
            Differentiable::Scalar(_) => panic!("Not a scalar"),
            Differentiable::Vector(v) => v
                .as_ref()
                .iter()
                .map(|d| extract_scalar(d).clone())
                .collect::<Vec<i32>>(),
        };

        assert_eq!(v, [4, 5]);
    }
}
