#[macro_export]
macro_rules! tensor {
    ($x:ty , $i: expr) => {[$x; $i]};
    ($x:ty , $i: expr, $($is:expr),+) => {[tensor!($x, $($is),+); $i]};
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_tensor_type() {
        let _: tensor!(f64, 1, 2, 3) = [[[1.0, 3.0, 6.0], [-1.3, -30.0, -0.0]]];
    }
}

pub trait Extensible1<A> {
    fn apply<F>(&self, other: &A, op: &F) -> Self
    where
        F: Fn(&A, &A) -> A;
}

pub trait Extensible2<A> {
    fn apply<F>(&self, other: &Self, op: &F) -> Self
    where
        F: Fn(&A, &A) -> A;
}

impl<A, T, const N: usize> Extensible1<A> for [T; N]
where
    T: Extensible1<A> + Copy + Default,
{
    fn apply<F>(&self, other: &A, op: &F) -> Self
    where
        F: Fn(&A, &A) -> A,
    {
        let mut result = [Default::default(); N];
        for (i, coord) in self.iter().enumerate() {
            result[i] = T::apply(coord, other, op);
        }
        result
    }
}

impl<A, T, const N: usize> Extensible2<A> for [T; N]
where
    T: Extensible2<A> + Copy + Default,
{
    fn apply<F>(&self, other: &Self, op: &F) -> Self
    where
        F: Fn(&A, &A) -> A,
    {
        let mut result = [Default::default(); N];
        for (i, coord) in self.iter().enumerate() {
            result[i] = T::apply(coord, &other[i], op);
        }
        result
    }
}

#[macro_export]
macro_rules! extensible1 {
    ($x: ty) => {
        impl Extensible1<$x> for $x {
            fn apply<F>(&self, other: &$x, op: &F) -> Self
            where
                F: Fn(&Self, &Self) -> Self,
            {
                op(self, other)
            }
        }
    };
}

#[macro_export]
macro_rules! extensible2 {
    ($x: ty) => {
        impl Extensible2<$x> for $x {
            fn apply<F>(&self, other: &Self, op: &F) -> Self
            where
                F: Fn(&Self, &Self) -> Self,
            {
                op(self, other)
            }
        }
    };
}

extensible1!(u8);
extensible1!(f64);

extensible2!(u8);
extensible2!(f64);

pub fn extension1<T, A, F>(t1: &T, t2: &A, op: F) -> T
where
    T: Extensible1<A>,
    F: Fn(&A, &A) -> A,
{
    t1.apply::<F>(t2, &op)
}

pub fn extension2<T, A, F>(t1: &T, t2: &T, op: F) -> T
where
    T: Extensible2<A>,
    F: Fn(&A, &A) -> A,
{
    t1.apply::<F>(t2, &op)
}
