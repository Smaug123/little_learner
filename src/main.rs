use std::iter::Sum;
use std::ops::Mul;

type Point<A, const N: usize> = [A; N];

type Parameters<A, const N: usize, const M: usize> = [Point<A, N>; M];

fn dot_points<A: Mul, const N: usize>(x: &Point<A, N>, y: &Point<A, N>) -> A
where
    A: Sum<<A as Mul>::Output> + Copy,
{
    x.iter().zip(y).map(|(&x, &y)| x * y).sum()
}

fn dot<A, const N: usize, const M: usize>(x: &Point<A, N>, y: &Parameters<A, N, M>) -> Point<A, M>
where
    A: Mul<A> + Sum<<A as Mul>::Output> + Copy + Default,
{
    let mut result = [Default::default(); M];
    for (i, coord) in y.iter().map(|y| dot_points(x, y)).enumerate() {
        result[i] = coord;
    }
    result
}

fn line<A, const N: usize>(x: &Point<A, N>, theta: &Parameters<A, N, 1>) -> Point<A, 1>
where
    A: Mul<A> + Sum<<A as Mul>::Output> + Copy + Default,
{
    dot(x, theta)
}

//fn data_set() -> ([f64; 4], [f64; 4]) {
//    ([2.0, 1.0, 4.0, 3.0], [1.8, 1.2, 4.2, 3.3])
//}

fn linear_params_2d<A>(m: A, c: A) -> Parameters<A, 2, 1> {
    [[c, m]]
}

fn main() {
    let y = line(&[1.0, 7.3], &linear_params_2d(3.0, 1.0));
    println!("{:?}", y);
}
