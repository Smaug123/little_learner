use std::iter::Sum;
use std::ops::{Mul, Sub};

use little_learner::tensor;
use little_learner::tensor::{extension2, Extensible2};
use little_learner::traits::One;

type Point<A, const N: usize> = [A; N];

type Parameters<A, const N: usize, const M: usize> = [Point<A, N>; M];

fn dot_points<A: Mul, const N: usize>(x: &Point<A, N>, y: &Point<A, N>) -> A
where
    A: Sum<<A as Mul>::Output> + Copy + Default + Mul<Output = A> + Extensible2<A>,
{
    extension2(x, y, |&x, &y| x * y).into_iter().sum()
}

fn dot<A, const N: usize, const M: usize>(x: &Point<A, N>, y: &Parameters<A, N, M>) -> Point<A, M>
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Copy + Default + Extensible2<A>,
{
    let mut result = [Default::default(); M];
    for (i, coord) in y.iter().map(|y| dot_points(x, y)).enumerate() {
        result[i] = coord;
    }
    result
}

fn sum<A, const N: usize>(x: &tensor!(A, N)) -> A
where
    A: Sum<A> + Copy,
{
    A::sum(x.iter().cloned())
}

fn squared<A, const N: usize>(x: &tensor!(A, N)) -> tensor!(A, N)
where
    A: Mul<Output = A> + Extensible2<A> + Copy + Default,
{
    extension2(x, x, |&a, &b| (a * b))
}

fn l2_norm<A, const N: usize>(prediction: &tensor!(A, N), data: &tensor!(A, N)) -> A
where
    A: Sum<A> + Mul<Output = A> + Extensible2<A> + Copy + Default + Sub<Output = A>,
{
    let diff = extension2(prediction, data, |&x, &y| x - y);
    sum(&squared(&diff))
}

pub fn l2_loss<A, F, Params, const N: usize>(
    target: F,
    data_xs: &tensor!(A, N),
    data_ys: &tensor!(A, N),
    params: &Params,
) -> A
where
    F: Fn(&tensor!(A, N), &Params) -> tensor!(A, N),
    A: Sum<A> + Mul<Output = A> + Extensible2<A> + Copy + Default + Sub<Output = A>,
{
    let pred_ys = target(data_xs, params);
    l2_norm(&pred_ys, data_ys)
}

pub fn predict_line<A, const N: usize>(xs: &tensor!(A, N), theta: &tensor!(A, 2)) -> tensor!(A, N)
where
    A: Mul<Output = A> + Sum<<A as Mul>::Output> + Copy + Default + Extensible2<A> + One,
{
    let mut result: tensor!(A, N) = [Default::default(); N];
    for (i, &x) in xs.iter().enumerate() {
        result[i] = dot(&[x, One::one()], &[*theta])[0];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use little_learner::tensor::extension1;

    #[test]
    fn test_extension() {
        let x: tensor!(u8, 1) = [2];
        assert_eq!(extension1(&x, &7, |x, y| x + y), [9]);
        let y: tensor!(u8, 1) = [7];
        assert_eq!(extension2(&x, &y, |x, y| x + y), [9]);

        let x: tensor!(u8, 3) = [5, 6, 7];
        assert_eq!(extension1(&x, &2, |x, y| x + y), [7, 8, 9]);
        let y: tensor!(u8, 3) = [2, 0, 1];
        assert_eq!(extension2(&x, &y, |x, y| x + y), [7, 6, 8]);

        let x: tensor!(u8, 2, 3) = [[4, 6, 7], [2, 0, 1]];
        assert_eq!(extension1(&x, &2, |x, y| x + y), [[6, 8, 9], [4, 2, 3]]);
        let y: tensor!(u8, 2, 3) = [[1, 2, 2], [6, 3, 1]];
        assert_eq!(extension2(&x, &y, |x, y| x + y), [[5, 8, 9], [8, 3, 2]]);
    }

    #[test]
    fn test_l2_norm() {
        assert_eq!(
            l2_norm(&[4.0, -3.0, 0.0, -4.0, 3.0], &[0.0, 0.0, 0.0, 0.0, 0.0]),
            50.0
        )
    }

    #[test]
    fn test_l2_loss() {
        let loss = l2_loss(
            predict_line,
            &[2.0, 1.0, 4.0, 3.0],
            &[1.8, 1.2, 4.2, 3.3],
            &[0.0, 0.0],
        );
        assert_eq!(loss, 33.21);

        let loss = l2_loss(
            predict_line,
            &[2.0, 1.0, 4.0, 3.0],
            &[1.8, 1.2, 4.2, 3.3],
            &[0.0099, 0.0],
        );
        assert_eq!((100.0 * loss).round() / 100.0, 32.59);
    }
}
