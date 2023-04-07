#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod with_tensor;

use core::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, Neg};

use little_learner::auto_diff::{of_scalar, of_slice, to_scalar, Differentiable};

use little_learner::loss::{l2_loss_2, predict_quadratic};
use little_learner::scalar::Scalar;
use little_learner::traits::{Exp, One, Zero};
use ordered_float::NotNan;

fn iterate<A, F>(f: &F, start: A, n: u32) -> A
where
    F: Fn(A) -> A,
{
    let mut v = start;
    for _ in 0..n {
        v = f(v);
    }
    v
}

struct GradientDescentHyper<A, const RANK: usize> {
    learning_rate: A,
    iterations: u32,
}

fn gradient_descent_step<A, F, const RANK: usize>(
    f: &F,
    theta: Differentiable<A, RANK>,
    params: &GradientDescentHyper<A, RANK>,
) -> Differentiable<A, RANK>
where
    A: Clone
        + Mul<Output = A>
        + Neg<Output = A>
        + Add<Output = A>
        + Hash
        + AddAssign
        + Div<Output = A>
        + Zero
        + One
        + Eq
        + Exp,
    F: Fn(Differentiable<A, RANK>) -> Differentiable<A, RANK>,
{
    let delta = Differentiable::grad(f, &theta);
    Differentiable::map2(&theta, &delta, &|theta, delta| {
        (*theta).clone() - (Scalar::make((params.learning_rate).clone()) * (*delta).clone())
    })
}

fn main() {
    let xs = [-1.0, 0.0, 1.0, 2.0, 3.0];
    let ys = [2.55, 2.1, 4.35, 10.2, 18.25];

    let hyper = GradientDescentHyper {
        learning_rate: NotNan::new(0.001).expect("not nan"),
        iterations: 1000,
    };

    let iterated = {
        let xs = xs.map(|x| NotNan::new(x).expect("not nan"));
        let ys = ys.map(|x| NotNan::new(x).expect("not nan"));
        iterate(
            &|theta| {
                gradient_descent_step(
                    &|x| {
                        Differentiable::of_vector(vec![of_scalar(l2_loss_2(
                            predict_quadratic,
                            of_slice(&xs),
                            of_slice(&ys),
                            x,
                        ))])
                    },
                    theta,
                    &hyper,
                )
            },
            of_slice(&[
                NotNan::<f64>::zero(),
                NotNan::<f64>::zero(),
                NotNan::<f64>::zero(),
            ]),
            hyper.iterations,
        )
    };

    println!(
        "After iteration: {:?}",
        Differentiable::to_vector(iterated)
            .into_iter()
            .map(|x| to_scalar(x).real_part().into_inner())
            .collect::<Vec<_>>()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayvec::ArrayVec;
    use little_learner::{
        auto_diff::to_scalar,
        loss::{predict_line_2, square},
    };

    use crate::with_tensor::{l2_loss, predict_line};

    #[test]
    fn loss_example() {
        let xs = [2.0, 1.0, 4.0, 3.0];
        let ys = [1.8, 1.2, 4.2, 3.3];
        let loss = l2_loss_2(
            predict_line_2,
            of_slice(&xs),
            of_slice(&ys),
            of_slice(&[0.0, 0.0]),
        );

        assert_eq!(*loss.real_part(), 33.21);
    }

    #[test]
    fn l2_loss_non_autodiff_example() {
        let xs = [2.0, 1.0, 4.0, 3.0];
        let ys = [1.8, 1.2, 4.2, 3.3];
        let loss = l2_loss(predict_line, &xs, &ys, &[0.0099, 0.0]);
        assert_eq!(loss, 32.5892403);
    }

    #[test]
    fn grad_example() {
        let input_vec = of_slice(&[NotNan::new(27.0).expect("not nan")]);

        let grad: Vec<_> = Differentiable::to_vector(Differentiable::grad(
            |x| Differentiable::map(x, &mut |x| square(&x)),
            &input_vec,
        ))
        .into_iter()
        .map(|x| to_scalar(x).real_part().into_inner())
        .collect();
        assert_eq!(grad, [54.0]);
    }

    #[test]
    fn loss_gradient() {
        let input_vec = of_slice(&[NotNan::<f64>::zero(), NotNan::<f64>::zero()]);
        let xs = [2.0, 1.0, 4.0, 3.0].map(|x| NotNan::new(x).expect("not nan"));
        let ys = [1.8, 1.2, 4.2, 3.3].map(|x| NotNan::new(x).expect("not nan"));
        let grad = Differentiable::grad(
            |x| {
                Differentiable::of_vector(vec![of_scalar(l2_loss_2(
                    predict_line_2,
                    of_slice(&xs),
                    of_slice(&ys),
                    x,
                ))])
            },
            &input_vec,
        );

        assert_eq!(
            Differentiable::to_vector(grad)
                .into_iter()
                .map(|x| *(to_scalar(x).real_part()))
                .collect::<Vec<_>>(),
            [-63.0, -21.0]
        );
    }

    #[test]
    fn test_iterate() {
        let f = |t: [i32; 3]| {
            let mut vec = ArrayVec::<i32, 3>::new();
            for i in t {
                vec.push(i - 3);
            }
            vec.into_inner().unwrap()
        };
        assert_eq!(iterate(&f, [1, 2, 3], 5u32), [-14, -13, -12]);
    }

    #[test]
    fn first_optimisation_test() {
        let xs = [2.0, 1.0, 4.0, 3.0];
        let ys = [1.8, 1.2, 4.2, 3.3];

        let hyper = GradientDescentHyper {
            learning_rate: NotNan::new(0.01).expect("not nan"),
            iterations: 1000,
        };
        let iterated = {
            let xs = xs.map(|x| NotNan::new(x).expect("not nan"));
            let ys = ys.map(|x| NotNan::new(x).expect("not nan"));
            iterate(
                &|theta| {
                    gradient_descent_step(
                        &|x| {
                            Differentiable::of_vector(vec![of_scalar(l2_loss_2(
                                predict_line_2,
                                of_slice(&xs),
                                of_slice(&ys),
                                x,
                            ))])
                        },
                        theta,
                        &hyper,
                    )
                },
                of_slice(&[NotNan::<f64>::zero(), NotNan::<f64>::zero()]),
                hyper.iterations,
            )
        };
        let iterated = Differentiable::to_vector(iterated)
            .into_iter()
            .map(|x| to_scalar(x).real_part().into_inner())
            .collect::<Vec<_>>();

        assert_eq!(iterated, vec![1.0499993623489503, 0.0000018747718457656533]);
    }

    #[test]
    fn optimise_quadratic() {
        let xs = [-1.0, 0.0, 1.0, 2.0, 3.0];
        let ys = [2.55, 2.1, 4.35, 10.2, 18.25];

        let hyper = GradientDescentHyper {
            learning_rate: NotNan::new(0.001).expect("not nan"),
            iterations: 1000,
        };

        let iterated = {
            let xs = xs.map(|x| NotNan::new(x).expect("not nan"));
            let ys = ys.map(|x| NotNan::new(x).expect("not nan"));
            iterate(
                &|theta| {
                    gradient_descent_step(
                        &|x| {
                            Differentiable::of_vector(vec![of_scalar(l2_loss_2(
                                predict_quadratic,
                                of_slice(&xs),
                                of_slice(&ys),
                                x,
                            ))])
                        },
                        theta,
                        &hyper,
                    )
                },
                of_slice(&[
                    NotNan::<f64>::zero(),
                    NotNan::<f64>::zero(),
                    NotNan::<f64>::zero(),
                ]),
                hyper.iterations,
            )
        };
        let iterated = Differentiable::to_vector(iterated)
            .into_iter()
            .map(|x| to_scalar(x).real_part().into_inner())
            .collect::<Vec<_>>();

        println!("{:?}", iterated);

        assert_eq!(
            iterated,
            [2.0546423148479684, 0.9928606519360353, 1.4787394427094362]
        );
    }
}
