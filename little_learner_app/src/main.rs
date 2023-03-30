#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod with_tensor;

use little_learner::auto_diff::{of_scalar, of_slice, Differentiable};

use little_learner::loss::{l2_loss_2, predict_line_2, square};
use little_learner::traits::Zero;
use ordered_float::NotNan;

use crate::with_tensor::{l2_loss, predict_line};

#[allow(dead_code)]
fn l2_loss_non_autodiff_example() {
    let xs = [2.0, 1.0, 4.0, 3.0];
    let ys = [1.8, 1.2, 4.2, 3.3];
    let loss = l2_loss(predict_line, &xs, &ys, &[0.0099, 0.0]);
    println!("{:?}", loss);
}

fn main() {
    let input_vec = of_slice(&[NotNan::new(27.0).expect("not nan")]);

    let grad = Differentiable::grad(|x| Differentiable::map(x, &mut |x| square(&x)), input_vec);
    println!("Gradient of the x^2 function at x=27: {}", grad);

    let xs = [2.0, 1.0, 4.0, 3.0];
    let ys = [1.8, 1.2, 4.2, 3.3];

    let loss = l2_loss_2(
        predict_line_2,
        of_slice(&xs),
        of_slice(&ys),
        of_slice(&[0.0, 0.0]),
    );
    println!("Computation of L2 loss: {}", loss);

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
        input_vec,
    );

    println!("{}", grad);
}
