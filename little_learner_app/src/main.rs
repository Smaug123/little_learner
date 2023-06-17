#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use crate::iris::{Iris, IrisType};
use crate::rms_example::rms_example;

mod iris;
mod rms_example;

fn main() {
    rms_example();

    let irises = iris::import();

    let expected = Iris {
        class: IrisType::Setosa,
        petal_length: 5.1,
        petal_width: 3.5,
        sepal_length: 1.4,
        sepal_width: 0.2,
    };
    assert_eq!(irises[0], expected);
}
