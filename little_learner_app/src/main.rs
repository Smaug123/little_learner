#![allow(incomplete_features)]
#![feature(generic_const_exprs)]


use crate::rms_example::rms_example;

mod iris;
mod rms_example;

fn main() {
    rms_example();

    let irises = iris::import();
    assert_eq!(irises[0], crate::iris::EXPECTED_FIRST);
}
