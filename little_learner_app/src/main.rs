#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use crate::rms_example::rms_example;
use little_learner::auto_diff::RankedDifferentiable;
use little_learner::block;
use ordered_float::NotNan;

mod iris;
mod rms_example;

fn main() {
    rms_example();

    let irises = iris::import::<f64, _>();
    let mut xs = Vec::with_capacity(irises.len());
    let mut ys = Vec::with_capacity(irises.len());
    for iris in irises {
        let (x, y) = iris.one_hot();
        xs.push(x);
        ys.push(y);
    }
    let _xs = RankedDifferentiable::of_vector(xs);
    let _ys = RankedDifferentiable::of_vector(ys);

    let _network = block::compose(block::dense::<NotNan<f64>, ()>(6, 3), block::dense(4, 6), 2);
}
