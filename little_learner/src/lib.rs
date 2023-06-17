#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(array_methods)]
#![feature(closure_lifetime_binder)]

pub mod auto_diff;
pub mod block;
pub mod decider;
pub mod ext;
pub mod gradient_descent;
pub mod hyper;
pub mod layer;
pub mod loss;
pub mod not_nan;
pub mod predictor;
pub mod sample;
pub mod scalar;
pub mod smooth;
pub mod traits;
