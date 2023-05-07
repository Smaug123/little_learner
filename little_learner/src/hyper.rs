use crate::loss::{NakedHypers, RmsHyper, VelocityHypers};
use crate::traits::{NumLike, Zero};
use rand::{rngs::StdRng, Rng};

/// Hyperparameters which apply to any possible optimisation algorithm that uses gradient descent.t
pub struct BaseGradientDescentHyper<A, R: Rng> {
    pub sampling: Option<(R, usize)>,
    pub iterations: u32,
    params: NakedHypers<A>,
}

impl<A> BaseGradientDescentHyper<A, StdRng>
where
    A: NumLike + NumLike,
{
    pub fn naked(learning_rate: A, iterations: u32) -> Self {
        BaseGradientDescentHyper {
            params: NakedHypers { learning_rate },
            iterations,
            sampling: None,
        }
    }

    pub fn with_rng<S: Rng>(self, rng: S, size: usize) -> BaseGradientDescentHyper<A, S> {
        BaseGradientDescentHyper {
            params: self.params,
            iterations: self.iterations,
            sampling: Some((rng, size)),
        }
    }

    pub fn with_iterations(self, n: u32) -> Self {
        BaseGradientDescentHyper {
            sampling: self.sampling,
            iterations: n,
            params: self.params,
        }
    }

    pub fn to_immutable(&self) -> NakedHypers<A> {
        self.params.clone()
    }
}

#[derive(Clone)]
pub struct VelocityGradientDescentHyper<A, R: Rng> {
    sampling: Option<(R, usize)>,
    learning_rate: A,
    iterations: u32,
    mu: A,
}

impl<A> VelocityGradientDescentHyper<A, StdRng>
where
    A: Zero,
{
    pub fn naked(learning_rate: A, iterations: u32) -> Self {
        VelocityGradientDescentHyper {
            sampling: None,
            learning_rate,
            iterations,
            mu: A::zero(),
        }
    }
}

impl<A, R: Rng> VelocityGradientDescentHyper<A, R> {
    pub fn with_mu(self, mu: A) -> Self {
        VelocityGradientDescentHyper {
            sampling: self.sampling,
            mu,
            learning_rate: self.learning_rate,
            iterations: self.iterations,
        }
    }

    pub fn to_immutable(&self) -> VelocityHypers<A>
    where
        A: Clone,
    {
        VelocityHypers {
            mu: self.mu.clone(),
            learning_rate: self.learning_rate.clone(),
        }
    }
}

impl<A, R: Rng> From<VelocityGradientDescentHyper<A, R>> for BaseGradientDescentHyper<A, R> {
    fn from(val: VelocityGradientDescentHyper<A, R>) -> BaseGradientDescentHyper<A, R> {
        BaseGradientDescentHyper {
            sampling: val.sampling,
            iterations: val.iterations,
            params: NakedHypers {
                learning_rate: val.learning_rate,
            },
        }
    }
}

#[derive(Clone)]
pub struct RmsGradientDescentHyper<A, R: Rng> {
    sampling: Option<(R, usize)>,
    iterations: u32,
    rms: RmsHyper<A>,
}

impl<A> RmsGradientDescentHyper<A, StdRng> {
    pub fn default(learning_rate: A, iterations: u32) -> Self
    where
        A: NumLike,
    {
        let two = A::one() + A::one();
        let ten = two.clone() * two.clone() * two.clone() + two;
        let one_tenth = A::one() / ten.clone();
        let one_hundredth = one_tenth.clone() * one_tenth;
        let one_ten_k = one_hundredth.clone() * one_hundredth;

        RmsGradientDescentHyper {
            sampling: None,
            iterations,
            rms: RmsHyper {
                stabilizer: one_ten_k.clone() * one_ten_k,
                beta: A::one() + -(A::one() / ten),
                learning_rate,
            },
        }
    }
}

impl<A, R: Rng> RmsGradientDescentHyper<A, R> {
    pub fn with_stabilizer(self, stabilizer: A) -> Self {
        RmsGradientDescentHyper {
            sampling: self.sampling,
            rms: RmsHyper {
                stabilizer,
                beta: self.rms.beta,
                learning_rate: self.rms.learning_rate,
            },
            iterations: self.iterations,
        }
    }

    pub fn with_beta(self, beta: A) -> Self {
        RmsGradientDescentHyper {
            sampling: self.sampling,
            rms: RmsHyper {
                stabilizer: self.rms.stabilizer,
                beta,
                learning_rate: self.rms.learning_rate,
            },
            iterations: self.iterations,
        }
    }

    pub fn to_immutable(&self) -> RmsHyper<A>
    where
        A: Clone,
    {
        self.rms.clone()
    }
}

impl<A, R: Rng> From<RmsGradientDescentHyper<A, R>> for BaseGradientDescentHyper<A, R> {
    fn from(val: RmsGradientDescentHyper<A, R>) -> BaseGradientDescentHyper<A, R> {
        BaseGradientDescentHyper {
            sampling: val.sampling,
            iterations: val.iterations,
            params: NakedHypers {
                learning_rate: val.rms.learning_rate,
            },
        }
    }
}
