use crate::loss::{NakedHypers, RmsHyper, VelocityHypers};
use crate::traits::{NumLike, Zero};
use rand::rngs::StdRng;

/// Hyperparameters which apply to any possible optimisation algorithm that uses gradient descent.
pub struct BaseGradientDescentHyper<Rng> {
    pub sampling: Option<(Rng, usize)>,
    pub iterations: u32,
}

impl BaseGradientDescentHyper<StdRng> {
    pub fn new(iterations: u32) -> BaseGradientDescentHyper<StdRng> {
        BaseGradientDescentHyper {
            sampling: None,
            iterations,
        }
    }
}

impl<Rng> BaseGradientDescentHyper<Rng> {
    pub fn with_rng<Rng2>(self, rng: Rng2, size: usize) -> BaseGradientDescentHyper<Rng2> {
        BaseGradientDescentHyper {
            iterations: self.iterations,
            sampling: Some((rng, size)),
        }
    }

    pub fn with_iterations(self, n: u32) -> Self {
        BaseGradientDescentHyper {
            sampling: self.sampling,
            iterations: n,
        }
    }
}

pub struct NakedGradientDescentHyper<A, Rng> {
    base: BaseGradientDescentHyper<Rng>,
    naked: NakedHypers<A>,
}

impl<A> NakedGradientDescentHyper<A, StdRng>
where
    A: Zero,
{
    pub fn new(learning_rate: A, iterations: u32) -> Self {
        NakedGradientDescentHyper {
            base: BaseGradientDescentHyper::new(iterations),
            naked: NakedHypers { learning_rate },
        }
    }
}

impl<A, Rng> NakedGradientDescentHyper<A, Rng> {
    pub fn to_immutable(&self) -> NakedHypers<A>
    where
        A: Clone,
    {
        self.naked.clone()
    }

    pub fn with_rng<Rng2>(self, rng: Rng2, size: usize) -> NakedGradientDescentHyper<A, Rng2> {
        NakedGradientDescentHyper {
            base: self.base.with_rng(rng, size),
            naked: self.naked,
        }
    }
}

impl<A, Rng> From<NakedGradientDescentHyper<A, Rng>> for BaseGradientDescentHyper<Rng> {
    fn from(val: NakedGradientDescentHyper<A, Rng>) -> BaseGradientDescentHyper<Rng> {
        val.base
    }
}

pub struct VelocityGradientDescentHyper<A, Rng> {
    base: BaseGradientDescentHyper<Rng>,
    velocity: VelocityHypers<A>,
}

impl<A> VelocityGradientDescentHyper<A, StdRng>
where
    A: Zero,
{
    pub fn zero_momentum(learning_rate: A, iterations: u32) -> Self {
        VelocityGradientDescentHyper {
            base: BaseGradientDescentHyper::new(iterations),
            velocity: VelocityHypers {
                learning_rate,
                mu: A::zero(),
            },
        }
    }
}

impl<A, Rng> VelocityGradientDescentHyper<A, Rng> {
    pub fn with_mu(self, mu: A) -> Self {
        VelocityGradientDescentHyper {
            base: self.base,
            velocity: VelocityHypers {
                learning_rate: self.velocity.learning_rate,
                mu,
            },
        }
    }

    pub fn to_immutable(&self) -> VelocityHypers<A>
    where
        A: Clone,
    {
        self.velocity.clone()
    }
}

impl<A, Rng> From<VelocityGradientDescentHyper<A, Rng>> for BaseGradientDescentHyper<Rng> {
    fn from(val: VelocityGradientDescentHyper<A, Rng>) -> BaseGradientDescentHyper<Rng> {
        val.base
    }
}

pub struct RmsGradientDescentHyper<A, Rng> {
    base: BaseGradientDescentHyper<Rng>,
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
            base: BaseGradientDescentHyper::new(iterations),
            rms: RmsHyper {
                stabilizer: one_ten_k.clone() * one_ten_k,
                beta: A::one() + -(A::one() / ten),
                learning_rate,
            },
        }
    }
}

impl<A, Rng> RmsGradientDescentHyper<A, Rng> {
    pub fn with_stabilizer(self, stabilizer: A) -> Self {
        RmsGradientDescentHyper {
            base: self.base,
            rms: RmsHyper {
                stabilizer,
                beta: self.rms.beta,
                learning_rate: self.rms.learning_rate,
            },
        }
    }

    pub fn with_beta(self, beta: A) -> Self {
        RmsGradientDescentHyper {
            base: self.base,
            rms: RmsHyper {
                stabilizer: self.rms.stabilizer,
                beta,
                learning_rate: self.rms.learning_rate,
            },
        }
    }

    pub fn to_immutable(&self) -> RmsHyper<A>
    where
        A: Clone,
    {
        self.rms.clone()
    }
}

impl<A, Rng> From<RmsGradientDescentHyper<A, Rng>> for BaseGradientDescentHyper<Rng> {
    fn from(val: RmsGradientDescentHyper<A, Rng>) -> BaseGradientDescentHyper<Rng> {
        val.base
    }
}
