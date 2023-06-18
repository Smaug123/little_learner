use crate::predictor::{AdamHyper, NakedHypers, RmsHyper, VelocityHypers};
use crate::traits::{NumLike, Zero};
use rand::rngs::StdRng;

/// Hyperparameters which apply to any possible optimisation algorithm that uses gradient descent.
pub struct BaseGradientDescent<Rng> {
    pub sampling: Option<(Rng, usize)>,
    pub iterations: u32,
}

impl BaseGradientDescent<StdRng> {
    #[must_use]
    pub fn new(iterations: u32) -> BaseGradientDescent<StdRng> {
        BaseGradientDescent {
            sampling: None,
            iterations,
        }
    }
}

impl<Rng> BaseGradientDescent<Rng> {
    #[must_use]
    pub fn with_rng<Rng2>(self, rng: Rng2, size: usize) -> BaseGradientDescent<Rng2> {
        BaseGradientDescent {
            iterations: self.iterations,
            sampling: Some((rng, size)),
        }
    }

    #[must_use]
    pub fn with_iterations(self, n: u32) -> Self {
        BaseGradientDescent {
            sampling: self.sampling,
            iterations: n,
        }
    }
}

pub struct NakedGradientDescent<A, Rng> {
    base: BaseGradientDescent<Rng>,
    naked: NakedHypers<A>,
}

impl<A> NakedGradientDescent<A, StdRng>
where
    A: Zero,
{
    #[must_use]
    pub fn new(learning_rate: A, iterations: u32) -> Self {
        NakedGradientDescent {
            base: BaseGradientDescent::new(iterations),
            naked: NakedHypers { learning_rate },
        }
    }
}

impl<A, Rng> NakedGradientDescent<A, Rng> {
    pub fn to_immutable(&self) -> NakedHypers<A>
    where
        A: Clone,
    {
        self.naked.clone()
    }

    #[must_use]
    pub fn with_rng<Rng2>(self, rng: Rng2, size: usize) -> NakedGradientDescent<A, Rng2> {
        NakedGradientDescent {
            base: self.base.with_rng(rng, size),
            naked: self.naked,
        }
    }
}

impl<A, Rng> From<NakedGradientDescent<A, Rng>> for BaseGradientDescent<Rng> {
    fn from(val: NakedGradientDescent<A, Rng>) -> BaseGradientDescent<Rng> {
        val.base
    }
}

pub struct VelocityGradientDescent<A, Rng> {
    base: BaseGradientDescent<Rng>,
    velocity: VelocityHypers<A>,
}

impl<A> VelocityGradientDescent<A, StdRng>
where
    A: Zero,
{
    #[must_use]
    pub fn zero_momentum(learning_rate: A, iterations: u32) -> Self {
        VelocityGradientDescent {
            base: BaseGradientDescent::new(iterations),
            velocity: VelocityHypers {
                learning_rate,
                mu: A::zero(),
            },
        }
    }
}

impl<A, Rng> VelocityGradientDescent<A, Rng> {
    #[must_use]
    pub fn with_mu(self, mu: A) -> Self {
        VelocityGradientDescent {
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

impl<A, Rng> From<VelocityGradientDescent<A, Rng>> for BaseGradientDescent<Rng> {
    fn from(val: VelocityGradientDescent<A, Rng>) -> BaseGradientDescent<Rng> {
        val.base
    }
}

fn ten<A>() -> A
where
    A: NumLike,
{
    let two = A::one() + A::one();
    two.clone() * two.clone() * two.clone() + two
}

fn one_ten_k<A>() -> A
where
    A: NumLike,
{
    let one_tenth = A::one() / ten();
    let one_hundredth = one_tenth.clone() * one_tenth;
    one_hundredth.clone() * one_hundredth
}

pub struct RmsGradientDescent<A, Rng> {
    base: BaseGradientDescent<Rng>,
    rms: RmsHyper<A>,
}

impl<A> RmsGradientDescent<A, StdRng> {
    pub fn default(learning_rate: A, iterations: u32) -> Self
    where
        A: NumLike,
    {
        RmsGradientDescent {
            base: BaseGradientDescent::new(iterations),
            rms: RmsHyper {
                stabilizer: one_ten_k::<A>() * one_ten_k(),
                beta: A::one() + -(A::one() / ten()),
                learning_rate,
            },
        }
    }
}

impl<A, Rng> RmsGradientDescent<A, Rng> {
    #[must_use]
    pub fn with_stabilizer(self, stabilizer: A) -> Self {
        RmsGradientDescent {
            base: self.base,
            rms: RmsHyper {
                stabilizer,
                beta: self.rms.beta,
                learning_rate: self.rms.learning_rate,
            },
        }
    }

    #[must_use]
    pub fn with_beta(self, beta: A) -> Self {
        RmsGradientDescent {
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

impl<A, Rng> From<RmsGradientDescent<A, Rng>> for BaseGradientDescent<Rng> {
    fn from(val: RmsGradientDescent<A, Rng>) -> BaseGradientDescent<Rng> {
        val.base
    }
}

pub struct AdamGradientDescent<A, Rng> {
    base: BaseGradientDescent<Rng>,
    adam: AdamHyper<A>,
}

impl<A> AdamGradientDescent<A, StdRng> {
    pub fn default(learning_rate: A, iterations: u32) -> Self
    where
        A: NumLike,
    {
        AdamGradientDescent {
            base: BaseGradientDescent::new(iterations),
            adam: AdamHyper {
                mu: A::zero(),
                rms: RmsHyper {
                    learning_rate,
                    stabilizer: one_ten_k::<A>() * one_ten_k(),
                    beta: A::one() + -(A::one() / ten()),
                },
            },
        }
    }
}

impl<A, Rng> AdamGradientDescent<A, Rng> {
    #[must_use]
    pub fn with_stabilizer(self, stabilizer: A) -> Self {
        AdamGradientDescent {
            base: self.base,
            adam: self.adam.with_stabilizer(stabilizer),
        }
    }

    #[must_use]
    pub fn with_beta(self, beta: A) -> Self {
        AdamGradientDescent {
            base: self.base,
            adam: self.adam.with_beta(beta),
        }
    }

    #[must_use]
    pub fn with_mu(self, mu: A) -> Self {
        AdamGradientDescent {
            base: self.base,
            adam: self.adam.with_mu(mu),
        }
    }

    pub fn to_immutable(&self) -> AdamHyper<A>
    where
        A: Clone,
    {
        self.adam.clone()
    }
}

impl<A, Rng> From<AdamGradientDescent<A, Rng>> for BaseGradientDescent<Rng> {
    fn from(val: AdamGradientDescent<A, Rng>) -> BaseGradientDescent<Rng> {
        val.base
    }
}

pub struct HyperAndFreeze<Hyper, ImmutableHyper, H>
where
    H: FnOnce(&Hyper) -> ImmutableHyper,
{
    pub to_immutable: H,
    pub hyper: Hyper,
}
