use std::marker::PhantomData;

pub struct ConstTeq<const A: usize, const B: usize> {
    phantom_a: PhantomData<[(); A]>,
    phantom_b: PhantomData<[(); B]>,
}

pub fn make<const A: usize>() -> ConstTeq<A, A> {
    ConstTeq {
        phantom_a: Default::default(),
        phantom_b: Default::default(),
    }
}
