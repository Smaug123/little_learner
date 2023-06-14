use crate::scalar::Scalar;
use crate::traits::{Exp, One, Sqrt, Zero};
use core::hash::Hash;
use std::collections::HashMap;
use std::ops::Add;
use std::{
    fmt::{Display, Write},
    ops::{AddAssign, Div, Mul, Neg},
};

impl<A, Tag> Zero for DifferentiableTagged<A, Tag>
where
    A: Zero,
    Tag: Zero,
{
    fn zero() -> DifferentiableTagged<A, Tag> {
        DifferentiableTagged {
            contents: DifferentiableContents::Scalar(Scalar::Number(A::zero(), None), Tag::zero()),
        }
    }
}

impl<A> One for Scalar<A>
where
    A: One,
{
    fn one() -> Scalar<A> {
        Scalar::Number(A::one(), None)
    }
}

impl<A, Tag> One for DifferentiableTagged<A, Tag>
where
    A: One,
    Tag: Zero,
{
    fn one() -> DifferentiableTagged<A, Tag> {
        DifferentiableTagged {
            contents: DifferentiableContents::Scalar(Scalar::one(), Tag::zero()),
        }
    }
}

impl<A, Tag> Clone for DifferentiableContents<A, Tag>
where
    A: Clone,
    Tag: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Scalar(arg0, tag) => Self::Scalar(arg0.clone(), tag.clone()),
            Self::Vector(arg0, rank) => Self::Vector(arg0.clone(), *rank),
        }
    }
}

impl<A, Tag> Clone for DifferentiableTagged<A, Tag>
where
    A: Clone,
    Tag: Clone,
{
    fn clone(&self) -> Self {
        DifferentiableTagged {
            contents: self.contents.clone(),
        }
    }
}

#[derive(Debug)]
enum DifferentiableContents<A, Tag> {
    Scalar(Scalar<A>, Tag),
    // Contains the rank of this differentiable (i.e. one more than the rank of the inputs).
    Vector(Vec<DifferentiableTagged<A, Tag>>, usize),
}

#[derive(Debug)]
pub struct DifferentiableTagged<A, Tag> {
    contents: DifferentiableContents<A, Tag>,
}

impl<A, Tag> Display for DifferentiableContents<A, Tag>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DifferentiableContents::Scalar(s, _) => f.write_fmt(format_args!("{s}")),
            DifferentiableContents::Vector(v, _rank) => {
                f.write_char('[')?;
                for v in v.iter() {
                    f.write_fmt(format_args!("{v}"))?;
                    f.write_char(',')?;
                }
                f.write_char(']')
            }
        }
    }
}

impl<A, Tag> Display for DifferentiableTagged<A, Tag>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.contents))
    }
}

pub type Differentiable<A> = DifferentiableTagged<A, ()>;
pub type RankedDifferentiable<A, const RANK: usize> = RankedDifferentiableTagged<A, (), RANK>;

impl<A, Tag> DifferentiableContents<A, Tag> {
    fn map<B, F>(&self, f: &mut F) -> DifferentiableContents<B, Tag>
    where
        F: FnMut(Scalar<A>) -> Scalar<B>,
        A: Clone,
        Tag: Clone,
    {
        match self {
            DifferentiableContents::Scalar(a, tag) => {
                DifferentiableContents::Scalar(f(a.clone()), (*tag).clone())
            }
            DifferentiableContents::Vector(slice, rank) => {
                DifferentiableContents::Vector(slice.iter().map(|x| x.map(f)).collect(), *rank)
            }
        }
    }

    fn map_with_tag<B, F>(&self, f: &mut F) -> DifferentiableContents<B, Tag>
    where
        F: FnMut(Scalar<A>, &Tag) -> Scalar<B>,
        A: Clone,
        Tag: Clone,
    {
        match self {
            DifferentiableContents::Scalar(a, tag) => {
                DifferentiableContents::Scalar(f(a.clone(), tag), (*tag).clone())
            }
            DifferentiableContents::Vector(slice, rank) => DifferentiableContents::Vector(
                slice.iter().map(|x| x.map_with_tag(f)).collect(),
                *rank,
            ),
        }
    }

    fn map_tag<Tag2, F>(&self, f: &mut F) -> DifferentiableContents<A, Tag2>
    where
        F: FnMut(&Tag) -> Tag2,
        A: Clone,
    {
        match self {
            DifferentiableContents::Scalar(a, tag) => {
                DifferentiableContents::Scalar((*a).clone(), f(tag))
            }
            DifferentiableContents::Vector(slice, rank) => {
                DifferentiableContents::Vector(slice.iter().map(|x| x.map_tag(f)).collect(), *rank)
            }
        }
    }

    /// This function does *not* check that its inputs are of exactly the same shape, though it
    /// does check ranks. If you have two vectors of different lengths, you will silently get the
    /// shorter one.
    ///
    /// # Panics
    /// Panics if the two inputs have different shapes (e.g. if they have different ranks).
    fn map2<B, C, Tag2, Tag3, F>(
        &self,
        other: &DifferentiableContents<B, Tag2>,
        f: &mut F,
    ) -> DifferentiableContents<C, Tag3>
    where
        F: FnMut(&Scalar<A>, Tag, &Scalar<B>, Tag2) -> (Scalar<C>, Tag3),
        A: Clone,
        B: Clone,
        Tag: Clone,
        Tag2: Clone,
    {
        match (self, other) {
            (DifferentiableContents::Scalar(a, tag1), DifferentiableContents::Scalar(b, tag2)) => {
                let (scalar, tag) = f(a, tag1.clone(), b, tag2.clone());
                DifferentiableContents::Scalar(scalar, tag)
            }
            (
                DifferentiableContents::Vector(slice_a, rank_a),
                DifferentiableContents::Vector(slice_b, rank_b),
            ) => {
                assert_eq!(rank_a, rank_b, "Unexpectedly different ranks in map2");
                DifferentiableContents::Vector(
                    slice_a
                        .iter()
                        .zip(slice_b.iter())
                        .map(|(a, b)| a.map2_tagged(b, f))
                        .collect(),
                    *rank_a,
                )
            }
            _ => panic!("Wrong shapes!"),
        }
    }

    /// Unwraps one layer of each input, so the passed function takes inputs which have decreased
    /// the ranks of the `map2_once_tagged` input by one.
    /// Panics if passed a scalar or if the input vectors are not the same length.
    pub fn map2_once_tagged<B, C, Tag2, Tag3, F>(
        self: &DifferentiableContents<A, Tag>,
        other: &DifferentiableContents<B, Tag2>,
        mut f: F,
    ) -> DifferentiableContents<C, Tag3>
    where
        F: FnMut(
            &DifferentiableTagged<A, Tag>,
            &DifferentiableTagged<B, Tag2>,
        ) -> DifferentiableTagged<C, Tag3>,
    {
        match (self, other) {
            (DifferentiableContents::Scalar(_, _), _) => {
                panic!("First arg needed to have non-scalar rank")
            }
            (_, DifferentiableContents::Scalar(_, _)) => {
                panic!("Second arg needed to have non-scalar rank")
            }
            (
                DifferentiableContents::Vector(v1, rank1),
                DifferentiableContents::Vector(v2, _rank2),
            ) => {
                assert_eq!(
                    v1.len(),
                    v2.len(),
                    "Must map two vectors of the same length, got {rank1} and {_rank2}"
                );
                assert_ne!(
                    v1.len(),
                    0,
                    "Cannot determine a rank of a zero-length vector"
                );
                let mut rank = 0usize;
                DifferentiableContents::Vector(
                    v1.iter()
                        .zip(v2.iter())
                        .map(|(a, b)| {
                            let result = f(a, b);
                            match result.contents {
                                DifferentiableContents::Vector(_, discovered_rank) => {
                                    rank = discovered_rank + 1;
                                }
                                DifferentiableContents::Scalar(_, _) => {
                                    rank = 1;
                                }
                            }
                            result
                        })
                        .collect(),
                    rank,
                )
            }
        }
    }

    fn of_slice<'a, T, I>(tag: Tag, input: I) -> DifferentiableContents<T, Tag>
    where
        T: Clone + 'a,
        Tag: Clone,
        I: IntoIterator<Item = &'a T>,
    {
        DifferentiableContents::Vector(
            input
                .into_iter()
                .map(|v| DifferentiableTagged {
                    contents: DifferentiableContents::Scalar(
                        Scalar::Number(v.clone(), None),
                        tag.clone(),
                    ),
                })
                .collect(),
            1,
        )
    }

    fn rank(&self) -> usize {
        match self {
            DifferentiableContents::Scalar(_, _) => 0,
            DifferentiableContents::Vector(_, rank) => *rank,
        }
    }
}

impl<A, Tag> DifferentiableTagged<A, Tag> {
    pub fn map<B, F>(&self, f: &mut F) -> DifferentiableTagged<B, Tag>
    where
        A: Clone,
        Tag: Clone,
        F: FnMut(Scalar<A>) -> Scalar<B>,
    {
        DifferentiableTagged {
            contents: self.contents.map(f),
        }
    }

    pub fn map_with_tag<B, F>(&self, f: &mut F) -> DifferentiableTagged<B, Tag>
    where
        A: Clone,
        Tag: Clone,
        F: FnMut(Scalar<A>, &Tag) -> Scalar<B>,
    {
        DifferentiableTagged {
            contents: self.contents.map_with_tag(f),
        }
    }

    pub fn map_tag<Tag2, F>(&self, f: &mut F) -> DifferentiableTagged<A, Tag2>
    where
        F: FnMut(&Tag) -> Tag2,
        A: Clone,
    {
        DifferentiableTagged {
            contents: self.contents.map_tag(f),
        }
    }

    pub fn map2_tagged<B, C, Tag2, Tag3, F>(
        &self,
        other: &DifferentiableTagged<B, Tag2>,
        f: &mut F,
    ) -> DifferentiableTagged<C, Tag3>
    where
        F: FnMut(&Scalar<A>, Tag, &Scalar<B>, Tag2) -> (Scalar<C>, Tag3),
        A: Clone,
        B: Clone,
        Tag2: Clone,
        Tag: Clone,
    {
        DifferentiableTagged {
            contents: self.contents.map2(&other.contents, f),
        }
    }

    pub fn map2_once_tagged<B, C, Tag2, Tag3, F>(
        self: &DifferentiableTagged<A, Tag>,
        other: &DifferentiableTagged<B, Tag2>,
        f: F,
    ) -> DifferentiableTagged<C, Tag3>
    where
        F: FnMut(
            &DifferentiableTagged<A, Tag>,
            &DifferentiableTagged<B, Tag2>,
        ) -> DifferentiableTagged<C, Tag3>,
    {
        DifferentiableTagged {
            contents: self.contents.map2_once_tagged(&other.contents, f),
        }
    }

    pub fn attach_rank<const RANK: usize>(
        self: DifferentiableTagged<A, Tag>,
    ) -> Option<RankedDifferentiableTagged<A, Tag, RANK>> {
        if self.contents.rank() == RANK {
            Some(RankedDifferentiableTagged { contents: self })
        } else {
            None
        }
    }

    pub fn of_scalar_tagged(s: Scalar<A>, tag: Tag) -> DifferentiableTagged<A, Tag> {
        DifferentiableTagged {
            contents: DifferentiableContents::Scalar(s, tag),
        }
    }
}

impl<A> Differentiable<A> {
    pub fn map2<B, C, F>(&self, other: &Differentiable<B>, f: &mut F) -> Differentiable<C>
    where
        F: FnMut(&Scalar<A>, &Scalar<B>) -> Scalar<C>,
        A: Clone,
        B: Clone,
    {
        DifferentiableTagged::map2_tagged(self, other, &mut |a, (), b, ()| (f(a, b), ()))
    }
}

impl<A> Differentiable<A> {
    pub fn of_scalar(s: Scalar<A>) -> Differentiable<A> {
        DifferentiableTagged::of_scalar_tagged(s, ())
    }
}

impl<A, Tag> DifferentiableContents<A, Tag> {
    fn into_scalar(self) -> Scalar<A> {
        match self {
            DifferentiableContents::Scalar(s, _) => s,
            DifferentiableContents::Vector(_, _) => panic!("not a scalar"),
        }
    }

    fn into_vector(self) -> Vec<DifferentiableTagged<A, Tag>> {
        match self {
            DifferentiableContents::Scalar(_, _) => panic!("not a vector"),
            DifferentiableContents::Vector(v, _) => v,
        }
    }

    fn borrow_scalar(&self) -> &Scalar<A> {
        match self {
            DifferentiableContents::Scalar(s, _) => s,
            DifferentiableContents::Vector(_, _) => panic!("not a scalar"),
        }
    }

    fn borrow_vector(&self) -> &Vec<DifferentiableTagged<A, Tag>> {
        match self {
            DifferentiableContents::Scalar(_, _) => panic!("not a vector"),
            DifferentiableContents::Vector(v, _) => v,
        }
    }
}

impl<A, Tag> DifferentiableTagged<A, Tag> {
    pub fn into_scalar(self) -> Scalar<A> {
        self.contents.into_scalar()
    }

    pub fn into_vector(self) -> Vec<DifferentiableTagged<A, Tag>> {
        self.contents.into_vector()
    }

    pub fn borrow_scalar(&self) -> &Scalar<A> {
        self.contents.borrow_scalar()
    }

    pub fn borrow_vector(&self) -> &Vec<DifferentiableTagged<A, Tag>> {
        self.contents.borrow_vector()
    }

    fn of_slice<'a, T>(input: T, tag: Tag) -> DifferentiableTagged<A, Tag>
    where
        A: Clone + 'a,
        Tag: Clone,
        T: IntoIterator<Item = &'a A>,
    {
        DifferentiableTagged {
            contents: DifferentiableContents::<A, Tag>::of_slice(tag, input),
        }
    }

    /// # Panics
    /// Panics if the input is empty (otherwise we can't determine a rank).
    #[must_use]
    pub fn of_vec(input: Vec<DifferentiableTagged<A, Tag>>) -> DifferentiableTagged<A, Tag> {
        assert!(!input.is_empty(), "Can't make an empty tensor");
        let rank = input[0].rank();
        DifferentiableTagged {
            contents: DifferentiableContents::Vector(input, 1 + rank),
        }
    }

    pub fn rank(&self) -> usize {
        self.contents.rank()
    }
}

impl<A, Tag> DifferentiableContents<A, Tag>
where
    A: Clone
        + Eq
        + Hash
        + AddAssign
        + Add<Output = A>
        + Mul<Output = A>
        + Exp
        + Div<Output = A>
        + Zero
        + One
        + Sqrt
        + Neg<Output = A>,
{
    fn accumulate_gradients_vec(
        v: &[DifferentiableTagged<A, Tag>],
        acc: &mut HashMap<Scalar<A>, A>,
    ) {
        for v in v.iter().rev() {
            v.contents.accumulate_gradients(acc);
        }
    }

    fn accumulate_gradients(&self, acc: &mut HashMap<Scalar<A>, A>) {
        match self {
            DifferentiableContents::Scalar(y, _) => {
                let k = y.clone_link();
                k.invoke(y, A::one(), acc);
            }
            DifferentiableContents::Vector(y, _rank) => {
                DifferentiableContents::accumulate_gradients_vec(y, acc);
            }
        }
    }

    fn grad_once<const PARAM_NUM: usize>(
        self,
        wrt: [DifferentiableTagged<A, Tag>; PARAM_NUM],
    ) -> [DifferentiableTagged<A, Tag>; PARAM_NUM]
    where
        Tag: Clone,
    {
        let mut acc = HashMap::new();
        self.accumulate_gradients(&mut acc);

        wrt.map(|wrt| {
            wrt.map(&mut |d| match acc.get(&d) {
                None => Scalar::Number(A::zero(), None),
                Some(x) => Scalar::Number(x.clone(), None),
            })
        })
    }
}

#[derive(Clone, Debug)]
pub struct RankedDifferentiableTagged<A, Tag, const RANK: usize> {
    contents: DifferentiableTagged<A, Tag>,
}

impl<A, Tag, const RANK: usize> Display for RankedDifferentiableTagged<A, Tag, RANK>
where
    A: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.contents, f)
    }
}

impl<A, Tag> RankedDifferentiableTagged<A, Tag, 0> {
    pub fn to_scalar(self) -> Scalar<A> {
        self.contents.contents.into_scalar()
    }

    pub fn of_scalar_tagged(s: Scalar<A>, tag: Tag) -> RankedDifferentiableTagged<A, Tag, 0> {
        RankedDifferentiableTagged {
            contents: DifferentiableTagged::of_scalar_tagged(s, tag),
        }
    }
}

impl<A> RankedDifferentiable<A, 0> {
    pub fn of_scalar(s: Scalar<A>) -> Self {
        RankedDifferentiableTagged::of_scalar_tagged(s, ())
    }
}

impl<A, Tag> RankedDifferentiableTagged<A, Tag, 1> {
    pub fn of_slice_tagged<'a, T>(input: T, tag: Tag) -> RankedDifferentiableTagged<A, Tag, 1>
    where
        A: Clone + 'a,
        Tag: Clone,
        T: IntoIterator<Item = &'a A>,
    {
        RankedDifferentiableTagged {
            contents: DifferentiableTagged {
                contents: DifferentiableContents::<A, Tag>::of_slice(tag, input),
            },
        }
    }
}

impl<A> RankedDifferentiable<A, 1> {
    pub fn of_slice<'a, T>(input: T) -> RankedDifferentiable<A, 1>
    where
        A: Clone + 'a,
        T: IntoIterator<Item = &'a A>,
    {
        RankedDifferentiableTagged::of_slice_tagged(input, ())
    }

    pub fn collect(self: RankedDifferentiable<A, 1>) -> Vec<A>
    where
        A: Copy,
    {
        self.to_vector()
            .into_iter()
            .map(|x| *x.to_scalar().real_part())
            .collect::<Vec<_>>()
    }
}

impl<A, Tag> RankedDifferentiableTagged<A, Tag, 2> {
    pub fn of_slice_2_tagged<T, const N: usize>(
        input: &[T],
        tag: Tag,
    ) -> RankedDifferentiableTagged<A, Tag, 2>
    where
        A: Clone,
        T: AsRef<[A]>,
        Tag: Clone,
    {
        let v = input
            .iter()
            .map(|x| DifferentiableTagged::of_slice(x.as_ref(), tag.clone()))
            .collect::<Vec<_>>();
        RankedDifferentiableTagged {
            contents: DifferentiableTagged::of_vec(v),
        }
    }
}

impl<A> RankedDifferentiable<A, 2> {
    pub fn of_slice_2<T, const N: usize>(input: &[T]) -> RankedDifferentiable<A, 2>
    where
        A: Clone,
        T: AsRef<[A]>,
    {
        RankedDifferentiableTagged::of_slice_2_tagged::<_, N>(input, ())
    }
}

impl<A, Tag, const RANK: usize> RankedDifferentiableTagged<A, Tag, RANK> {
    pub fn to_unranked(self) -> DifferentiableTagged<A, Tag> {
        self.contents
    }

    pub fn to_unranked_borrow(&self) -> &DifferentiableTagged<A, Tag> {
        &self.contents
    }

    #[must_use]
    pub fn of_vector(
        s: Vec<RankedDifferentiableTagged<A, Tag, RANK>>,
    ) -> RankedDifferentiableTagged<A, Tag, { RANK + 1 }> {
        RankedDifferentiableTagged {
            contents: DifferentiableTagged::of_vec(s.into_iter().map(|v| v.contents).collect()),
        }
    }

    pub fn map_tagged<B, F>(
        self: RankedDifferentiableTagged<A, Tag, RANK>,
        f: &mut F,
    ) -> RankedDifferentiableTagged<B, Tag, RANK>
    where
        F: FnMut(Scalar<A>) -> Scalar<B>,
        A: Clone,
        Tag: Clone,
    {
        RankedDifferentiableTagged {
            contents: DifferentiableTagged::map(&self.contents, f),
        }
    }

    pub fn map_tag<Tag2, F>(
        self: &RankedDifferentiableTagged<A, Tag, RANK>,
        f: &mut F,
    ) -> RankedDifferentiableTagged<A, Tag2, RANK>
    where
        A: Clone,
        F: FnMut(&Tag) -> Tag2,
    {
        RankedDifferentiableTagged {
            contents: DifferentiableTagged::map_tag(&self.contents, f),
        }
    }

    pub fn map2_tagged<'a, 'b, B, C, Tag2, Tag3, F>(
        self: &'a RankedDifferentiableTagged<A, Tag, RANK>,
        other: &'a RankedDifferentiableTagged<B, Tag2, RANK>,
        f: &'b mut F,
    ) -> RankedDifferentiableTagged<C, Tag3, RANK>
    where
        F: FnMut(&Scalar<A>, Tag, &Scalar<B>, Tag2) -> (Scalar<C>, Tag3),
        A: Clone,
        B: Clone,
        Tag: Clone,
        Tag2: Clone,
    {
        RankedDifferentiableTagged {
            contents: DifferentiableTagged::map2_tagged(&self.contents, &other.contents, f),
        }
    }
    pub fn map2_once_tagged<
        'a,
        'c,
        B,
        C: 'a,
        Tag2,
        Tag3: 'a,
        F,
        const RANK_B: usize,
        const RANK_OUT: usize,
    >(
        self: &'a RankedDifferentiableTagged<A, Tag, RANK>,
        other: &'a RankedDifferentiableTagged<B, Tag2, RANK_B>,
        f: &'c mut F,
    ) -> RankedDifferentiableTagged<C, Tag3, RANK_OUT>
    where
        F: FnMut(
            &RankedDifferentiableTagged<A, Tag, { RANK - 1 }>,
            &RankedDifferentiableTagged<B, Tag2, { RANK_B - 1 }>,
        ) -> RankedDifferentiableTagged<C, Tag3, { RANK_OUT - 1 }>,
        A: Clone,
        B: Clone,
        Tag: Clone,
        Tag2: Clone,
        'c: 'a,
    {
        RankedDifferentiableTagged {
            contents: DifferentiableTagged::map2_once_tagged(
                &self.contents,
                &other.contents,
                &mut |a: &DifferentiableTagged<A, Tag>, b: &DifferentiableTagged<B, Tag2>| {
                    let a = (*a).clone().attach_rank::<{ RANK - 1 }>().unwrap();
                    let b = (*b).clone().attach_rank::<{ RANK_B - 1 }>().unwrap();
                    f(&a, &b).to_unranked()
                },
            ),
        }
    }

    pub fn to_vector(
        self: RankedDifferentiableTagged<A, Tag, RANK>,
    ) -> Vec<RankedDifferentiableTagged<A, Tag, { RANK - 1 }>> {
        self.contents
            .into_vector()
            .into_iter()
            .map(|v| RankedDifferentiableTagged { contents: v })
            .collect()
    }
}

impl<A, const RANK: usize> RankedDifferentiable<A, RANK> {
    pub fn map<B, F>(
        self: RankedDifferentiable<A, RANK>,
        f: &mut F,
    ) -> RankedDifferentiable<B, RANK>
    where
        F: FnMut(Scalar<A>) -> Scalar<B>,
        A: Clone,
    {
        self.map_tagged(f)
    }

    pub fn map2<B, C, F>(
        self: &RankedDifferentiable<A, RANK>,
        other: &RankedDifferentiable<B, RANK>,
        f: &mut F,
    ) -> RankedDifferentiable<C, RANK>
    where
        F: FnMut(&Scalar<A>, &Scalar<B>) -> Scalar<C>,
        A: Clone,
        B: Clone,
    {
        self.map2_tagged(other, &mut |a, (), b, ()| (f(a, b), ()))
    }

    pub fn map2_once<B, C, F, const RANK_B: usize, const RANK_OUT: usize>(
        self: &RankedDifferentiable<A, RANK>,
        other: &RankedDifferentiable<B, RANK_B>,
        f: &mut F,
    ) -> RankedDifferentiable<C, RANK_OUT>
    where
        F: FnMut(
            &RankedDifferentiable<A, { RANK - 1 }>,
            &RankedDifferentiable<B, { RANK_B - 1 }>,
        ) -> RankedDifferentiable<C, { RANK_OUT - 1 }>,
        A: Clone,
        B: Clone,
    {
        self.map2_once_tagged(other, f)
    }
}

pub fn grad<A, Tag, F, const RANK: usize, const PARAM_RANK: usize>(
    mut f: F,
    theta: &[DifferentiableTagged<A, Tag>; PARAM_RANK],
) -> [DifferentiableTagged<A, Tag>; PARAM_RANK]
where
    F: FnMut(
        &[DifferentiableTagged<A, Tag>; PARAM_RANK],
    ) -> RankedDifferentiableTagged<A, Tag, RANK>,
    A: ?Sized
        + Clone
        + Hash
        + AddAssign
        + Add<Output = A>
        + Mul<Output = A>
        + Exp
        + Div<Output = A>
        + Zero
        + One
        + Neg<Output = A>
        + Sqrt
        + Eq,
    Tag: Clone,
{
    let mut i = 0usize;
    let wrt = theta.each_ref().map(|theta| {
        theta.map(&mut |x| {
            let result = Scalar::truncate_dual(x, Some(i));
            i += 1;
            result
        })
    });
    let after_f = f(&wrt);
    DifferentiableContents::grad_once(after_f.contents.contents, wrt)
}

#[cfg(test)]
mod tests {
    use ordered_float::NotNan;

    use crate::loss::{l2_loss_2, predict_line_2_unranked};
    use crate::not_nan::to_not_nan_1;

    use super::*;

    fn extract_scalar<A, Tag>(d: &DifferentiableTagged<A, Tag>) -> &A {
        d.borrow_scalar().real_part()
    }

    #[test]
    fn test_map() {
        let v = DifferentiableTagged::of_vec(vec![
            Differentiable::of_scalar(Scalar::Number(
                NotNan::new(3.0).expect("3 is not NaN"),
                Some(0usize),
            )),
            DifferentiableTagged::of_scalar(Scalar::Number(
                NotNan::new(4.0).expect("4 is not NaN"),
                Some(1usize),
            )),
        ]);
        let mapped = v.map(&mut |x: Scalar<NotNan<f64>>| match x {
            Scalar::Number(i, n) => Scalar::Number(i + NotNan::new(1.0).expect("1 is not NaN"), n),
            Scalar::Dual(_, _) => panic!("Not hit"),
        });

        let v = mapped
            .into_vector()
            .iter()
            .map(|d| *extract_scalar(d))
            .collect::<Vec<_>>();

        assert_eq!(v, [4.0, 5.0]);
    }

    #[test]
    fn test_autodiff() {
        let input_vec = [
            RankedDifferentiableTagged::of_scalar(Scalar::<NotNan<f64>>::zero()).contents,
            RankedDifferentiableTagged::of_scalar(Scalar::<NotNan<f64>>::zero()).contents,
        ];
        let xs = [2.0, 1.0, 4.0, 3.0].map(|x| NotNan::new(x).expect("not nan"));
        let ys = [1.8, 1.2, 4.2, 3.3].map(|x| NotNan::new(x).expect("not nan"));
        let grad = grad(
            |x| {
                RankedDifferentiableTagged::of_vector(vec![RankedDifferentiable::of_scalar(
                    l2_loss_2(
                        predict_line_2_unranked,
                        RankedDifferentiableTagged::of_slice(xs.iter()),
                        RankedDifferentiableTagged::of_slice(ys.iter()),
                        x,
                    ),
                )])
            },
            &input_vec,
        );

        let grad_vec = grad
            .map(DifferentiableTagged::into_scalar)
            .map(|x| f64::from(*x.real_part()));
        assert_eq!(grad_vec, [-63.0, -21.0]);
    }

    #[test]
    fn grad_example() {
        let input_vec = [DifferentiableTagged::of_scalar(Scalar::make(
            NotNan::new(27.0).expect("not nan"),
        ))];

        let grad: Vec<_> = grad(
            |x| {
                RankedDifferentiableTagged::of_scalar(
                    x[0].borrow_scalar().clone() * x[0].borrow_scalar().clone(),
                )
            },
            &input_vec,
        )
        .into_iter()
        .map(|x| x.into_scalar().real_part().into_inner())
        .collect();
        assert_eq!(grad, [54.0]);
    }

    #[test]
    fn loss_gradient() {
        let zero = Scalar::<NotNan<f64>>::zero();
        let input_vec = [
            RankedDifferentiableTagged::of_scalar(zero.clone()).to_unranked(),
            RankedDifferentiableTagged::of_scalar(zero).to_unranked(),
        ];
        let xs = to_not_nan_1([2.0, 1.0, 4.0, 3.0]);
        let ys = to_not_nan_1([1.8, 1.2, 4.2, 3.3]);
        let grad = grad(
            |x| {
                RankedDifferentiableTagged::of_vector(vec![RankedDifferentiableTagged::of_scalar(
                    l2_loss_2(
                        predict_line_2_unranked,
                        RankedDifferentiableTagged::of_slice(&xs),
                        RankedDifferentiableTagged::of_slice(&ys),
                        x,
                    ),
                )])
            },
            &input_vec,
        );

        assert_eq!(
            grad.into_iter()
                .map(|x| *(x.into_scalar().real_part()))
                .collect::<Vec<_>>(),
            [-63.0, -21.0]
        );
    }
}
