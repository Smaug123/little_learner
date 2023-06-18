use crate::auto_diff::RankedDifferentiableTagged;

pub fn argmax_1<A, Tag>(t: &RankedDifferentiableTagged<A, Tag, 1>) -> usize
where
    A: Clone + PartialOrd + Ord,
{
    t.to_unranked_borrow()
        .borrow_vector()
        .iter()
        .map(|x| x.borrow_scalar().clone_real_part())
        .enumerate()
        .max_by_key(|(_, y)| y.clone())
        .unwrap()
        .0
}

pub fn one_hot_class_eq<A, Tag1, Tag2>(
    t1: &RankedDifferentiableTagged<A, Tag1, 1>,
    t2: &RankedDifferentiableTagged<A, Tag2, 1>,
) -> bool
where
    A: Clone + Ord + PartialOrd,
{
    argmax_1(t1) == argmax_1(t2)
}
