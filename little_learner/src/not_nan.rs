use ordered_float::NotNan;

pub fn to_not_nan_1<T, const N: usize>(xs: [T; N]) -> [NotNan<T>; N]
where
    T: ordered_float::Float,
{
    xs.map(|x| NotNan::new(x).expect("not nan"))
}

pub fn to_not_nan_2<T, const N: usize, const M: usize>(xs: [[T; N]; M]) -> [[NotNan<T>; N]; M]
where
    T: ordered_float::Float,
{
    xs.map(to_not_nan_1)
}
