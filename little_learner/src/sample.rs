use rand::Rng;

/// Grab `n` random samples from `from_x` and `from_y`, collecting them into a vector.
pub fn sample2<R: Rng, T, U, I, J>(rng: &mut R, n: usize, from_x: I, from_y: J) -> (Vec<T>, Vec<U>)
where
    T: Copy,
    U: Copy,
    I: AsRef<[T]>,
    J: AsRef<[U]>,
{
    let from_x = from_x.as_ref();
    let from_y = from_y.as_ref();
    let mut out_x = Vec::with_capacity(n);
    let mut out_y = Vec::with_capacity(n);
    for _ in 0..n {
        let sample = rng.gen_range(0..from_x.len());
        out_x.push(from_x[sample]);
        out_y.push(from_y[sample]);
    }
    (out_x, out_y)
}
