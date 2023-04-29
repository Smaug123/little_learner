use rand::Rng;

// Generates numbers to fill the input slice, between 0 (inclusive)
// and `max` (exclusive).
fn sample<R: Rng, T, I>(rng: &mut R, from: I, out: &mut [T])
where
    T: Copy,
    I: AsRef<[T]>,
{
    let from = from.as_ref();
    for item in out {
        *item = from[rng.gen_range(0..from.len())]
    }
}
