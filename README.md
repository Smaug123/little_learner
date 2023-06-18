# The Little Learner, in Rust

[![Rust](https://github.com/Smaug123/little_learner/actions/workflows/rust.yml/badge.svg)](https://github.com/Smaug123/little_learner/actions/workflows/rust.yml)

Me running through [The Little Learner](https://www.thelittlelearner.com/), but in Rust instead of Scheme.

I started out by trying to make it reasonably type-safe, but the further I got through the book, the more strongly its style resisted being made safe.
So now there are many vestiges of the old type-safety around, but the whole thing is primarily unsafe, and all that's left is quite a lot of annoying friction.
