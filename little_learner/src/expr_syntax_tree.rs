use immutable_chunkmap::map;
use std::ops::{Add, Mul};

/*
An untyped syntax tree for an expression whose constants are all of type `A`.
*/
#[derive(Clone, Debug)]
pub enum Expr<A> {
    Const(A),
    Sum(Box<Expr<A>>, Box<Expr<A>>),
    Variable(u32),
    // The first `Expr` here is a function, which may reference the input variable `Variable(i)`.
    // For example, `(fun x y -> x + y) 3 4` is expressed as:
    // Apply(0, Apply(1, Sum(Variable(0), Variable(1)), Const(4)), Const(3))
    Apply(u32, Box<Expr<A>>, Box<Expr<A>>),
    Mul(Box<Expr<A>>, Box<Expr<A>>),
}

impl<A> Expr<A> {
    fn eval_inner<const SIZE: usize>(e: &Expr<A>, ctx: &map::Map<u32, A, SIZE>) -> A
    where
        A: Clone + Add<Output = A> + Mul<Output = A>,
    {
        match &e {
            Expr::Const(x) => x.clone(),
            Expr::Sum(x, y) => Expr::eval_inner(x, ctx) + Expr::eval_inner(y, ctx),
            Expr::Variable(id) => ctx
                .get(id)
                .unwrap_or_else(|| panic!("No binding found for free variable {}", id))
                .clone(),
            Expr::Apply(variable, func, arg) => {
                let arg = Expr::eval_inner(arg, ctx);
                let (updated_context, _) = ctx.insert(*variable, arg);
                Expr::eval_inner(func, &updated_context)
            }
            Expr::Mul(x, y) => Expr::eval_inner(x, ctx) * Expr::eval_inner(y, ctx),
        }
    }

    pub fn eval<const MAX_VAR_NUM: usize>(e: &Expr<A>) -> A
    where
        A: Clone + Add<Output = A> + Mul<Output = A>,
    {
        Expr::eval_inner(e, &map::Map::<u32, A, MAX_VAR_NUM>::new())
    }

    pub fn apply(var: u32, f: Expr<A>, arg: Expr<A>) -> Expr<A> {
        Expr::Apply(var, Box::new(f), Box::new(arg))
    }

    pub fn differentiate(one: &A, zero: &A, var: u32, f: &Expr<A>) -> Expr<A>
    where
        A: Clone,
    {
        match f {
            Expr::Const(_) => Expr::Const(zero.clone()),
            Expr::Sum(x, y) => {
                Expr::differentiate(one, zero, var, x) + Expr::differentiate(one, zero, var, y)
            }
            Expr::Variable(i) => {
                if *i == var {
                    Expr::Const(one.clone())
                } else {
                    Expr::Const(zero.clone())
                }
            }
            Expr::Mul(x, y) => {
                Expr::Mul(
                    Box::new(Expr::differentiate(one, zero, var, x.as_ref())),
                    (*y).clone(),
                ) + Expr::Mul(
                    Box::new(Expr::differentiate(one, zero, var, y.as_ref())),
                    (*x).clone(),
                )
            }
            Expr::Apply(new_var, func, expr) => {
                if *new_var == var {
                    panic!(
                        "cannot differentiate with respect to variable {} that's been assigned",
                        var
                    )
                }
                let expr_deriv = Expr::differentiate(one, zero, var, expr);
                Expr::mul(
                    expr_deriv,
                    Expr::Apply(
                        *new_var,
                        Box::new(Expr::differentiate(one, zero, *new_var, func)),
                        (*expr).clone(),
                    ),
                )
            }
        }
    }
}

impl<A> Add for Expr<A> {
    type Output = Expr<A>;
    fn add(self: Expr<A>, y: Expr<A>) -> Expr<A> {
        Expr::Sum(Box::new(self), Box::new(y))
    }
}

impl<A> Mul for Expr<A> {
    type Output = Expr<A>;
    fn mul(self: Expr<A>, y: Expr<A>) -> Expr<A> {
        Expr::Mul(Box::new(self), Box::new(y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr() {
        let expr = Expr::apply(
            0,
            Expr::apply(1, Expr::Variable(0) + Expr::Variable(1), Expr::Const(4)),
            Expr::Const(3),
        );

        assert_eq!(Expr::eval::<2>(&expr), 7);
    }

    #[test]
    fn test_derivative() {
        let add_four = Expr::Variable(0) + Expr::Const(4);
        let mul_five = Expr::Variable(1) * Expr::Const(5);

        {
            let mul_five_then_add_four = Expr::apply(0, add_four.clone(), mul_five.clone());
            let mul_then_add_diff = Expr::differentiate(&1, &0, 1, &mul_five_then_add_four);
            for i in 3..10 {
                // (5x + 4) differentiates to 5
                assert_eq!(
                    Expr::eval::<2>(&Expr::apply(1, mul_then_add_diff.clone(), Expr::Const(i))),
                    5
                );
            }
        }

        {
            let add_four_then_mul_five = Expr::apply(1, mul_five.clone(), add_four.clone());
            let add_then_mul_diff = Expr::differentiate(&1, &0, 0, &add_four_then_mul_five);
            for i in 3..10 {
                // ((x + 4) * 5) differentiates to 5
                assert_eq!(
                    Expr::eval::<2>(&Expr::apply(0, add_then_mul_diff.clone(), Expr::Const(i))),
                    5
                );
            }
        }
    }
}
