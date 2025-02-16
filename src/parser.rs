use koto::{derive::*, prelude::*, runtime::Result};

type Signal = Box<Expr>;

#[derive(Debug, Clone, KotoType, KotoCopy)]
pub(crate) enum Expr {
    Constant(f32),
    Sine {
        freq: Signal,
    },
    Square {
        freq: Signal,
    },
    Saw {
        freq: Signal,
    },
    Pulse {
        freq: Signal,
    },
    Noise,
    Mix {
        lhs: Signal,
        rhs: Signal,
    },
    Gain {
        lhs: Signal,
        rhs: Signal,
    },
    AR {
        trig: Signal,
        attack: Signal,
        release: Signal,
    },
    SVF {
        cutoff: Signal,
        resonance: Signal,
        input: Signal,
    },
    Seq {
        seq: Vec<f32>,
        trig: Signal,
    },
}

impl KotoObject for Expr {
    // TODO: test these
    fn add(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(value), KValue::Number(num)) => Ok(KValue::Object(
                Expr::Constant(value + f32::from(num)).into(),
            )),
            (_, KValue::Number(num)) => Ok(KValue::Object(
                Expr::Mix {
                    lhs: Box::new(self.clone()),
                    rhs: Box::new(Expr::Constant(num.into())).into(),
                }
                .into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                Expr::Mix {
                    lhs: Box::new(self.clone()),
                    rhs: Box::new(obj.cast::<Expr>()?.clone()),
                }
                .into(),
            )),
            _ => panic!("invalid add operation"),
        }
    }

    fn multiply(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(value), KValue::Number(num)) => Ok(KValue::Object(
                Expr::Constant(value * f32::from(num)).into(),
            )),
            (_, KValue::Number(num)) => Ok(KValue::Object(
                Expr::Gain {
                    lhs: Box::new(self.clone()),
                    rhs: Box::new(Expr::Constant((num).into())),
                }
                .into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                Expr::Gain {
                    lhs: Box::new(self.clone()),
                    rhs: Box::new(obj.cast::<Expr>()?.clone()),
                }
                .into(),
            )),
            _ => panic!("invalid multiply operation"),
        }
    }

    fn subtract(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(value), KValue::Number(num)) => Ok(KValue::Object(
                Expr::Constant(value - f32::from(num)).into(),
            )),
            (_, KValue::Number(num)) => Ok(KValue::Object(
                Expr::Mix {
                    lhs: Box::new(self.clone()),
                    rhs: Box::new(Expr::Constant((-num).into())),
                }
                .into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                Expr::Mix {
                    lhs: Box::new(self.clone()),
                    rhs: Box::new(obj.cast::<Expr>()?.clone()),
                }
                .into(),
            )),
            _ => panic!("invalid subtract operation"),
        }
    }

    fn divide(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(value), KValue::Number(num)) => Ok(KValue::Object(
                Expr::Constant(value / f32::from(num)).into(),
            )),
            (_, KValue::Number(num)) => Ok(KValue::Object(
                Expr::Gain {
                    lhs: Box::new(self.clone()),
                    rhs: Box::new(Expr::Constant(num.into())),
                }
                .into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                Expr::Gain {
                    lhs: Box::new(self.clone()),
                    rhs: Box::new(obj.cast::<Expr>()?.clone()),
                }
                .into(),
            )),
            _ => panic!("invalid divide operation"),
        }
    }
}

impl KotoEntries for Expr {
    fn entries(&self) -> Option<KMap> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let expr = Expr::Constant(1.0);
        let result = expr.add(&KValue::Number(2.0.into())).unwrap();
    }
}
