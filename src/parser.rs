use koto::{derive::*, prelude::*};

#[derive(Debug, Clone, Copy)]
pub(crate) enum OperatorType {
    Sine,
    Square,
    Noise,
    Skew,
    Offset,
    Gain,
}

#[derive(Debug, Clone, KotoType, KotoCopy)]
pub(crate) enum Expr {
    Operator {
        kind: OperatorType,
        input: Box<Expr>,
        args: Vec<Expr>,
    },
    Number(f32),
    Null,
}

impl KotoObject for Expr {
    fn add(&self, rhs: &KValue) -> koto::Result<KValue> {
        match (self, rhs) {
            (Self::Number(value), KValue::Number(num)) => {
                Ok(KValue::Number((value + f32::from(num)).into()))
            }
            (Self::Operator { .. }, KValue::Number(num)) => Ok(KValue::Object(
                Expr::Operator {
                    kind: OperatorType::Offset,
                    input: Box::new(self.clone()),
                    args: vec![Expr::Number(num.into())],
                }
                .into(),
            )),
            _ => panic!("invalid add operation"),
        }
    }

    fn subtract(&self, rhs: &KValue) -> koto::Result<KValue> {
        match (self, rhs) {
            (Self::Number(value), KValue::Number(num)) => {
                Ok(KValue::Number((value - f32::from(num)).into()))
            }
            (Self::Operator { .. }, KValue::Number(num)) => Ok(KValue::Object(
                Expr::Operator {
                    kind: OperatorType::Offset,
                    input: Box::new(self.clone()),
                    args: vec![Expr::Number((-f32::from(num)).into())],
                }
                .into(),
            )),
            // TODO: handle case where rhs is an operator
            (Self::Operator { .. }, KValue::Object(obj)) => Ok(KValue::Object(
                Expr::Operator {
                    kind: OperatorType::Offset,
                    input: Box::new(self.clone()),
                    args: vec![obj.cast::<Expr>()?.clone()],
                }
                .into(),
            )),
            _ => panic!("invalid subtract operation"),
        }
    }

    fn multiply(&self, rhs: &KValue) -> koto::Result<KValue> {
        match (self, rhs) {
            (Self::Number(value), KValue::Number(num)) => {
                Ok(KValue::Number((value * f32::from(num)).into()))
            }
            (Self::Operator { .. }, KValue::Number(num)) => Ok(KValue::Object(
                Expr::Operator {
                    kind: OperatorType::Gain,
                    input: Box::new(self.clone()),
                    args: vec![Expr::Number(num.into())],
                }
                .into(),
            )),
            _ => panic!("invalid multiply operation"),
        }
    }

    fn divide(&self, rhs: &KValue) -> koto::Result<KValue> {
        match (self, rhs) {
            (Self::Number(value), KValue::Number(num)) => {
                Ok(KValue::Number((value / f32::from(num)).into()))
            }
            (Self::Operator { .. }, KValue::Number(num)) => Ok(KValue::Object(
                Expr::Operator {
                    kind: OperatorType::Gain,
                    input: Box::new(self.clone()),
                    args: vec![Expr::Number((1.0 / f32::from(num)).into())],
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
