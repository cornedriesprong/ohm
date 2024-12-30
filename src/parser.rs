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
}

impl KotoObject for Expr {
    fn multiply(&self, rhs: &KValue) -> koto::Result<KValue> {
        match (self, rhs) {
            (Self::Number(value), KValue::Number(num)) => {
                Ok(KValue::Number((value * (num.as_i64() as f32)).into()))
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
}

impl KotoEntries for Expr {
    fn entries(&self) -> Option<KMap> {
        None
    }
}
