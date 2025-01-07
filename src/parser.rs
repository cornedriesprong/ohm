use koto::{derive::*, prelude::*};

#[derive(Debug, Clone, Copy, PartialEq)]
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

impl Expr {
    fn apply_number_op(&self, num: &KNumber, op: impl Fn(f32, f32) -> f32) -> koto::Result<KValue> {
        match self {
            Self::Number(value) => Ok(KValue::Number(op(*value, f32::from(num)).into())),
            _ => panic!("invalid number operation"),
        }
    }

    fn apply_operator(
        &self,
        rhs: &KValue,
        kind: OperatorType,
        num_transform: impl Fn(f32) -> f32,
    ) -> koto::Result<KValue> {
        match rhs {
            KValue::Number(num) => Ok(KValue::Object(
                Expr::Operator {
                    kind,
                    input: Box::new(self.clone()),
                    args: vec![Expr::Number(num_transform(f32::from(num)).into())],
                }
                .into(),
            )),
            KValue::Object(obj) => Ok(KValue::Object(
                Expr::Operator {
                    kind,
                    input: Box::new(self.clone()),
                    args: vec![obj.cast::<Expr>()?.clone()],
                }
                .into(),
            )),
            _ => panic!("invalid operator argument"),
        }
    }
}

impl KotoObject for Expr {
    fn add(&self, rhs: &KValue) -> koto::Result<KValue> {
        match (self, rhs) {
            (Self::Number(_), KValue::Number(num)) => self.apply_number_op(num, |a, b| a + b),
            (Self::Operator { .. }, _) => self.apply_operator(rhs, OperatorType::Offset, |x| x),
            _ => panic!("invalid add operation"),
        }
    }

    fn subtract(&self, rhs: &KValue) -> koto::Result<KValue> {
        match (self, rhs) {
            (Self::Number(_), KValue::Number(num)) => self.apply_number_op(num, |a, b| a - b),
            (Self::Operator { .. }, _) => self.apply_operator(rhs, OperatorType::Offset, |x| -x),
            _ => panic!("invalid subtract operation"),
        }
    }

    fn multiply(&self, rhs: &KValue) -> koto::Result<KValue> {
        match (self, rhs) {
            (Self::Number(_), KValue::Number(num)) => self.apply_number_op(num, |a, b| a * b),
            (Self::Operator { .. }, _) => self.apply_operator(rhs, OperatorType::Gain, |x| x),
            _ => panic!("invalid multiply operation"),
        }
    }

    fn divide(&self, rhs: &KValue) -> koto::Result<KValue> {
        match (self, rhs) {
            (Self::Number(_), KValue::Number(num)) => self.apply_number_op(num, |a, b| a / b),
            (Self::Operator { .. }, _) => self.apply_operator(rhs, OperatorType::Gain, |x| 1.0 / x),
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

    fn assert_number_result(result: koto::Result<KValue>, expected: f32) {
        match result {
            Ok(KValue::Number(n)) => assert_eq!(f32::from(n), expected),
            _ => panic!("Expected number result"),
        }
    }

    fn assert_operator_result(
        result: koto::Result<KValue>,
        expected_kind: OperatorType,
        expected_arg: f32,
    ) {
        match result {
            Ok(KValue::Object(obj)) => {
                let expr = obj.cast::<Expr>().unwrap().clone();
                match expr {
                    Expr::Operator { kind, args, .. } => {
                        assert_eq!(kind, expected_kind);
                        match &args[0] {
                            Expr::Number(n) => assert_eq!(*n, expected_arg),
                            _ => panic!("Expected number argument"),
                        }
                    }
                    _ => panic!("Expected operator"),
                }
            }
            _ => panic!("Expected operator result"),
        }
    }

    #[test]
    fn test_number_operations() {
        let expr = Expr::Number(10.0);
        let num = KValue::Number(5.0.into());

        assert_number_result(expr.add(&num), 15.0);
        assert_number_result(expr.subtract(&num), 5.0);
        assert_number_result(expr.multiply(&num), 50.0);
        assert_number_result(expr.divide(&num), 2.0);
    }

    #[test]
    fn test_operator_arithmetic() {
        let base_op = Expr::Operator {
            kind: OperatorType::Sine,
            input: Box::new(Expr::Null),
            args: vec![],
        };
        let num = KValue::Number(5.0.into());

        assert_operator_result(base_op.add(&num), OperatorType::Offset, 5.0);
        assert_operator_result(base_op.subtract(&num), OperatorType::Offset, -5.0);
        assert_operator_result(base_op.multiply(&num), OperatorType::Gain, 5.0);
        assert_operator_result(base_op.divide(&num), OperatorType::Gain, 0.2);
    }

    #[test]
    fn test_operator_with_expr() {
        let base_op = Expr::Operator {
            kind: OperatorType::Sine,
            input: Box::new(Expr::Null),
            args: vec![],
        };

        let nested_expr = Expr::Number(3.0);
        let obj_value = KValue::Object(nested_expr.into());

        // Test that we can create an operator with another expression as argument
        match base_op.add(&obj_value) {
            Ok(KValue::Object(obj)) => {
                let expr = obj.cast::<Expr>().unwrap().clone();
                match expr {
                    Expr::Operator { kind, args, .. } => {
                        assert!(matches!(kind, OperatorType::Offset));
                        assert!(matches!(&args[0], Expr::Number(3.0)));
                    }
                    _ => panic!("Expected operator"),
                }
            }
            _ => panic!("Expected operator result"),
        }
    }

    #[test]
    #[should_panic(expected = "invalid number operation")]
    fn test_invalid_number_operation() {
        let op = Expr::Operator {
            kind: OperatorType::Sine,
            input: Box::new(Expr::Null),
            args: vec![],
        };
        let num = KNumber::from(5.0);
        let _ = op.apply_number_op(&num, |a, b| a + b);
    }

    #[test]
    #[should_panic(expected = "invalid operator argument")]
    fn test_invalid_operator_argument() {
        let op = Expr::Operator {
            kind: OperatorType::Sine,
            input: Box::new(Expr::Null),
            args: vec![],
        };
        let invalid_arg = KValue::Null;
        let _ = op.apply_operator(&invalid_arg, OperatorType::Offset, |x| x);
    }
}
