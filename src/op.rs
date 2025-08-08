use crate::nodes::{Frame, Node, NodeKind};
use crate::KObject;
use crate::KString;
use koto::derive::{KotoCopy, KotoType};
use koto::runtime::{KMap, KValue, KotoCopy, KotoEntries, KotoObject, KotoType, Result};
use rtsan_standalone::nonblocking;
use seahash::SeaHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone, KotoType, KotoCopy)]
pub enum Op {
    Constant(f32),
    Mix(Box<Op>, Box<Op>),
    Gain(Box<Op>, Box<Op>),
    Wrap(Box<Op>, Box<Op>),
    Negate(Box<Op>),
    Node {
        kind: NodeKind,
        inputs: Vec<Op>,
        node: Box<dyn Node>,
    },
}

enum BinaryOp {
    Add,
    Multiply,
    Subtract,
    Divide,
}

impl Op {
    // Helper to operate on two constant values.
    fn op_constant<F>(lhs: f32, rhs: f32, f: F) -> Op
    where
        F: Fn(f32, f32) -> f32,
    {
        Op::Constant(f(lhs, rhs))
    }

    // Helper to choose the operation variant for non-numeric binary operations.
    fn op_variant(&self, rhs: Op, op: BinaryOp) -> Op {
        match op {
            BinaryOp::Add | BinaryOp::Subtract => Op::Mix(Box::new(self.clone()), Box::new(rhs)),
            BinaryOp::Multiply | BinaryOp::Divide => {
                Op::Gain(Box::new(self.clone()), Box::new(rhs))
            }
        }
    }

    fn binary_op(&self, rhs: &KValue, op: BinaryOp) -> Result<KValue> {
        // Convert KValue to Op.
        let rhs_to_op = |num: &KValue| -> Result<Op> {
            match num {
                KValue::Number(n) => Ok(Op::Constant(f32::from(n))),
                KValue::Object(obj) => Ok(obj.cast::<Op>()?.clone()),
                _ => panic!("invalid type"),
            }
        };

        match (self, rhs) {
            (Op::Constant(lhs), KValue::Number(num)) => {
                let rhs_val = f32::from(num);
                let result = match op {
                    BinaryOp::Add => lhs + rhs_val,
                    BinaryOp::Multiply => lhs * rhs_val,
                    BinaryOp::Subtract => lhs - rhs_val,
                    BinaryOp::Divide => lhs / rhs_val,
                };
                Ok(KValue::Object(Op::Constant(result).into()))
            }
            _ => {
                let base_rhs = match op {
                    BinaryOp::Subtract => match rhs_to_op(rhs)? {
                        Op::Constant(n) => Op::Constant(-n),
                        op => op,
                    },
                    BinaryOp::Divide => match rhs_to_op(rhs)? {
                        Op::Constant(n) => Op::Constant(1.0 / n),
                        op => op,
                    },
                    _ => rhs_to_op(rhs)?,
                };
                Ok(KValue::Object(self.op_variant(base_rhs, op).into()))
            }
        }
    }

    /// Generic helper to compare self with the other KValue by first extracting a constant value.
    fn compare<F>(&self, other: &KValue, cmp: F) -> Result<bool>
    where
        F: Fn(f32, f32) -> bool,
    {
        // Helper to extract constant from KValue.
        fn extract_constant(val: &KValue) -> Option<f32> {
            match val {
                KValue::Number(num) => Some(f32::from(num)),
                KValue::Object(obj) => {
                    let op = obj.cast::<Op>().ok()?;
                    if let Op::Constant(n) = *op {
                        Some(n)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }

        if let Some(lhs_val) = match self {
            Op::Constant(val) => Some(*val),
            _ => {
                // If self is not constant attempt to extract a constant.
                if let Op::Constant(val) = self {
                    Some(*val)
                } else {
                    None
                }
            }
        } {
            if let Some(rhs_val) = extract_constant(other) {
                Ok(cmp(lhs_val, rhs_val))
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    pub(crate) fn compute_hash(&self) -> u64 {
        let mut hasher = SeaHasher::new();
        self.hash_structure(&mut hasher);
        hasher.finish()
    }

    fn hash_structure(&self, hasher: &mut SeaHasher) {
        match self {
            Op::Constant(val) => {
                0u8.hash(hasher);
                val.to_bits().hash(hasher);
            }
            Op::Node { kind, .. } => match kind {
                NodeKind::BufferReader { id } => {
                    1u8.hash(hasher);
                    id.hash(hasher)
                }
                NodeKind::BufferTap { id } => {
                    2u8.hash(hasher);
                    id.hash(hasher)
                }
                NodeKind::BufferWriter { id } => {
                    3u8.hash(hasher);
                    id.hash(hasher)
                }
                _ => kind.hash(hasher),
            },
            Op::Mix(lhs, rhs) => {
                4u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            Op::Gain(lhs, rhs) => {
                5u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            Op::Wrap(lhs, rhs) => {
                6u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            Op::Negate(val) => {
                7u8.hash(hasher);
                val.hash_structure(hasher);
            }
        }
    }
}

impl KotoObject for Op {
    fn negate(&self) -> Result<KValue> {
        if let Op::Constant(val) = self {
            Ok(KValue::Number((-val).into()))
        } else {
            Ok(KValue::Object(Op::Negate(Box::new(self.clone())).into()))
        }
    }

    fn add(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Add)
    }

    fn add_rhs(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Add)
    }

    fn subtract(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Subtract)
    }

    fn subtract_rhs(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Subtract)
    }

    fn multiply(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Multiply)
    }

    fn multiply_rhs(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Multiply)
    }

    fn divide(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Divide)
    }

    fn divide_rhs(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Divide)
    }

    fn remainder(&self, other: &KValue) -> Result<KValue> {
        match (self, other) {
            (Op::Constant(lhs), KValue::Number(num)) => {
                Ok(KValue::Object(Op::Constant(lhs % f32::from(num)).into()))
            }
            _ => {
                let inv = match other {
                    KValue::Number(n) => Op::Constant(1.0 / f32::from(n)),
                    KValue::Object(obj) => {
                        let op_val = obj.cast::<Op>()?.clone();
                        Op::Constant(match op_val {
                            Op::Constant(n) => 1.0 / n,
                            _ => panic!("Invalid op for remainder"),
                        })
                    }
                    _ => panic!("invalid remainder operation"),
                };
                Ok(KValue::Object(
                    Op::Wrap(Box::new(self.clone()), Box::new(inv)).into(),
                ))
            }
        }
    }

    fn less(&self, other: &KValue) -> Result<bool> {
        self.compare(other, |a, b| a < b)
    }

    fn less_or_equal(&self, other: &KValue) -> Result<bool> {
        self.compare(other, |a, b| a <= b)
    }

    fn greater(&self, other: &KValue) -> Result<bool> {
        self.compare(other, |a, b| a > b)
    }

    fn greater_or_equal(&self, other: &KValue) -> Result<bool> {
        self.compare(other, |a, b| a >= b)
    }

    fn equal(&self, other: &KValue) -> Result<bool> {
        self.compare(other, |a, b| a == b)
    }
}

// necessary to satisfy KotoEntries trait
impl KotoEntries for Op {
    fn entries(&self) -> Option<KMap> {
        None
    }
}

impl Node for Op {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        match self {
            Op::Constant(val) => [*val; 2],
            Op::Node { kind, node, .. } => match kind {
                NodeKind::Print => {
                    println!("{:?}", inputs[0][0]);
                    inputs[0]
                }
                _ => node.tick(inputs),
            },
            Op::Gain { .. } => match inputs {
                [[l0, r0], [l1, r1]] => [l0 * l1, r0 * r1],
                _ => unimplemented!(),
            },
            Op::Mix { .. } => match inputs {
                [[l0, r0], [l1, r1]] => [l0 + l1, r0 + r1],
                _ => unimplemented!(),
            },
            Op::Wrap { .. } => match inputs {
                [[l0, r0], [l1, r1]] => [l0 % l1, r0 % r1],
                _ => unimplemented!(),
            },
            Op::Negate { .. } => match inputs {
                [[l, r]] => [-l, -r],
                _ => unimplemented!(),
            },
        }
    }

    #[inline(always)]
    fn tick_read_buffer(&mut self, inputs: &[Frame], buffer: &[Frame]) -> Frame {
        match self {
            Op::Node { kind, node, .. } => match kind {
                NodeKind::BufferReader { .. } | NodeKind::BufferTap { .. } => {
                    node.tick_read_buffer(inputs, buffer)
                }
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    fn tick_write_buffer(&mut self, inputs: &[Frame], buffer: &mut [Frame]) {
        match self {
            Op::Node { kind, node, .. } => match kind {
                NodeKind::BufferWriter { .. } => node.tick_write_buffer(inputs, buffer),
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        }
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}
