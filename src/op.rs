use crate::nodes::{Frame, Node, NodeKind};
use crate::KObject;
use koto::derive::{KotoCopy, KotoType};
use koto::runtime::{KMap, KValue, KotoCopy, KotoEntries, KotoObject, Result};
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
    Power(Box<Op>, Box<Op>),
    Greater(Box<Op>, Box<Op>),
    Less(Box<Op>, Box<Op>),
    Equal(Box<Op>, Box<Op>),
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
    Power,
    Modulo,
}

impl Op {
    // Helper to choose the operation variant for non-numeric binary operations.
    fn op_variant(&self, rhs: Op, op: BinaryOp) -> Op {
        match op {
            BinaryOp::Add | BinaryOp::Subtract => Op::Mix(Box::new(self.clone()), Box::new(rhs)),
            BinaryOp::Multiply | BinaryOp::Divide => {
                Op::Gain(Box::new(self.clone()), Box::new(rhs))
            }
            BinaryOp::Power => Op::Power(Box::new(self.clone()), Box::new(rhs)),
            BinaryOp::Modulo => Op::Wrap(Box::new(self.clone()), Box::new(rhs)),
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
                    BinaryOp::Power => lhs.powf(rhs_val),
                    BinaryOp::Modulo => lhs % rhs_val,
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
            Op::Power(lhs, rhs) => {
                8u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            Op::Greater(lhs, rhs) => {
                9u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            Op::Less(lhs, rhs) => {
                10u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            Op::Equal(lhs, rhs) => {
                11u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
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

    fn power(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Power)
    }

    fn power_rhs(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Power)
    }

    fn remainder(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Modulo)
    }

    fn remainder_rhs(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Modulo)
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
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        const EMPTY: &[Frame] = &[];
        let chunk_size = outputs.len();

        match self {
            Op::Constant(val) => {
                let frame = [*val; 2];
                outputs.fill(frame);
            }
            Op::Node { kind, node, .. } => match kind {
                NodeKind::Log => {
                    let input = inputs.get(0).unwrap_or(&EMPTY);
                    for i in 0..chunk_size {
                        let frame = input.get(i).copied().unwrap_or([0.0; 2]);
                        println!("{:?}", frame[0]);
                        outputs[i] = frame;
                    }
                }
                _ => {
                    node.process(inputs, outputs);
                }
            },
            Op::Gain { .. } => {
                let in0 = inputs.get(0).unwrap_or(&EMPTY);
                let in1 = inputs.get(1).unwrap_or(&EMPTY);
                for i in 0..chunk_size {
                    let [l0, r0] = in0.get(i).copied().unwrap_or([0.0; 2]);
                    let [l1, r1] = in1.get(i).copied().unwrap_or([0.0; 2]);
                    outputs[i] = [l0 * l1, r0 * r1];
                }
            }
            Op::Mix { .. } => {
                let in0 = inputs.get(0).unwrap_or(&EMPTY);
                let in1 = inputs.get(1).unwrap_or(&EMPTY);
                for i in 0..chunk_size {
                    let [l0, r0] = in0.get(i).copied().unwrap_or([0.0; 2]);
                    let [l1, r1] = in1.get(i).copied().unwrap_or([0.0; 2]);
                    outputs[i] = [l0 + l1, r0 + r1];
                }
            }
            Op::Wrap { .. } => {
                let in0 = inputs.get(0).unwrap_or(&EMPTY);
                let in1 = inputs.get(1).unwrap_or(&EMPTY);
                for i in 0..chunk_size {
                    let [l0, r0] = in0.get(i).copied().unwrap_or([0.0; 2]);
                    let [l1, r1] = in1.get(i).copied().unwrap_or([0.0; 2]);
                    outputs[i] = [l0 % l1, r0 % r1];
                }
            }
            Op::Negate { .. } => {
                let input = inputs.get(0).unwrap_or(&EMPTY);
                for i in 0..chunk_size {
                    let [l, r] = input.get(i).copied().unwrap_or([0.0; 2]);
                    outputs[i] = [-l, -r];
                }
            }
            Op::Power { .. } => {
                let in0 = inputs.get(0).unwrap_or(&EMPTY);
                let in1 = inputs.get(1).unwrap_or(&EMPTY);
                for i in 0..chunk_size {
                    let [l0, r0] = in0.get(i).copied().unwrap_or([0.0; 2]);
                    let [l1, r1] = in1.get(i).copied().unwrap_or([0.0; 2]);
                    outputs[i] = [l0.powf(l1), r0.powf(r1)];
                }
            }
            Op::Greater { .. } => {
                let in0 = inputs.get(0).unwrap_or(&EMPTY);
                let in1 = inputs.get(1).unwrap_or(&EMPTY);
                for i in 0..chunk_size {
                    let [l0, r0] = in0.get(i).copied().unwrap_or([0.0; 2]);
                    let [l1, r1] = in1.get(i).copied().unwrap_or([0.0; 2]);
                    outputs[i] = [
                        if l0 > l1 { 1.0 } else { 0.0 },
                        if r0 > r1 { 1.0 } else { 0.0 },
                    ];
                }
            }
            Op::Less { .. } => {
                let in0 = inputs.get(0).unwrap_or(&EMPTY);
                let in1 = inputs.get(1).unwrap_or(&EMPTY);
                for i in 0..chunk_size {
                    let [l0, r0] = in0.get(i).copied().unwrap_or([0.0; 2]);
                    let [l1, r1] = in1.get(i).copied().unwrap_or([0.0; 2]);
                    outputs[i] = [
                        if l0 < l1 { 1.0 } else { 0.0 },
                        if r0 < r1 { 1.0 } else { 0.0 },
                    ];
                }
            }
            Op::Equal { .. } => {
                let in0 = inputs.get(0).unwrap_or(&EMPTY);
                let in1 = inputs.get(1).unwrap_or(&EMPTY);
                for i in 0..chunk_size {
                    let [l0, r0] = in0.get(i).copied().unwrap_or([0.0; 2]);
                    let [l1, r1] = in1.get(i).copied().unwrap_or([0.0; 2]);
                    outputs[i] = [
                        if l0 == l1 { 1.0 } else { 0.0 },
                        if r0 == r1 { 1.0 } else { 0.0 },
                    ];
                }
            }
        }
    }

    #[inline(always)]
    fn process_read_buffer(
        &mut self,
        inputs: &[&[Frame]],
        buffer: &[Frame],
        outputs: &mut [Frame],
    ) {
        match self {
            Op::Node { kind, node, .. } => match kind {
                NodeKind::BufferReader { .. } | NodeKind::BufferTap { .. } => {
                    node.process_read_buffer(inputs, buffer, outputs)
                }
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    fn process_write_buffer(&mut self, inputs: &[&[Frame]], buffer: &mut [Frame]) {
        match self {
            Op::Node { kind, node, .. } => match kind {
                NodeKind::BufferWriter { .. } => node.process_write_buffer(inputs, buffer),
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        }
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}
