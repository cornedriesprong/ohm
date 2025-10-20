use crate::nodes::{Frame, Node, NodeKind};
use rtsan_standalone::nonblocking;
use seahash::SeaHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
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

impl Op {
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
