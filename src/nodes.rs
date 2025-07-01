use crate::consts::SAMPLE_RATE;
use crate::dsp::delay::Delay;
use crate::utils::freq_to_period;
use core::fmt;
use fmt::Debug;
use fundsp::hacker32::AudioUnit;
use koto::{derive::*, prelude::*, runtime::Result};
use rand::Rng;
use rtsan_standalone::nonblocking;
use seahash::SeaHasher;
use std::f32::consts::PI;
use std::hash::{Hash, Hasher};

pub type Frame = [f32; 2];

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum NodeKind {
    Sin,
    Sqr,
    Saw,
    Tri,
    Ramp,
    Lp,
    Bp,
    Hp,
    Pulse,
    Print,
    Noise,
    Env,
    Seq,
    Pan,
    Pluck,
    Reverb,
    Delay,
    Moog,
    Wav,
}

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
    fn binary_op(&self, rhs: &KValue, op: BinaryOp) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(lhs), KValue::Number(rhs)) => {
                let result = match op {
                    BinaryOp::Add => lhs + f32::from(rhs),
                    BinaryOp::Multiply => lhs * f32::from(rhs),
                    BinaryOp::Subtract => lhs - f32::from(rhs),
                    BinaryOp::Divide => lhs / f32::from(rhs),
                };
                Ok(KValue::Object(Op::Constant(result).into()))
            }
            (_, KValue::Number(num)) => {
                let rhs_val = match op {
                    BinaryOp::Subtract => Op::Constant((-num).into()),
                    BinaryOp::Divide => Op::Constant(1.0 / f32::from(num)),
                    _ => Op::Constant(num.into()),
                };
                let op_variant = match op {
                    BinaryOp::Add | BinaryOp::Subtract => {
                        Op::Mix(Box::new(self.clone()), Box::new(rhs_val))
                    }
                    BinaryOp::Multiply | BinaryOp::Divide => {
                        Op::Gain(Box::new(self.clone()), Box::new(rhs_val))
                    }
                };
                Ok(KValue::Object(op_variant.into()))
            }
            (_, KValue::Object(obj)) => {
                let op_variant = match op {
                    BinaryOp::Add | BinaryOp::Subtract => {
                        Op::Mix(Box::new(self.clone()), Box::new(obj.cast::<Op>()?.clone()))
                    }
                    BinaryOp::Multiply | BinaryOp::Divide => {
                        Op::Gain(Box::new(self.clone()), Box::new(obj.cast::<Op>()?.clone()))
                    }
                };
                Ok(KValue::Object(op_variant.into()))
            }
            _ => {
                let op_name = match op {
                    BinaryOp::Add => "add",
                    BinaryOp::Multiply => "multiply",
                    BinaryOp::Subtract => "subtract",
                    BinaryOp::Divide => "divide",
                };
                panic!("invalid {} operation", op_name)
            }
        }
    }
}

impl KotoObject for Op {
    fn add(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Add)
    }

    fn add_rhs(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Add)
    }

    fn multiply(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Multiply)
    }

    fn multiply_rhs(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Multiply)
    }

    fn subtract(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Subtract)
    }

    fn subtract_rhs(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Subtract)
    }

    fn divide(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Divide)
    }

    fn divide_rhs(&self, rhs: &KValue) -> Result<KValue> {
        self.binary_op(rhs, BinaryOp::Divide)
    }

    fn remainder(&self, other: &KValue) -> Result<KValue> {
        match (self, other) {
            (Self::Constant(lhs), KValue::Number(rhs)) => {
                Ok(KValue::Object(Op::Constant(lhs % f32::from(rhs)).into()))
            }
            (_, KValue::Number(num)) => Ok(KValue::Object(
                Op::Wrap(
                    Box::new(self.clone()),
                    Box::new(Op::Constant(1.0 / f32::from(num))),
                )
                .into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                Op::Wrap(
                    Box::new(self.clone()),
                    Box::new(obj.cast::<Op>()?.clone()).into(),
                )
                .into(),
            )),
            _ => panic!("invalid remainder operation"),
        }
    }

    fn negate(&self) -> Result<KValue> {
        match self {
            Self::Constant(val) => Ok(KValue::Number((-val).into())),
            _ => Ok(KValue::Object(Op::Negate(Box::new(self.clone())).into())),
        }
    }
}

// necessary to satisfy KotoEntries trait
impl KotoEntries for Op {
    fn entries(&self) -> Option<KMap> {
        None
    }
}

impl Node for Op {
    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        match self {
            Op::Constant(val) => [*val; 2],

            Op::Node { kind, node, .. } => match kind {
                NodeKind::Print => {
                    println!("{:?}", inputs);
                    inputs[0]
                }
                _ => node.tick(inputs),
            },
            Op::Gain { .. } => match inputs {
                [[l0, r0], [l1, r1]] => [l0 * l1, r0 * r1],
                _ => inputs[0],
            },
            Op::Mix { .. } => match inputs {
                [[l0, r0], [l1, r1]] => [l0 + l1, r0 + r1],
                _ => inputs[0],
            },
            Op::Wrap { .. } => match inputs {
                [[l0, r0], [l1, r1]] => [l0 % l1, r0 % r1],
                _ => inputs[0],
            },
            Op::Negate { .. } => match inputs {
                [[l, r]] => [-l, -r],
                _ => inputs[0],
            },
        }
    }
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
            Op::Node {
                kind: node_type, ..
            } => {
                node_type.hash(hasher);
            }
            Op::Mix(lhs, rhs) => {
                1u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            Op::Gain(lhs, rhs) => {
                2u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            Op::Wrap(lhs, rhs) => {
                3u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            Op::Negate(val) => {
                4u8.hash(hasher);
                val.hash_structure(hasher);
            }
        }
    }
}

pub(crate) trait Node: Send + Sync {
    fn tick(&mut self, inputs: &[Frame]) -> Frame;
    fn clone_box(&self) -> Box<dyn Node>;
}

#[derive(Clone)]
pub struct FunDSPNode {
    node: Box<dyn AudioUnit>,
    is_stereo: bool,
}

impl FunDSPNode {
    pub fn mono(node: Box<dyn AudioUnit>) -> Self {
        Self {
            node,
            is_stereo: false,
        }
    }

    pub fn stereo(node: Box<dyn AudioUnit>) -> Self {
        Self {
            node,
            is_stereo: true,
        }
    }
}

impl Node for FunDSPNode {
    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        if self.is_stereo {
            let mut output = [0.0; 2];
            self.node.tick(inputs[0].as_slice(), &mut output);
            output
        } else {
            let input: Vec<f32> = inputs.iter().map(|[l, _]| *l).collect();
            let mut output = [0.0; 1];
            self.node.tick(input.as_slice(), &mut output);
            [output[0]; 2]
        }
    }
}

impl Clone for Box<dyn Node> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone, Debug)]
pub struct EnvSegment {
    pub duration: usize, // Duration in samples or ms (interpreted as needed)
    pub target: f32,     // Target value at end of this segment
}

#[derive(Clone)]
pub struct EnvNode {
    pub(crate) current_idx: usize,
    pub(crate) value: f32,
    pub(crate) time: usize,
    pub(crate) active: bool,
    pub(crate) prev_trig: f32,
}

impl EnvNode {
    pub(crate) fn new() -> Self {
        Self {
            current_idx: 0,
            value: 0.0,
            time: 0,
            active: false,
            prev_trig: 0.0,
        }
    }

    fn start(&mut self) {
        self.active = true;
        self.current_idx = 0;
        self.time = 0;
    }

    fn advance(&mut self, segments: &[EnvSegment]) {
        self.current_idx += 1;
        self.time = 0;

        if self.current_idx >= segments.len() {
            self.active = false;
            self.value = 0.0;
        }
    }

    fn interpolate_segment(&self, segments: &[EnvSegment]) -> f32 {
        let segment = &segments[self.current_idx];

        if segment.duration == 0 {
            return segment.target;
        }

        let pow = 3.0; // curve
        let raw_t = (self.time as f32 / segment.duration as f32).clamp(0.0, 1.0);

        // reverse the curve if we're going downward
        let curved_t = if segment.target < self.value {
            // ease out
            1.0 - (1.0 - raw_t).powf(pow)
        } else {
            // ease in
            raw_t.powf(pow)
        };

        let prev = if self.current_idx == 0 {
            0.0
        } else {
            segments[self.current_idx - 1].target
        };

        prev + curved_t * (segment.target - prev)
    }
}

impl Node for EnvNode {
    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let trig = inputs.last().expect("env: missing trigger input")[0];
        let segments = &inputs[0..inputs.len() - 1]
            .chunks_exact(2)
            .map(|pair| EnvSegment {
                target: pair[0][0],
                duration: pair[1][0] as usize,
            })
            .collect::<Vec<_>>();

        if trig > 0.0 && self.prev_trig <= 0.0 {
            self.start();
        }

        self.prev_trig = trig;

        if !self.active {
            return [0.0, 0.0];
        }

        let segment = &segments[self.current_idx];
        self.value = self.interpolate_segment(&segments);
        self.time += 1;

        if self.time >= segment.duration {
            self.advance(&segments);
        }

        [self.value, self.value]
    }
}

#[derive(Clone)]
pub struct SeqNode {
    pub(crate) step: usize,
}

impl SeqNode {
    pub(crate) fn new() -> Self {
        Self { step: 0 }
    }
}

impl Node for SeqNode {
    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let ramp = inputs.last().expect("seq: missing trigger input");
        let values = &inputs[0..inputs.len() - 1];

        let segment = 1.0 / values.len() as f32;
        self.step = (ramp[0] / segment).floor() as usize;

        values[self.step]
    }
}

pub struct PulseNode {
    phase: f32,
    prev_phase: f32,
}

impl Clone for PulseNode {
    fn clone(&self) -> Self {
        Self {
            phase: 0.0,
            prev_phase: 0.0,
        }
    }
}

impl PulseNode {
    pub(crate) fn new() -> Self {
        Self {
            phase: 0.,
            prev_phase: 0.,
        }
    }
}

impl Node for PulseNode {
    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let freq = inputs.get(0).expect("pulse: missing freq input")[0];

        self.prev_phase = self.phase;
        self.phase += 2. * PI * freq / SAMPLE_RATE;

        if self.phase >= 2. * PI {
            self.phase -= 2. * PI;
            [1.0, 1.0]
        } else {
            [0.0, 0.0]
        }
    }
}

const BUFFER_SIZE: usize = 8192;

// this is the number of samples we need to represent a full period
// of the lowest possible MIDI pitch's frequency (A0 / 27.50 Hz)
enum Mode {
    String,
    Drum,
}

#[derive(Clone)]
pub struct PluckNode {
    // mode: Mode,
    // tone: f32,
    // damping: f32,
    buffer: [f32; BUFFER_SIZE],
    period: f32,
    read_pos: usize,
    pitch_track: f32,
    is_stopped: bool,
}

impl PluckNode {
    pub(crate) fn new() -> Self {
        Self {
            // mode: Mode::String,
            buffer: [0.0; BUFFER_SIZE],
            period: 1.0,
            read_pos: 0,
            pitch_track: 0.0,
            is_stopped: true,
        }
    }

    fn play(&mut self, freq: f32, tone: f32) {
        self.is_stopped = false;
        self.period = freq_to_period(SAMPLE_RATE, freq);
        self.read_pos = 0;

        self.pitch_track = (5.0 as f32).max(self.period / 7.0);
        assert!(self.period < BUFFER_SIZE as f32);

        for i in 0..self.period as usize {
            if i > self.period as usize {
                self.buffer[i] = 0.0;
            }
            // generate one period of a sine wave
            // let sine = ((i as f32 / self.period) * (PI * 2.0)).sin();
            let tri = Self::generate_triangle_wave(i as i32, self.period);

            let noise = if rand::thread_rng().gen::<bool>() {
                1.0
            } else {
                -1.0
            };
            let y = (tri * tone) + (noise * (1.0 - tone));
            self.buffer[i] = y;
        }
    }

    fn generate_triangle_wave(sample: i32, period: f32) -> f32 {
        let phase = sample as f32 / period;
        if phase < 0.25 {
            4.0 * phase
        } else if phase < 0.75 {
            2.0 - 4.0 * phase
        } else {
            -4.0 + 4.0 * phase
        }
    }
}

impl Node for PluckNode {
    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let freq = inputs.get(0).map_or(0.0, |arr| arr[0]);
        let tone = inputs.get(1).map_or(0.0, |arr| arr[0]);
        let damping = inputs.get(2).map_or(0.0, |arr| arr[0]);
        let trig = inputs.get(3).map_or(0.0, |arr| arr[0]);

        if trig > 0.0 {
            self.play(freq, tone);
        }

        // if !self.is_active() {
        //     return 0.0;
        // }
        // increment read position
        // TODO: is it a problem that we're rounding here?
        // should we interpolate between buffer values?
        self.read_pos = (self.read_pos + 1) % self.period as usize;

        // smooth signal using simple averaging
        // try more advanced filter
        let mut sum = 0.0;
        // let window = 10.0;
        let mut window = damping.powf(2.0);
        window = (2.0 as f32).max(window * self.pitch_track);
        for i in 0..window as usize {
            let idx = (self.read_pos + i) % self.period as usize;
            sum += self.buffer[idx];
        }
        self.buffer[self.read_pos] = sum * (1.0 / window);

        if self.is_stopped {
            // fade out note
            self.buffer[self.read_pos] *= 0.9;
        }

        let output = self.buffer[self.read_pos];
        [output, output]
    }
}

#[derive(Clone)]
pub struct DelayNode {
    delay: Delay,
}

impl DelayNode {
    pub(crate) fn new() -> Self {
        Self {
            delay: Delay::new(15000.0, 0.5),
        }
    }
}

impl Node for DelayNode {
    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let input = inputs.get(0).expect("delay: missing input")[0];
        let output = self.delay.process(input);
        [output, output]
    }
}
