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

#[derive(Clone, KotoType, KotoCopy)]
pub enum NodeKind {
    Constant(f32),
    Sine {
        freq: Box<NodeKind>,
        node: Box<dyn AudioUnit>,
    },
    Square {
        freq: Box<NodeKind>,
        node: Box<dyn AudioUnit>,
    },
    Saw {
        freq: Box<NodeKind>,
        node: Box<dyn AudioUnit>,
    },
    Triangle {
        freq: Box<NodeKind>,
        node: Box<dyn AudioUnit>,
    },
    Pulse {
        freq: Box<NodeKind>,
        node: PulseNode,
    },
    Noise(Box<dyn AudioUnit>),
    Mix(Box<NodeKind>, Box<NodeKind>),
    Gain(Box<NodeKind>, Box<NodeKind>),
    Env {
        trig: Box<NodeKind>,
        segments: Vec<(NodeKind, NodeKind)>,
        node: EnvNode,
    },
    SVF {
        input: Box<NodeKind>,
        cutoff: Box<NodeKind>,
        resonance: Box<NodeKind>,
        node: Box<dyn AudioUnit>,
    },
    Pan {
        input: Box<NodeKind>,
        value: Box<NodeKind>,
        node: Box<dyn AudioUnit>,
    },
    Seq {
        trig: Box<NodeKind>,
        values: Vec<NodeKind>,
        node: SeqNode,
    },
    Pluck {
        freq: Box<NodeKind>,
        tone: Box<NodeKind>,
        damping: Box<NodeKind>,
        trig: Box<NodeKind>,
        node: PluckNode,
    },
    Reverb {
        input: Box<NodeKind>,
        node: Box<dyn AudioUnit>,
    },
    Delay {
        input: Box<NodeKind>,
        node: DelayNode,
    },
    Moog {
        input: Box<NodeKind>,
        cutoff: Box<NodeKind>,
        resonance: Box<NodeKind>,
        node: Box<dyn AudioUnit>,
    },
}

impl KotoObject for NodeKind {
    // TODO: test these
    fn add(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(lhs), KValue::Number(rhs)) => Ok(KValue::Object(
                NodeKind::Constant(lhs + f32::from(rhs)).into(),
            )),
            (_, KValue::Number(num)) => Ok(KValue::Object(
                NodeKind::Mix(
                    Box::new(self.clone()),
                    Box::new(NodeKind::Constant((num).into())),
                )
                .into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                NodeKind::Mix(
                    Box::new(self.clone()),
                    Box::new(obj.cast::<NodeKind>()?.clone()).into(),
                )
                .into(),
            )),
            _ => panic!("invalid add operation"),
        }
    }

    fn multiply(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(lhs), KValue::Number(rhs)) => Ok(KValue::Object(
                NodeKind::Constant(lhs * f32::from(rhs)).into(),
            )),
            (_, KValue::Number(num)) => Ok(KValue::Object(
                NodeKind::Gain(
                    Box::new(self.clone()),
                    Box::new(NodeKind::Constant(num.into())),
                )
                .into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                NodeKind::Gain(
                    Box::new(self.clone()),
                    Box::new(obj.cast::<NodeKind>()?.clone()).into(),
                )
                .into(),
            )),
            _ => panic!("invalid multiply operation"),
        }
    }

    fn subtract(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(lhs), KValue::Number(rhs)) => Ok(KValue::Object(
                NodeKind::Constant(lhs - f32::from(-rhs)).into(),
            )),
            (_, KValue::Number(num)) => Ok(KValue::Object(
                NodeKind::Mix(
                    Box::new(self.clone()),
                    Box::new(NodeKind::Constant((-num).into())),
                )
                .into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                NodeKind::Mix(
                    Box::new(self.clone()),
                    Box::new(obj.cast::<NodeKind>()?.clone()).into(),
                )
                .into(),
            )),
            _ => panic!("invalid subtract operation"),
        }
    }

    fn divide(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(lhs), KValue::Number(rhs)) => Ok(KValue::Object(
                NodeKind::Constant(lhs / f32::from(rhs)).into(),
            )),
            (_, KValue::Number(num)) => Ok(KValue::Object(
                NodeKind::Gain(
                    Box::new(self.clone()),
                    Box::new(NodeKind::Constant(1.0 / f32::from(num))),
                )
                .into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                NodeKind::Gain(
                    Box::new(self.clone()),
                    Box::new(obj.cast::<NodeKind>()?.clone()).into(),
                )
                .into(),
            )),
            _ => panic!("invalid divide operation"),
        }
    }
}

// necessary to satisfy KotoEntries trait
impl KotoEntries for NodeKind {
    fn entries(&self) -> Option<KMap> {
        None
    }
}

impl Node for NodeKind {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        match self {
            NodeKind::Constant(val) => [*val, *val],
            NodeKind::Sine { node, .. }
            | NodeKind::Square { node, .. }
            | NodeKind::Saw { node, .. }
            | NodeKind::Triangle { node, .. }
            | NodeKind::SVF { node, .. }
            | NodeKind::Moog { node, .. }
            | NodeKind::Noise(node) => {
                let input: Vec<f32> = inputs.iter().map(|[l, _]| *l).collect();
                let mut output = [0.0];
                node.tick(input.as_slice(), &mut output);
                [output[0], output[0]]
            }

            NodeKind::Reverb { node, .. } => {
                let mut output = [0.0; 2];
                node.tick(inputs[0].as_slice(), &mut output);
                output
            }

            NodeKind::Pan { node, .. } => {
                let input: Vec<f32> = inputs.iter().map(|[left, _]| *left).collect();
                let mut output = [0.0; 2];
                node.tick(input.as_slice(), &mut output);
                output
            }

            NodeKind::Pulse { node, .. } => node.tick(inputs),
            NodeKind::Seq { node, .. } => node.tick(inputs),
            NodeKind::Pluck { node, .. } => node.tick(inputs),
            NodeKind::Delay { node, .. } => node.tick(inputs),
            NodeKind::Env { node, .. } => node.tick(inputs),

            NodeKind::Gain { .. } => {
                if let [[l0, r0], [l1, r1]] = inputs {
                    [l0 * l1, r0 * r1]
                } else {
                    panic!("Wrong input format");
                }
            }
            NodeKind::Mix { .. } => {
                if let [[l0, r0], [l1, r1]] = inputs {
                    [l0 + l1, r0 + r1]
                } else {
                    panic!("Wrong input format");
                }
            }
        }
    }
}

impl PartialEq for NodeKind {
    fn eq(&self, other: &Self) -> bool {
        macro_rules! simple_eq {
            ($($variant:ident),* $(,)?) => {
                match (self, other) {
                    $( (Self::$variant { .. }, Self::$variant { .. }) => true, )*
                    (Self::Constant(lhs), Self::Constant(rhs)) => lhs == rhs,
                    (Self::Pluck { node: lhs, .. }, Self::Pluck { node: rhs, .. }) => lhs.buffer == rhs.buffer,
                    _ => false,
                }
            }
        }

        simple_eq!(
            Sine, Square, Saw, Pulse, Noise, Gain, Mix, Env, SVF, Reverb, Delay, Triangle, Moog,
        )
    }
}
macro_rules! transfer_node_state {
    ( $self:ident, $other:ident; $( $variant:ident ),* $(,)? ) => {
        match ($self, $other) {
            $(
                (NodeKind::$variant { node: new, .. }, NodeKind::$variant { node: old, .. }) => {
                    *new = old.clone();
                }
            )*
            _ => {}
        }
    };
}

impl NodeKind {
    pub(crate) fn compute_hash(&self) -> u64 {
        let mut hasher = SeaHasher::new();
        self.hash_structure(&mut hasher);
        hasher.finish()
    }

    fn hash_structure(&self, hasher: &mut SeaHasher) {
        macro_rules! hash_node {
        ($tag:expr $(, $field:expr)* $(,)?) => {{
            ($tag as u8).hash(hasher);
            $( $field.hash_structure(hasher); )*
        }};
    }

        match self {
            NodeKind::Constant(val) => {
                0u8.hash(hasher);
                val.to_bits().hash(hasher);
            }
            NodeKind::Sine { freq, .. } => hash_node!(1, freq),
            NodeKind::Square { freq, .. } => hash_node!(2, freq),
            NodeKind::Saw { freq, .. } => hash_node!(3, freq),
            NodeKind::Pulse { freq, .. } => hash_node!(4, freq),
            NodeKind::Noise(_) => hash_node!(5),
            NodeKind::Mix(lhs, rhs) => hash_node!(6, lhs, rhs),
            NodeKind::Gain(lhs, rhs) => hash_node!(7, lhs, rhs),
            NodeKind::Env { segments, trig, .. } => {
                hash_node!(8, trig);
                for (val, dur) in segments {
                    val.hash_structure(hasher);
                    dur.hash_structure(hasher);
                }
            }
            NodeKind::SVF {
                input,
                cutoff,
                resonance,
                ..
            } => hash_node!(10, input, cutoff, resonance),
            NodeKind::Seq { trig, values, .. } => {
                hash_node!(12, trig);
                for val in values {
                    val.hash_structure(hasher);
                }
            }
            NodeKind::Pan { input, value, .. } => {
                hash_node!(18, input, value);
            }
            NodeKind::Pluck {
                freq,
                tone,
                damping,
                trig,
                ..
            } => hash_node!(13, freq, tone, damping, trig),
            NodeKind::Reverb { input, .. } => hash_node!(14, input),
            NodeKind::Delay { input, .. } => hash_node!(15, input),
            NodeKind::Triangle { freq, .. } => hash_node!(16, freq),
            NodeKind::Moog {
                input,
                cutoff,
                resonance,
                ..
            } => hash_node!(17, input, cutoff, resonance),
        }
    }

    pub(crate) fn transfer_state_from(&mut self, other: &NodeKind) {
        transfer_node_state!(self, other;
            Sine,
            Square,
            Saw,
            Pulse,
            Triangle,
            Env,
            SVF,
            Seq,
            Pluck,
            Reverb,
            Delay,
            Moog
        );
    }
}

pub(crate) trait Node: Send + Sync {
    fn tick(&mut self, inputs: &[Frame]) -> Frame;
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

    fn increment(&mut self, values: &[Frame]) {
        self.step += 1;
        if self.step >= values.len() {
            self.step = 0;
        }
    }
}

impl Node for SeqNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let trig = inputs.last().expect("seq: missing trigger input");
        let values = &inputs[0..inputs.len() - 1];
        if (*trig)[0] > 0.0 {
            self.increment(values);
        }

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
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let input = inputs.get(0).expect("delay: missing input")[0];
        let output = self.delay.process(input);
        [output, output]
    }
}
