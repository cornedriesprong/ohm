use crate::audio_graph::BoxedNode;
use crate::consts::SAMPLE_RATE;
use crate::dsp::delay::Delay;
use crate::utils::freq_to_period;
use core::fmt;
use fmt::{Debug, Formatter};
use fundsp::hacker32::AudioUnit;
use koto::{derive::*, prelude::*, runtime::Result};
use rand::Rng;
use rtsan_standalone::nonblocking;
use seahash::SeaHasher;
use std::f32::consts::PI;
use std::hash::{Hash, Hasher};
use strum::AsRefStr;

#[derive(Clone, KotoType, KotoCopy, AsRefStr)]
pub(crate) enum NodeKind {
    Constant(f32),
    Sine {
        freq: BoxedNode,
        node: Box<dyn AudioUnit>,
    },
    Square {
        freq: BoxedNode,
        node: Box<dyn AudioUnit>,
    },
    Saw {
        freq: BoxedNode,
        node: Box<dyn AudioUnit>,
    },
    Triangle {
        freq: BoxedNode,
        node: Box<dyn AudioUnit>,
    },
    Pulse {
        freq: BoxedNode,
        node: PulseNode,
    },
    Noise(Box<dyn AudioUnit>),
    Mix(BoxedNode, BoxedNode),
    Gain(BoxedNode, BoxedNode),
    Env {
        trig: BoxedNode,
        segments: Vec<(NodeKind, NodeKind)>,
        node: EnvNode,
    },
    Lowpass {
        input: BoxedNode,
        cutoff: BoxedNode,
        resonance: BoxedNode,
        node: Box<dyn AudioUnit>,
    },
    Bandpass {
        input: BoxedNode,
        cutoff: BoxedNode,
        resonance: BoxedNode,
        node: Box<dyn AudioUnit>,
    },
    Highpass {
        input: BoxedNode,
        cutoff: BoxedNode,
        resonance: BoxedNode,
        node: Box<dyn AudioUnit>,
    },
    Seq {
        trig: BoxedNode,
        values: Vec<NodeKind>,
        node: SeqNode,
    },
    Pluck {
        freq: BoxedNode,
        tone: BoxedNode,
        damping: BoxedNode,
        trig: BoxedNode,
        node: PluckNode,
    },
    Reverb {
        input: BoxedNode,
        node: Box<dyn AudioUnit>,
    },
    Delay {
        input: BoxedNode,
        node: DelayNode,
    },
    Moog {
        input: BoxedNode,
        cutoff: BoxedNode,
        resonance: BoxedNode,
        node: Box<dyn AudioUnit>,
    },
}

impl KotoObject for NodeKind {
    // TODO: test these
    fn add(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(lhs), KValue::Number(rhs)) => {
                Ok(KValue::Object(constant(lhs + f32::from(rhs)).into()))
            }
            (_, KValue::Number(num)) => Ok(KValue::Object(
                mix(self.clone(), constant((num).into())).into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                mix(self.clone(), obj.cast::<NodeKind>()?.clone()).into(),
            )),
            _ => panic!("invalid add operation"),
        }
    }

    fn multiply(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(lhs), KValue::Number(rhs)) => {
                Ok(KValue::Object(constant(lhs * f32::from(rhs)).into()))
            }
            (_, KValue::Number(num)) => Ok(KValue::Object(
                gain(self.clone(), constant(num.into())).into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                gain(self.clone(), obj.cast::<NodeKind>()?.clone()).into(),
            )),
            _ => panic!("invalid multiply operation"),
        }
    }

    fn subtract(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(lhs), KValue::Number(rhs)) => {
                Ok(KValue::Object(constant(lhs - f32::from(-rhs)).into()))
            }
            (_, KValue::Number(num)) => Ok(KValue::Object(
                mix(self.clone(), constant((-num).into())).into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                mix(self.clone(), obj.cast::<NodeKind>()?.clone()).into(),
            )),
            _ => panic!("invalid subtract operation"),
        }
    }

    fn divide(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(lhs), KValue::Number(rhs)) => {
                Ok(KValue::Object(constant(lhs / f32::from(-rhs)).into()))
            }
            (_, KValue::Number(num)) => Ok(KValue::Object(
                gain(self.clone(), constant(num.into())).into(),
            )),
            (_, KValue::Object(obj)) => Ok(KValue::Object(
                gain(self.clone(), obj.cast::<NodeKind>()?.clone()).into(),
            )),
            _ => panic!("invalid divide operation"),
        }
    }
}

impl KotoEntries for NodeKind {
    fn entries(&self) -> Option<KMap> {
        None
    }
}

pub(crate) fn constant(value: f32) -> NodeKind {
    NodeKind::Constant(value)
}

pub(crate) fn sine(freq: NodeKind) -> NodeKind {
    use fundsp::hacker32::sine;
    NodeKind::Sine {
        freq: Box::new(freq),
        node: Box::new(sine()),
    }
}

pub(crate) fn square(freq: NodeKind) -> NodeKind {
    use fundsp::hacker32::square;
    NodeKind::Square {
        freq: Box::new(freq),
        node: Box::new(square()),
    }
}

pub(crate) fn saw(freq: NodeKind) -> NodeKind {
    use fundsp::hacker32::saw;
    NodeKind::Saw {
        freq: Box::new(freq),
        node: Box::new(saw()),
    }
}

pub(crate) fn triangle(freq: NodeKind) -> NodeKind {
    use fundsp::hacker32::triangle;
    NodeKind::Triangle {
        freq: Box::new(freq),
        node: Box::new(triangle()),
    }
}

pub(crate) fn noise() -> NodeKind {
    use fundsp::hacker32::noise;
    NodeKind::Noise(Box::new(noise()))
}

pub(crate) fn pulse(freq: NodeKind) -> NodeKind {
    NodeKind::Pulse {
        freq: Box::new(freq),
        node: PulseNode::new(),
    }
}

pub(crate) fn gain(lhs: NodeKind, rhs: NodeKind) -> NodeKind {
    NodeKind::Gain(Box::new(lhs), Box::new(rhs))
}

pub(crate) fn mix(lhs: NodeKind, rhs: NodeKind) -> NodeKind {
    NodeKind::Mix(Box::new(lhs), Box::new(rhs))
}

pub(crate) fn env(segments: Vec<(NodeKind, NodeKind)>, trig: NodeKind) -> NodeKind {
    NodeKind::Env {
        trig: Box::new(trig),
        segments,
        node: EnvNode::new(),
    }
}

pub(crate) fn lowpass(input: NodeKind, cutoff: NodeKind, resonance: NodeKind) -> NodeKind {
    use fundsp::hacker32::lowpass;
    NodeKind::Lowpass {
        input: Box::new(input),
        cutoff: Box::new(cutoff),
        resonance: Box::new(resonance),
        node: Box::new(lowpass()),
    }
}

pub(crate) fn bandpass(input: NodeKind, cutoff: NodeKind, resonance: NodeKind) -> NodeKind {
    use fundsp::hacker32::bandpass;
    NodeKind::Bandpass {
        input: Box::new(input),
        cutoff: Box::new(cutoff),
        resonance: Box::new(resonance),
        node: Box::new(bandpass()),
    }
}

pub(crate) fn highpass(input: NodeKind, cutoff: NodeKind, resonance: NodeKind) -> NodeKind {
    use fundsp::hacker32::highpass;
    NodeKind::Highpass {
        input: Box::new(input),
        cutoff: Box::new(cutoff),
        resonance: Box::new(resonance),
        node: Box::new(highpass()),
    }
}

pub(crate) fn seq(values: Vec<NodeKind>, trig: NodeKind) -> NodeKind {
    NodeKind::Seq {
        trig: Box::new(trig),
        values,
        node: SeqNode::new(),
    }
}

pub(crate) fn pluck(freq: NodeKind, tone: NodeKind, damping: NodeKind, trig: NodeKind) -> NodeKind {
    NodeKind::Pluck {
        freq: Box::new(freq),
        tone: Box::new(tone),
        damping: Box::new(damping),
        trig: Box::new(trig),
        node: PluckNode::new(),
    }
}

pub(crate) fn reverb(input: NodeKind) -> NodeKind {
    use fundsp::hacker32::{delay, fdn, fir, join, lerp, rnd1, split, stacki, U16};
    NodeKind::Reverb {
        input: Box::new(input),
        node: Box::new(
            split()
                >> fdn::<U16, _>(stacki::<U16, _, _>(|i| {
                    delay(lerp(0.01, 0.5, rnd1(i) as f32)) >> fir((0.2, 0.4, 0.2))
                }))
                >> join(),
        ),
    }
}

pub(crate) fn delay(input: NodeKind) -> NodeKind {
    NodeKind::Delay {
        input: Box::new(input),
        node: DelayNode::new(),
    }
}

pub(crate) fn moog(input: NodeKind, cutoff: NodeKind, resonance: NodeKind) -> NodeKind {
    use fundsp::hacker32::moog;
    NodeKind::Moog {
        cutoff: Box::new(cutoff),
        resonance: Box::new(resonance),
        input: Box::new(input),
        node: Box::new(moog()),
    }
}

impl Node for NodeKind {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        match self {
            NodeKind::Constant(val) => *val,
            NodeKind::Sine { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
            NodeKind::Square { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
            NodeKind::Saw { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
            NodeKind::Triangle { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
            NodeKind::Noise(node) => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
            NodeKind::Pulse { node, .. } => node.tick(inputs),
            NodeKind::Gain { .. } => inputs[0] * inputs[1],
            NodeKind::Mix { .. } => inputs[0] + inputs[1],
            NodeKind::Env { node, .. } => node.tick(inputs),
            NodeKind::Lowpass { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
            NodeKind::Bandpass { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
            NodeKind::Highpass { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
            NodeKind::Seq { node, .. } => node.tick(inputs),
            NodeKind::Pluck { node, .. } => node.tick(inputs),
            NodeKind::Reverb { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
            NodeKind::Delay { node, .. } => node.tick(inputs),
            NodeKind::Moog { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
        }
    }
}

impl PartialEq for NodeKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Constant(lhs), Self::Constant(rhs)) => lhs == rhs,
            (Self::Sine { .. }, Self::Sine { .. }) => true,
            (Self::Square { .. }, Self::Square { .. }) => true,
            (Self::Saw { .. }, Self::Saw { .. }) => true,
            (Self::Pulse { .. }, Self::Pulse { .. }) => true,
            (Self::Noise(_), Self::Noise(_)) => true,
            (Self::Gain { .. }, Self::Gain { .. }) => true,
            (Self::Mix { .. }, Self::Mix { .. }) => true,
            (Self::Env { .. }, Self::Env { .. }) => true,
            (Self::Lowpass { .. }, Self::Lowpass { .. }) => true,
            (Self::Bandpass { .. }, Self::Bandpass { .. }) => true,
            (Self::Highpass { .. }, Self::Highpass { .. }) => true,
            (Self::Seq { node: lhs, .. }, Self::Seq { node: rhs, .. }) => lhs.step == rhs.step,
            (Self::Pluck { node: lhs, .. }, Self::Pluck { node: rhs, .. }) => {
                lhs.buffer == rhs.buffer
            }
            (Self::Reverb { .. }, Self::Reverb { .. }) => true,
            (Self::Delay { .. }, Self::Delay { .. }) => true,
            (Self::Triangle { .. }, Self::Triangle { .. }) => true,
            (Self::Moog { .. }, Self::Moog { .. }) => true,
            _ => false,
            // _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
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
            // Handle tuple variants separately
            (NodeKind::Noise(new), NodeKind::Noise(old)) => {
                *new = old.clone();
            }
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
        match self {
            NodeKind::Constant(val) => {
                0u8.hash(hasher);
                val.to_bits().hash(hasher);
            }
            NodeKind::Sine { freq, .. } => {
                1u8.hash(hasher);
                freq.hash_structure(hasher);
            }
            NodeKind::Square { freq, .. } => {
                2u8.hash(hasher);
                freq.hash_structure(hasher);
            }
            NodeKind::Saw { freq, .. } => {
                3u8.hash(hasher);
                freq.hash_structure(hasher);
            }
            NodeKind::Pulse { freq, .. } => {
                4u8.hash(hasher);
                freq.hash_structure(hasher);
            }
            NodeKind::Noise(_) => {
                5u8.hash(hasher);
            }
            NodeKind::Mix(lhs, rhs) => {
                6u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            NodeKind::Gain(lhs, rhs) => {
                7u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            NodeKind::Env { segments, trig, .. } => {
                8u8.hash(hasher);
                trig.hash_structure(hasher);
                for segment in segments {
                    segment.0.hash_structure(hasher);
                    segment.1.hash_structure(hasher);
                }
            }
            NodeKind::Lowpass {
                input,
                cutoff,
                resonance,
                ..
            } => {
                9u8.hash(hasher);
                input.hash_structure(hasher);
                cutoff.hash_structure(hasher);
                resonance.hash_structure(hasher);
            }
            NodeKind::Bandpass {
                input,
                cutoff,
                resonance,
                ..
            } => {
                9u8.hash(hasher);
                input.hash_structure(hasher);
                cutoff.hash_structure(hasher);
                resonance.hash_structure(hasher);
            }
            NodeKind::Highpass {
                input,
                cutoff,
                resonance,
                ..
            } => {
                9u8.hash(hasher);
                input.hash_structure(hasher);
                cutoff.hash_structure(hasher);
                resonance.hash_structure(hasher);
            }
            NodeKind::Seq { trig, values, .. } => {
                10u8.hash(hasher);
                trig.hash_structure(hasher);
                for val in values {
                    val.hash_structure(hasher);
                    val.hash_structure(hasher);
                }
            }
            NodeKind::Pluck {
                freq,
                tone,
                damping,
                trig,
                ..
            } => {
                12u8.hash(hasher);
                freq.hash_structure(hasher);
                tone.hash_structure(hasher);
                damping.hash_structure(hasher);
                trig.hash_structure(hasher);
            }
            NodeKind::Reverb { input, .. } => {
                13u8.hash(hasher);
                input.hash_structure(hasher);
            }
            NodeKind::Delay { input, .. } => {
                14u8.hash(hasher);
                input.hash_structure(hasher);
            }
            NodeKind::Triangle { freq, .. } => {
                15u8.hash(hasher);
                freq.hash_structure(hasher);
            }
            NodeKind::Moog {
                input,
                cutoff,
                resonance,
                ..
            } => {
                16u8.hash(hasher);
                input.hash_structure(hasher);
                cutoff.hash_structure(hasher);
                resonance.hash_structure(hasher);
            }
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
            Lowpass,
            Bandpass,
            Highpass,
            Seq,
            Pluck,
            Reverb,
            Delay,
            Moog
        );
    }
}

impl Debug for NodeKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            NodeKind::Constant(val) => write!(f, "{}({})", self.as_ref(), val),
            _ => write!(f, "{}", self.as_ref()),
        }
    }
}

pub(crate) trait Node: Send + Sync {
    fn tick(&mut self, inputs: &[f32]) -> f32;
}

#[derive(Clone, Debug)]
pub struct EnvSegment {
    pub duration: usize, // Duration in samples or ms (interpreted as needed)
    pub target: f32,     // Target value at end of this segment
}

#[derive(Clone)]
pub(crate) struct EnvNode {
    pub(crate) current_idx: usize,
    pub(crate) value: f32,
    pub(crate) time: usize,
    pub(crate) active: bool,
    pub(crate) prev_trig: f32,
}

impl EnvNode {
    fn new() -> Self {
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
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let trig = *inputs.last().expect("env: missing trigger input");
        let segments = &inputs[0..inputs.len() - 1]
            .chunks_exact(2)
            .map(|pair| EnvSegment {
                target: pair[0],
                duration: pair[1] as usize,
            })
            .collect::<Vec<_>>();

        if trig > 0.0 && self.prev_trig <= 0.0 {
            self.start();
        }

        self.prev_trig = trig;

        if !self.active {
            return 0.0;
        }

        let segment = &segments[self.current_idx];
        self.value = self.interpolate_segment(&segments);
        self.time += 1;

        if self.time >= segment.duration {
            self.advance(&segments);
        }

        self.value
    }
}

#[derive(Clone)]
pub(crate) struct SeqNode {
    pub(crate) step: usize,
}

impl SeqNode {
    fn new() -> Self {
        Self { step: 0 }
    }

    fn increment(&mut self, values: &[f32]) {
        self.step += 1;
        if self.step >= values.len() {
            self.step = 0;
        }
    }
}

impl Node for SeqNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let trig = inputs.last().expect("seq: missing trigger input");
        let values = &inputs[0..inputs.len() - 1];
        if *trig > 0.0 {
            self.increment(values);
        }

        values[self.step]
    }
}

pub(crate) struct PulseNode {
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
    fn new() -> Self {
        Self {
            phase: 0.,
            prev_phase: 0.,
        }
    }
}

impl Node for PulseNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let freq = inputs.get(0).expect("pulse: missing freq input");
        let reset_phase = inputs.get(1).unwrap_or(&0.0);
        if *reset_phase > 0.0 {
            self.phase = 0.0;
        }

        self.prev_phase = self.phase;
        self.phase += 2. * PI * freq / SAMPLE_RATE;

        if self.phase >= 2. * PI {
            self.phase -= 2. * PI;
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Clone)]
pub(crate) struct GainNode {}

impl GainNode {
    fn new() -> Self {
        Self {}
    }
}

impl Node for GainNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let lhs = inputs.get(0).expect("gain: missing lhs input");
        let rhs = inputs.get(1).expect("gain: missing rhs input");
        lhs * rhs
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
    fn new() -> Self {
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
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let freq = inputs.get(0).expect("pluck: missing freq input");
        let tone = inputs.get(1).expect("pluck: missing tone input");
        let damping = inputs.get(2).expect("pluck: missing damping input");
        let trig = inputs.get(3).expect("pluck: missing trigger input");

        if *trig > 0.0 {
            self.play(*freq, *tone);
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

        self.buffer[self.read_pos]
    }
}

#[derive(Clone)]
pub struct DelayNode {
    delay: Delay,
}

impl DelayNode {
    fn new() -> Self {
        Self {
            delay: Delay::new(15000.0, 0.5),
        }
    }
}

impl Node for DelayNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let input = inputs.get(0).expect("delay: missing input");
        self.delay.process(*input)
    }
}
