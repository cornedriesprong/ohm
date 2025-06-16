use crate::audio_graph::BoxedNode;
use crate::consts::SAMPLE_RATE;
use crate::dsp::delay::Delay;
use crate::utils::{freq_to_period, lerp};
use core::fmt;
use fmt::{Debug, Formatter};
use fundsp::hacker32::AudioUnit;
use koto::{derive::*, prelude::*, runtime::Result};
use rand::Rng;
use rtsan_standalone::nonblocking;
use seahash::SeaHasher;
use std::hash::{Hash, Hasher};
use strum::AsRefStr;

#[derive(Clone, KotoType, KotoCopy, AsRefStr)]
pub(crate) enum NodeKind {
    Constant {
        value: f32,
        node: Box<dyn AudioUnit>,
    },
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
        duty: BoxedNode,
        node: Box<dyn AudioUnit>,
    },
    Noise(Box<dyn AudioUnit>),
    Mix {
        lhs: BoxedNode,
        rhs: BoxedNode,
        node: MixNode,
    },
    Gain {
        lhs: BoxedNode,
        rhs: BoxedNode,
        node: GainNode,
    },
    AR {
        trig: BoxedNode,
        attack: BoxedNode,
        release: BoxedNode,
        node: ARNode,
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
            (Self::Constant { value, .. }, KValue::Number(num)) => {
                Ok(KValue::Object(constant(value + f32::from(num)).into()))
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
            (Self::Constant { value, .. }, KValue::Number(num)) => {
                Ok(KValue::Object(constant(value * f32::from(num)).into()))
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
            (Self::Constant { value, .. }, KValue::Number(num)) => {
                Ok(KValue::Object(constant(value - f32::from(-num)).into()))
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
            (Self::Constant { value, .. }, KValue::Number(num)) => {
                Ok(KValue::Object(constant(value / f32::from(-num)).into()))
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

const BUFFER_SIZE: usize = 8192;

pub(crate) fn constant(value: f32) -> NodeKind {
    use fundsp::hacker32::dc;
    NodeKind::Constant {
        value,
        node: Box::new(dc(value)),
    }
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
    use fundsp::hacker32::pulse;
    NodeKind::Pulse {
        freq: Box::new(freq),
        duty: Box::new(constant(0.1)),
        node: Box::new(pulse()),
    }
}

pub(crate) fn gain(lhs: NodeKind, rhs: NodeKind) -> NodeKind {
    NodeKind::Gain {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        node: GainNode::new(),
    }
}

pub(crate) fn mix(lhs: NodeKind, rhs: NodeKind) -> NodeKind {
    NodeKind::Mix {
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        node: MixNode::new(),
    }
}

pub(crate) fn ar(attack: NodeKind, release: NodeKind, trig: NodeKind) -> NodeKind {
    NodeKind::AR {
        trig: Box::new(trig),
        attack: Box::new(attack),
        release: Box::new(release),
        node: ARNode::new(),
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

pub(crate) fn seq(values: Vec<f32>, trig: NodeKind) -> NodeKind {
    NodeKind::Seq {
        trig: Box::new(trig),
        node: SeqNode::new(values),
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
            NodeKind::Constant { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
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
            NodeKind::Pulse { node, .. } => {
                let mut output = [0.0];
                node.tick(inputs.into(), &mut output);
                output[0]
            }
            NodeKind::Gain { node, .. } => node.tick(inputs),
            NodeKind::Mix { node, .. } => node.tick(inputs),
            NodeKind::AR { node, .. } => node.tick(inputs),
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
            (Self::Constant { value: a, .. }, Self::Constant { value: b, .. }) => a == b,
            (Self::Sine { .. }, Self::Sine { .. }) => true,
            (Self::Square { .. }, Self::Square { .. }) => true,
            (Self::Saw { .. }, Self::Saw { .. }) => true,
            (Self::Pulse { .. }, Self::Pulse { .. }) => true,
            (Self::Noise(_), Self::Noise(_)) => true,
            (Self::Gain { .. }, Self::Gain { .. }) => true,
            (Self::Mix { .. }, Self::Mix { .. }) => true,
            (Self::AR { .. }, Self::AR { .. }) => true,
            (Self::Lowpass { .. }, Self::Lowpass { .. }) => true,
            (Self::Bandpass { .. }, Self::Bandpass { .. }) => true,
            (Self::Highpass { .. }, Self::Highpass { .. }) => true,
            (Self::Seq { node: n1, .. }, Self::Seq { node: n2, .. }) => n1.values == n2.values,
            (Self::Pluck { node: n1, .. }, Self::Pluck { node: n2, .. }) => n1.buffer == n2.buffer,
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
            NodeKind::Constant { value, .. } => {
                0u8.hash(hasher);
                value.to_bits().hash(hasher);
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
            NodeKind::Mix { lhs, rhs, .. } => {
                6u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            NodeKind::Gain { lhs, rhs, .. } => {
                7u8.hash(hasher);
                lhs.hash_structure(hasher);
                rhs.hash_structure(hasher);
            }
            NodeKind::AR {
                attack,
                release,
                trig,
                ..
            } => {
                8u8.hash(hasher);
                attack.hash_structure(hasher);
                release.hash_structure(hasher);
                trig.hash_structure(hasher);
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
            NodeKind::Seq { trig, node } => {
                10u8.hash(hasher);
                trig.hash_structure(hasher);
                for val in &node.values {
                    val.to_bits().hash(hasher);
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
            Constant,
            Sine,
            Square,
            Saw,
            Pulse,
            Triangle,
            Gain,
            Mix,
            AR,
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
            NodeKind::Constant { value, .. } => write!(f, "{}({})", self.as_ref(), value),
            _ => write!(f, "{}", self.as_ref()),
        }
    }
}

pub(crate) trait Node: Send + Sync {
    fn tick(&mut self, inputs: &[f32]) -> f32;
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

#[derive(Clone)]
pub(crate) struct MixNode {}

impl MixNode {
    fn new() -> Self {
        Self {}
    }
}

impl Node for MixNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let lhs = inputs.get(0).expect("mix: missing lhs input");
        let rhs = inputs.get(1).expect("mix: missing rhs input");
        lhs + rhs
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EnvelopeState {
    Attack,
    Release,
    Off,
}

#[derive(Clone)]
pub(crate) struct ARNode {
    pub(crate) state: EnvelopeState,
    pub(crate) value: f32,
    pub(crate) time: f32,
}

impl ARNode {
    fn new() -> Self {
        Self {
            value: 0.0,
            state: EnvelopeState::Off,
            time: 0.0,
        }
    }

    fn get_curve(&self, length: f32) -> f32 {
        lerp(self.time, length)
    }

    fn get_curve_rev(&self, length: f32) -> f32 {
        lerp(length - self.time, length)
    }
}

impl Node for ARNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let attack = inputs.get(0).expect("missing attack input");
        let release = inputs.get(1).expect("missing release input");
        let trig = inputs.get(2).expect("missing trigger input");

        if *trig > 0.0 {
            self.state = EnvelopeState::Attack;
        }

        use EnvelopeState as E;
        match self.state {
            E::Attack => {
                if *attack == 0.0 {
                    self.value = 1.0;
                } else {
                    self.value = self.get_curve(*attack);
                }

                if self.value >= 1.0 {
                    self.value = 1.0;
                    self.time = 0.0;
                    self.state = E::Release;
                }
            }
            E::Release => {
                self.value = self.get_curve_rev(*release);
                if self.value <= 0.0 {
                    self.value = 0.0;
                    self.time = 0.0;
                    self.state = E::Off;
                }
            }
            E::Off => {
                self.time = 0.0;
            }
        }
        self.time += 1.0;

        self.value
    }
}

#[derive(Clone)]
pub(crate) struct SeqNode {
    pub(crate) values: Vec<f32>,
    pub(crate) step: usize,
}

impl SeqNode {
    fn new(values: Vec<f32>) -> Self {
        Self { values, step: 0 }
    }

    fn increment(&mut self) {
        self.step += 1;
        if self.step >= self.values.len() {
            self.step = 0;
        }
    }
}

impl Node for SeqNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let trig = inputs.get(0).expect("seq: missing trigger input");
        if *trig > 0.0 {
            self.increment();
        }

        self.values[self.step]
    }
}

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
