use crate::audio_graph::BoxedNode;
use crate::consts::SAMPLE_RATE;
use crate::dsp::delay::Delay;
use crate::dsp::reverb::Reverb;
use crate::utils::{freq_to_period, lerp};
use core::fmt;
use fmt::{Debug, Formatter};
use koto::{derive::*, prelude::*, runtime::Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rtsan_standalone::nonblocking;
use std::f32::consts::{FRAC_PI_4, PI};
use std::sync::{Arc, Mutex};
use strum::AsRefStr;

#[derive(Clone, KotoType, KotoCopy, AsRefStr)]
pub(crate) enum NodeKind {
    Constant(ConstantNode),
    Sine {
        freq: BoxedNode,
        node: SineNode,
    },
    Square {
        freq: BoxedNode,
        node: SquareNode,
    },
    Saw {
        freq: BoxedNode,
        node: SawNode,
    },
    Pulse {
        freq: BoxedNode,
        node: PulseNode,
    },
    Noise(NoiseNode),
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
    SVF {
        mode: BoxedNode,
        cutoff: BoxedNode,
        resonance: BoxedNode,
        input: BoxedNode,
        node: SVFNode,
    },
    Seq {
        trig: BoxedNode,
        node: SeqNode,
    },
    Pipe {
        delay: BoxedNode,
        input: BoxedNode,
        node: PipeNode,
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
        node: ReverbNode,
    },
    Delay {
        input: BoxedNode,
        node: DelayNode,
    },
}

impl KotoObject for NodeKind {
    // TODO: test these
    fn add(&self, rhs: &KValue) -> Result<KValue> {
        match (self, rhs) {
            (Self::Constant(node), KValue::Number(num)) => {
                Ok(KValue::Object(constant(node.value + f32::from(num)).into()))
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
            (Self::Constant(node), KValue::Number(num)) => {
                Ok(KValue::Object(constant(node.value * f32::from(num)).into()))
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
            (Self::Constant(node), KValue::Number(num)) => Ok(KValue::Object(
                constant(node.value - f32::from(-num)).into(),
            )),
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
            (Self::Constant(node), KValue::Number(num)) => Ok(KValue::Object(
                constant(node.value / f32::from(-num)).into(),
            )),
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
    NodeKind::Constant(ConstantNode::new(value))
}

pub(crate) fn sine(freq: NodeKind) -> NodeKind {
    NodeKind::Sine {
        freq: Box::new(freq),
        node: SineNode::new(),
    }
}

pub(crate) fn square(freq: NodeKind) -> NodeKind {
    NodeKind::Square {
        freq: Box::new(freq),
        node: SquareNode::new(),
    }
}

pub(crate) fn saw(freq: NodeKind) -> NodeKind {
    NodeKind::Saw {
        freq: Box::new(freq),
        node: SawNode::new(),
    }
}

pub(crate) fn noise() -> NodeKind {
    NodeKind::Noise(NoiseNode::new())
}

pub(crate) fn pulse(freq: NodeKind) -> NodeKind {
    NodeKind::Pulse {
        freq: Box::new(freq),
        node: PulseNode::new(),
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

pub(crate) fn svf(
    mode: NodeKind,
    cutoff: NodeKind,
    resonance: NodeKind,
    input: NodeKind,
) -> NodeKind {
    NodeKind::SVF {
        mode: Box::new(mode),
        cutoff: Box::new(cutoff),
        resonance: Box::new(resonance),
        input: Box::new(input),
        node: SVFNode::new(),
    }
}

pub(crate) fn seq(values: Vec<f32>, trig: NodeKind) -> NodeKind {
    NodeKind::Seq {
        trig: Box::new(trig),
        node: SeqNode::new(values),
    }
}

pub(crate) fn pipe(delay: NodeKind, input: NodeKind) -> NodeKind {
    NodeKind::Pipe {
        delay: Box::new(delay),
        input: Box::new(input),
        node: PipeNode::new(),
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
    NodeKind::Reverb {
        input: Box::new(input),
        node: ReverbNode::new(),
    }
}

pub(crate) fn delay(input: NodeKind) -> NodeKind {
    NodeKind::Delay {
        input: Box::new(input),
        node: DelayNode::new(),
    }
}

impl Node for NodeKind {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        match self {
            NodeKind::Constant(node) => node.tick(inputs),
            NodeKind::Sine { node, .. } => node.tick(inputs),
            NodeKind::Square { node, .. } => node.tick(inputs),
            NodeKind::Saw { node, .. } => node.tick(inputs),
            NodeKind::Noise(node) => node.tick(inputs),
            NodeKind::Pulse { node, .. } => node.tick(inputs),
            NodeKind::Gain { node, .. } => node.tick(inputs),
            NodeKind::Mix { node, .. } => node.tick(inputs),
            NodeKind::AR { node, .. } => node.tick(inputs),
            NodeKind::SVF { node, .. } => node.tick(inputs),
            NodeKind::Seq { node, .. } => node.tick(inputs),
            NodeKind::Pipe { node, .. } => node.tick(inputs),
            NodeKind::Pluck { node, .. } => node.tick(inputs),
            NodeKind::Reverb { node, .. } => node.tick(inputs),
            NodeKind::Delay { node, .. } => node.tick(inputs),
        }
    }
}

impl PartialEq for NodeKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Constant(a), Self::Constant(b)) => a.value == b.value,
            (Self::Sine { .. }, Self::Sine { .. }) => true,
            (Self::Square { .. }, Self::Square { .. }) => true,
            (Self::Saw { .. }, Self::Saw { .. }) => true,
            (Self::Pulse { .. }, Self::Pulse { .. }) => true,
            (Self::Noise(_), Self::Noise(_)) => true,
            (Self::Gain { .. }, Self::Gain { .. }) => true,
            (Self::Mix { .. }, Self::Mix { .. }) => true,
            (Self::AR { .. }, Self::AR { .. }) => true,
            (Self::SVF { .. }, Self::SVF { .. }) => true,
            (Self::Seq { node: n1, .. }, Self::Seq { node: n2, .. }) => n1.values == n2.values,
            (Self::Pipe { node: n1, .. }, Self::Pipe { node: n2, .. }) => n1.buffer == n2.buffer,
            (Self::Pluck { node: n1, .. }, Self::Pluck { node: n2, .. }) => n1.buffer == n2.buffer,
            (Self::Reverb { .. }, Self::Reverb { .. }) => true,
            (Self::Delay { .. }, Self::Delay { .. }) => true,
            _ => false,
            // _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl Debug for NodeKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            NodeKind::Constant(c) => write!(f, "{}({})", self.as_ref(), c.value),
            _ => write!(f, "{}", self.as_ref()),
        }
    }
}

pub(crate) trait Node: Send + Sync {
    fn tick(&mut self, inputs: &[f32]) -> f32;
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConstantNode {
    pub value: f32,
}

impl ConstantNode {
    fn new(value: f32) -> Self {
        Self { value }
    }
}

impl Node for ConstantNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, _: &[f32]) -> f32 {
        self.value
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SineNode {
    phase: f32,
}

impl SineNode {
    fn new() -> Self {
        Self { phase: 0. }
    }
}

impl Node for SineNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let freq = inputs.get(0).expect("sine: missing freq input");
        let reset_phase = inputs.get(1).unwrap_or(&0.0);
        if *reset_phase > 0.0 {
            self.phase = 0.0;
        }

        let y = (2. * PI * self.phase).sin();
        self.phase += freq / SAMPLE_RATE;
        if self.phase >= 1. {
            self.phase -= 1.;
        }
        y
    }
}

#[derive(Clone)]
pub(crate) struct SquareNode {
    phase: f32,
}

impl SquareNode {
    fn new() -> Self {
        Self { phase: 0. }
    }
}

impl Node for SquareNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let freq = inputs.get(0).expect("square: missing freq input");
        let reset_phase = inputs.get(1).unwrap_or(&0.0);
        if *reset_phase > 0.0 {
            self.phase = 0.0;
        }

        let inc = 2. * PI * freq / SAMPLE_RATE;
        let y = if self.phase < PI { 1. } else { -1. };
        self.phase += inc;

        if self.phase >= 2. * PI {
            self.phase -= 2. * PI;
        }

        y
    }
}

#[derive(Clone)]
pub struct SawNode {
    period: f32,
    amplitude: f32,
    phase: f32,
    phase_max: f32,
    inc: f32,
    sin0: f32,
    sin1: f32,
    dsin: f32,
    dc: f32,
    saw: f32,
    sample_rate: f32,
}

impl SawNode {
    fn new() -> Self {
        Self {
            period: 0.0,
            amplitude: 1.0,
            phase: 0.0,
            phase_max: 0.0,
            inc: 0.0,
            sin0: 0.0,
            sin1: 0.0,
            dsin: 0.0,
            dc: 0.0,
            saw: 0.0,
            sample_rate: SAMPLE_RATE,
        }
    }

    #[inline(always)]
    #[nonblocking]
    fn next_sample(&mut self) -> f32 {
        let y;
        self.phase += self.inc;

        if self.phase <= FRAC_PI_4 {
            let half_period = self.period / 2.0;
            self.phase_max = (0.5 + half_period).floor() - 0.5;
            self.dc = 0.5 * self.amplitude / self.phase_max; // calculate DC offset
            self.phase_max *= std::f32::consts::PI;

            self.inc = self.phase_max / half_period;
            self.phase = -self.phase;

            // digital resonator approximation of a sine function
            self.sin0 = self.amplitude * self.phase.sin();
            self.sin1 = self.amplitude * (self.phase - self.inc).sin();
            self.dsin = 2.0 * self.inc.cos();

            if self.phase * self.phase > 1e-9 {
                y = self.sin0 / self.phase;
            } else {
                y = self.amplitude;
            }
        } else {
            if self.phase > self.phase_max {
                self.phase = self.phase_max + self.phase_max - self.phase;
                self.inc = -self.inc;
            }

            let sinp = self.dsin * self.sin0 - self.sin1;
            self.sin1 = self.sin0;
            self.sin0 = sinp;
            y = sinp / self.phase;
        }

        y - self.dc
    }
}

impl Node for SawNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let freq = inputs.get(0).expect("saw: missing freq input");
        let reset_phase = inputs.get(1).unwrap_or(&0.0);
        if *reset_phase > 0.0 {
            self.phase = 0.0;
        }

        self.period = self.sample_rate / freq;
        let sample = self.next_sample();
        self.saw = self.saw * 0.997 + sample;
        self.saw
    }
}

#[derive(Clone)]
pub(crate) struct NoiseNode {
    rng: Arc<Mutex<SmallRng>>,
}

impl NoiseNode {
    fn new() -> Self {
        Self {
            rng: Arc::new(Mutex::new(SmallRng::from_entropy())),
        }
    }
}

impl Node for NoiseNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, _: &[f32]) -> f32 {
        self.rng.lock().unwrap().gen::<f32>() * 2. - 1.
    }
}

#[derive(Clone)]
pub(crate) struct PulseNode {
    phase: f32,
    prev_phase: f32,
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
    state: EnvelopeState,
    value: f32,
    time: f32,
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
pub enum SVFMode {
    Lowpass,
    Highpass,
    Bandpass,
}

/// Cytomic (Andrew Simper) state-variable filter
#[derive(Clone)]
pub struct SVFNode {
    g: f32,
    k: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    ic1eq: f32,
    ic2eq: f32,
    mode: SVFMode,
    set: bool,
}

impl SVFNode {
    fn new() -> SVFNode {
        let mut svf = Self {
            g: 0.0,
            k: 0.0,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            ic1eq: 0.0,
            ic2eq: 0.0,
            mode: SVFMode::Lowpass,
            set: false,
        };
        svf.update_coefficients();
        svf
    }

    #[inline]
    #[nonblocking]
    fn update_coefficients(&mut self) {
        self.a1 = 1.0 / (1.0 + self.g * (self.g + self.k));
        self.a2 = self.g * self.a1;
        self.a3 = self.g * self.a2;
    }
}

impl Node for SVFNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let mode = *inputs.get(0).expect("svf: missing mode") as i32;
        let cutoff = inputs.get(1).expect("svf: missing cutoff input");
        self.g = (std::f32::consts::PI * cutoff / SAMPLE_RATE).tan();
        let resonance = inputs.get(2).expect("svf: missing resonance input");
        let input = inputs.get(3).expect("svf: missing input");
        self.k = 1.0 / resonance;
        self.update_coefficients();

        let v3 = input - self.ic2eq;
        let v1 = self.a1 * self.ic1eq + self.a2 * v3;
        let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;
        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;

        match mode {
            0 => v2,                                        // lowpass
            1 => input - self.ic2eq - self.a2 * self.ic1eq, // highpass
            2 => self.k * v1,                               // bandpass
            _ => panic!("invalid mode"),
        }
    }
}

#[derive(Clone)]
pub(crate) struct SeqNode {
    values: Vec<f32>,
    step: usize,
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

#[derive(Clone)]
pub(crate) struct PipeNode {
    buffer: [f32; BUFFER_SIZE],
    read_pos: usize,
}

impl PipeNode {
    fn new() -> Self {
        Self {
            buffer: [0.0; BUFFER_SIZE],
            read_pos: 0,
        }
    }
}

impl Node for PipeNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let delay = inputs.get(0).expect("pipe: missing delay input");
        let input = inputs.get(1).expect("pipe: missing input");
        self.buffer[self.read_pos] = *input;
        self.read_pos = (self.read_pos + 1) % BUFFER_SIZE;
        *self
            .buffer
            .get((self.read_pos + *delay as usize) % BUFFER_SIZE)
            .unwrap_or(&0.0)
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
pub struct ReverbNode {
    reverb: Reverb,
}

impl ReverbNode {
    fn new() -> Self {
        Self {
            reverb: Reverb::new(),
        }
    }
}

impl Node for ReverbNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let input = inputs.get(0).expect("reverb: missing input");
        self.reverb.process(*input)
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
