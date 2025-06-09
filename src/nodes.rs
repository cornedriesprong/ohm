use crate::audio_graph::BoxedNode;
use crate::consts::SAMPLE_RATE;
use crate::dsp::delay::Delay;
use crate::dsp::filters::SVFMode;
use crate::dsp::reverb::Reverb;
use crate::utils::{freq_to_period, lerp};
use core::fmt;
use fmt::{Debug, Formatter};
use fundsp::hacker32::*;
use fundsp::typenum::{UInt, UTerm, B1, U1, U3};
use koto::{derive::*, prelude::*, runtime::Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rtsan_standalone::nonblocking;
use seahash::SeaHasher;
use std::f32::consts::PI;
use std::hash::{Hash, Hasher};
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
    Triangle {
        freq: BoxedNode,
        node: TriangleNode,
    },
    Moog {
        cutoff: BoxedNode,
        resonance: BoxedNode,
        input: BoxedNode,
        node: MoogNode,
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

pub(crate) fn triangle(freq: NodeKind) -> NodeKind {
    NodeKind::Triangle {
        freq: Box::new(freq),
        node: TriangleNode::new(),
    }
}

pub(crate) fn moog(cutoff: NodeKind, resonance: NodeKind, input: NodeKind) -> NodeKind {
    NodeKind::Moog {
        cutoff: Box::new(cutoff),
        resonance: Box::new(resonance),
        input: Box::new(input),
        node: MoogNode::new(),
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
            NodeKind::Triangle { node, .. } => node.tick(inputs),
            NodeKind::Moog { node, .. } => node.tick(inputs),
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
            (Self::Triangle { .. }, Self::Triangle { .. }) => true,
            (Self::Moog { .. }, Self::Moog { .. }) => true,
            _ => false,
            // _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl NodeKind {
    pub(crate) fn compute_hash(&self) -> u64 {
        let mut hasher = SeaHasher::new();
        self.hash_structure(&mut hasher);
        hasher.finish()
    }

    fn hash_structure(&self, hasher: &mut SeaHasher) {
        match self {
            NodeKind::Constant(node) => {
                0u8.hash(hasher);
                node.value.to_bits().hash(hasher);
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
            NodeKind::SVF {
                mode,
                cutoff,
                resonance,
                input,
                ..
            } => {
                9u8.hash(hasher);
                mode.hash_structure(hasher);
                cutoff.hash_structure(hasher);
                resonance.hash_structure(hasher);
                input.hash_structure(hasher);
            }
            NodeKind::Seq { trig, node } => {
                10u8.hash(hasher);
                trig.hash_structure(hasher);
                for val in &node.values {
                    val.to_bits().hash(hasher);
                }
            }
            NodeKind::Pipe { delay, input, .. } => {
                11u8.hash(hasher);
                delay.hash_structure(hasher);
                input.hash_structure(hasher);
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
                input: audio,
                cutoff,
                resonance,
                ..
            } => {
                16u8.hash(hasher);
                audio.hash_structure(hasher);
                cutoff.hash_structure(hasher);
                resonance.hash_structure(hasher);
            }
        }
    }

    pub(crate) fn transfer_state_from(&mut self, other: &NodeKind) {
        match (self, other) {
            (NodeKind::Sine { node: new, .. }, NodeKind::Sine { node: old, .. }) => {
                *new = SineNode::new_with_phase(old.get_phase());
            }
            (NodeKind::Square { node: new, .. }, NodeKind::Square { node: old, .. }) => {
                *new = SquareNode::new_with_phase(old.get_phase());
            }
            (NodeKind::Saw { node: new, .. }, NodeKind::Saw { node: old, .. }) => {
                *new = SawNode::new_with_phase(old.get_phase());
            }
            (NodeKind::Pulse { node: new, .. }, NodeKind::Pulse { node: old, .. }) => {
                new.phase = old.phase;
                new.prev_phase = old.prev_phase;
            }
            (NodeKind::AR { node: new, .. }, NodeKind::AR { node: old, .. }) => {
                new.state = old.state;
                new.value = old.value;
                new.time = old.time;
            }
            (NodeKind::Seq { node: new, .. }, NodeKind::Seq { node: old, .. }) => {
                new.step = old.step;
            }
            (NodeKind::Pipe { node: new, .. }, NodeKind::Pipe { node: old, .. }) => {
                new.buffer = old.buffer;
                new.read_pos = old.read_pos;
            }
            (NodeKind::Pluck { node: new, .. }, NodeKind::Pluck { node: old, .. }) => {
                new.buffer = old.buffer;
                new.period = old.period;
                new.read_pos = old.read_pos;
                new.pitch_track = old.pitch_track;
                new.is_stopped = old.is_stopped;
            }
            (NodeKind::Reverb { node: new, .. }, NodeKind::Reverb { node: old, .. }) => {
                new.reverb = old.reverb.clone();
            }
            (NodeKind::Delay { node: new, .. }, NodeKind::Delay { node: old, .. }) => {
                new.delay = old.delay.clone();
            }
            (NodeKind::Triangle { node: new, .. }, NodeKind::Triangle { node: old, .. }) => {
                let old_phase = old.get_phase();
                *new = TriangleNode::new_with_phase(old_phase);
            }
            (NodeKind::Moog { node: new, .. }, NodeKind::Moog { node: old, .. }) => {
                *new = old.clone(); // Filters preserve internal state via clone
            }
            _ => {}
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

macro_rules! define_node {
    (
        $name:ident,
        osc: $osc_type:ty = $osc_fn:ident,
        ramp: $ramp_type:ty = $ramp_fn:ident,
        ramp_phase_fn: $ramp_phase_fn:ident,
    ) => {
        #[derive(Clone)]
        pub(crate) struct $name {
            osc: An<$osc_type>,
            ramp: An<$ramp_type>,
            phase: f32,
        }

        impl $name {
            fn new() -> Self {
                use fundsp::hacker32::{$osc_fn, $ramp_fn};
                Self {
                    osc: $osc_fn(),
                    ramp: $ramp_fn(),
                    phase: 0.0,
                }
            }

            fn new_with_phase(phase: f32) -> Self {
                use fundsp::hacker32::{$osc_fn, $ramp_fn};
                Self {
                    osc: $osc_fn(),
                    ramp: $ramp_fn(),
                    phase: phase,
                }
            }

            pub fn get_phase(&self) -> f32 {
                self.phase
            }
        }

        impl Node for $name {
            #[inline(always)]
            #[nonblocking]
            fn tick(&mut self, inputs: &[f32]) -> f32 {
                let freq = inputs
                    .get(0)
                    .expect(concat!(stringify!($name), ": missing freq input"));
                self.phase = self.ramp.tick(&[*freq].into())[0];
                self.osc.tick(&[*freq].into())[0]
            }
        }
    };
}

define_node!(
    SineNode,
    osc: Sine<f32> = sine,
    ramp: Ramp<f32> = ramp,
    ramp_phase_fn: ramp_phase,
);

define_node!(
    SquareNode,
    osc: WaveSynth<UInt<UTerm, B1>> = square,
    ramp: Ramp<f32> = ramp,
    ramp_phase_fn: ramp_phase,
);

define_node!(
    SawNode,
    osc: WaveSynth<UInt<UTerm, B1>> = saw,
    ramp: Ramp<f32> = ramp,
    ramp_phase_fn: ramp_phase,
);

define_node!(
    TriangleNode,
    osc: WaveSynth<U1> = triangle,
    ramp: Ramp<f32> = ramp,
    ramp_phase_fn: ramp_phase,
);

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

/// FunDSP-based state-variable filter
pub enum SVFNode {
    Lowpass(An<Svf<f32, LowpassMode<f32>>>),
    Highpass(An<Svf<f32, HighpassMode<f32>>>),
    Bandpass(An<Svf<f32, BandpassMode<f32>>>),
}

impl Clone for SVFNode {
    fn clone(&self) -> Self {
        match self {
            SVFNode::Lowpass(filter) => SVFNode::Lowpass(filter.clone()),
            SVFNode::Highpass(filter) => SVFNode::Highpass(filter.clone()),
            SVFNode::Bandpass(filter) => SVFNode::Bandpass(filter.clone()),
        }
    }
}

impl SVFNode {
    fn new() -> SVFNode {
        SVFNode::Lowpass(lowpass())
    }
}

impl Node for SVFNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let mode = *inputs.get(0).expect("svf: missing mode") as i32;
        let cutoff = *inputs.get(1).expect("svf: missing cutoff input");
        let resonance = *inputs.get(2).expect("svf: missing resonance input");
        let input = *inputs.get(3).expect("svf: missing input");

        // Update filter based on mode (only if mode changed)
        let target_mode = match mode {
            0 => SVFMode::Lowpass,
            1 => SVFMode::Highpass,
            2 => SVFMode::Bandpass,
            _ => panic!("invalid mode"),
        };

        // Check if we need to change filter type
        let needs_mode_change = match (&*self, target_mode) {
            (SVFNode::Lowpass(_), SVFMode::Lowpass) => false,
            (SVFNode::Highpass(_), SVFMode::Highpass) => false,
            (SVFNode::Bandpass(_), SVFMode::Bandpass) => false,
            _ => true,
        };

        if needs_mode_change {
            *self = match target_mode {
                SVFMode::Lowpass => SVFNode::Lowpass(lowpass()),
                SVFMode::Highpass => SVFNode::Highpass(highpass()),
                SVFMode::Bandpass => SVFNode::Bandpass(bandpass()),
            };
        }

        // Process with dynamic parameters - filters accept [audio, cutoff, Q]
        match self {
            SVFNode::Lowpass(filter) => filter.tick(&[input, cutoff, resonance].into())[0],
            SVFNode::Highpass(filter) => filter.tick(&[input, cutoff, resonance].into())[0],
            SVFNode::Bandpass(filter) => filter.tick(&[input, cutoff, resonance].into())[0],
        }
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

#[derive(Clone)]
pub(crate) struct MoogNode {
    filter: An<Moog<f32, U3>>,
}

impl MoogNode {
    fn new() -> Self {
        use fundsp::hacker32::moog;
        Self { filter: moog() }
    }
}

impl Node for MoogNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[f32]) -> f32 {
        let cutoff = inputs[0];
        let resonance = inputs[1];
        let input = inputs[2];
        self.filter.tick(&[input, cutoff, resonance].into())[0]
    }
}
