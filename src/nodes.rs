use crate::utils::lerp;
use core::fmt;
use fmt::{Debug, Formatter};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::f32::consts::{FRAC_PI_4, PI};
use std::sync::{Arc, Mutex};

pub const SAMPLE_RATE: f32 = 44100.;

pub(crate) trait Node: Send + Sync {
    fn tick(&mut self) -> f32;
}

// #[derive(Clone)]
pub(crate) enum NodeKind {
    Constant(Constant),
    Gain(Gain),
    Mix(Mix),
    Sine(Sine),
    Square(Square),
    Saw(Saw),
    Noise(Noise),
    Pulse(Pulse),
    AR(AR),
    SVF(SVF),
    Seq(Seq),
}

impl NodeKind {
    pub(crate) fn constant(value: f32) -> Self {
        NodeKind::Constant(Constant::new(value))
    }

    pub(crate) fn gain(lhs: Box<dyn Node>, rhs: Box<dyn Node>) -> Self {
        NodeKind::Gain(Gain::new(lhs, rhs))
    }

    pub(crate) fn mix(lhs: Box<dyn Node>, rhs: Box<dyn Node>) -> Self {
        NodeKind::Mix(Mix::new(lhs, rhs))
    }

    pub(crate) fn sine(freq: Box<dyn Node>) -> Self {
        NodeKind::Sine(Sine::new(freq))
    }

    pub(crate) fn square(freq: Box<dyn Node>) -> Self {
        NodeKind::Square(Square::new(freq))
    }

    pub(crate) fn saw(freq: Box<dyn Node>) -> Self {
        NodeKind::Saw(Saw::new(freq))
    }

    pub(crate) fn noise() -> Self {
        NodeKind::Noise(Noise::new())
    }

    pub(crate) fn pulse(freq: Box<dyn Node>) -> Self {
        NodeKind::Pulse(Pulse::new(freq))
    }

    pub(crate) fn ar(attack: Box<dyn Node>, release: Box<dyn Node>, trig: Box<dyn Node>) -> Self {
        NodeKind::AR(AR::new(attack, release, trig))
    }

    pub(crate) fn svf(
        cutoff: Box<dyn Node>,
        resonance: Box<dyn Node>,
        input: Box<dyn Node>,
    ) -> Self {
        NodeKind::SVF(SVF::new(cutoff, resonance, input))
    }

    pub(crate) fn seq(values: Vec<f32>, trig: Box<dyn Node>) -> Self {
        NodeKind::Seq(Seq::new(values, trig))
    }
}

impl Node for NodeKind {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        match self {
            NodeKind::Constant(node) => node.tick(),
            NodeKind::Sine(node) => node.tick(),
            NodeKind::Gain(node) => node.tick(),
            NodeKind::Mix(node) => node.tick(),
            NodeKind::Square(node) => node.tick(),
            NodeKind::Saw(node) => node.tick(),
            NodeKind::Noise(node) => node.tick(),
            NodeKind::Pulse(node) => node.tick(),
            NodeKind::AR(node) => node.tick(),
            NodeKind::SVF(node) => node.tick(),
            NodeKind::Seq(node) => node.tick(),
        }
    }
}

impl PartialEq for NodeKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (NodeKind::Constant(a), NodeKind::Constant(b)) => a.value == b.value,
            (NodeKind::Mix(_), NodeKind::Mix(_)) => true,
            (NodeKind::Gain(_), NodeKind::Gain(_)) => true,
            (NodeKind::Sine(_), NodeKind::Sine(_)) => true,
            (NodeKind::AR(_), NodeKind::AR(_)) => true,
            (NodeKind::Square(_), NodeKind::Square(_)) => true,
            (NodeKind::Saw(_), NodeKind::Saw(_)) => true,
            (NodeKind::Noise(_), NodeKind::Noise(_)) => true,
            (NodeKind::Pulse(_), NodeKind::Pulse(_)) => true,
            (NodeKind::SVF(_), NodeKind::SVF(_)) => true,
            (NodeKind::Seq(a), NodeKind::Seq(b)) => a.values == b.values,
            _ => false,
        }
    }
}

impl Debug for NodeKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            NodeKind::Constant(c) => write!(f, "Constant({})", c.value),
            NodeKind::Gain(_) => write!(f, "Gain"),
            NodeKind::Mix(_) => write!(f, "Mix"),
            NodeKind::Sine(_) => write!(f, "Sine"),
            NodeKind::Square(_) => write!(f, "Square"),
            NodeKind::Saw(_) => write!(f, "Saw"),
            NodeKind::Noise(_) => write!(f, "Noise"),
            NodeKind::Pulse(_) => write!(f, "Pulse"),
            NodeKind::AR(_) => write!(f, "AR"),
            NodeKind::SVF(_) => write!(f, "SVF"),
            NodeKind::Seq(_) => write!(f, "Seq"),
        }
    }
}

// #[derive(Clone)]
pub(crate) struct Gain {
    lhs: Box<dyn Node>,
    rhs: Box<dyn Node>,
}

impl Gain {
    pub(crate) fn new(lhs: Box<dyn Node>, rhs: Box<dyn Node>) -> Self {
        Self { lhs, rhs }
    }
}

impl Node for Gain {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        self.lhs.tick() * self.rhs.tick()
    }
}

pub(crate) struct Mix {
    lhs: Box<dyn Node>,
    rhs: Box<dyn Node>,
}

impl Mix {
    pub(crate) fn new(lhs: Box<dyn Node>, rhs: Box<dyn Node>) -> Self {
        Self { lhs, rhs }
    }
}

impl Node for Mix {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        self.lhs.tick() + self.rhs.tick()
    }
}

pub(crate) struct Constant {
    pub value: f32,
}

impl Constant {
    pub(crate) fn new(value: f32) -> Self {
        Self { value }
    }
}

impl Node for Constant {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        self.value
    }
}

pub(crate) struct Sine {
    freq: Box<dyn Node>,
    phase: f32,
}

impl Sine {
    pub(crate) fn new(freq: Box<dyn Node>) -> Self {
        Self { freq, phase: 0. }
    }
}

impl Node for Sine {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        let hz = self.freq.tick();
        let y = (2. * PI * self.phase).sin();
        self.phase += hz / SAMPLE_RATE;
        if self.phase >= 1. {
            self.phase -= 1.;
        }
        y
    }
}

pub(crate) struct Square {
    freq: Box<dyn Node>,
    phase: f32,
}

impl Square {
    pub(crate) fn new(freq: Box<dyn Node>) -> Self {
        Self { freq, phase: 0. }
    }
}

impl Node for Square {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        let hz = self.freq.tick();
        let inc = 2. * PI * hz / SAMPLE_RATE;
        let y = if self.phase < PI { 1. } else { -1. };
        self.phase += inc;

        if self.phase >= 2. * PI {
            self.phase -= 2. * PI;
        }

        y
    }
}

pub struct Saw {
    freq: Box<dyn Node>,
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

impl Saw {
    pub fn new(freq: Box<dyn Node>) -> Self {
        Self {
            freq,
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

impl Node for Saw {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        let hz = self.freq.tick();
        self.period = self.sample_rate / hz;
        let sample = self.next_sample();
        self.saw = self.saw * 0.997 + sample;
        self.saw
    }
}

// #[derive(Clone)]
pub(crate) struct Noise {
    rng: Arc<Mutex<SmallRng>>,
}

impl Noise {
    pub(crate) fn new() -> Self {
        Self {
            rng: Arc::new(Mutex::new(SmallRng::from_entropy())),
        }
    }
}

impl Node for Noise {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        self.rng.lock().unwrap().gen::<f32>() * 2. - 1.
    }
}

pub(crate) struct Pulse {
    freq: Box<dyn Node>,
    phase: f32,
    prev_phase: f32,
}

impl Pulse {
    pub(crate) fn new(freq: Box<dyn Node>) -> Self {
        Self {
            freq,
            phase: 0.,
            prev_phase: 0.,
        }
    }
}

impl Node for Pulse {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        let hz = self.freq.tick();
        self.prev_phase = self.phase;
        self.phase += 2. * PI * hz / SAMPLE_RATE;

        if self.phase >= 2. * PI {
            self.phase -= 2. * PI;
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EnvelopeState {
    Attack,
    Release,
    Off,
}

pub(crate) struct AR {
    attack: Box<dyn Node>,
    release: Box<dyn Node>,
    trig: Box<dyn Node>,
    state: EnvelopeState,
    value: f32,
    time: f32,
}

impl AR {
    pub(crate) fn new(attack: Box<dyn Node>, release: Box<dyn Node>, trig: Box<dyn Node>) -> Self {
        Self {
            attack,
            release,
            trig,
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

impl Node for AR {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        let attack = self.attack.tick();
        let release = self.release.tick();
        let trig = self.trig.tick();

        if trig > 0.0 {
            self.state = EnvelopeState::Attack;
        }

        use EnvelopeState as E;
        match self.state {
            E::Attack => {
                if attack == 0.0 {
                    self.value = 1.0;
                } else {
                    self.value = self.get_curve(attack);
                }

                if self.value >= 1.0 {
                    self.value = 1.0;
                    self.time = 0.0;
                    self.state = E::Release;
                }
            }
            E::Release => {
                self.value = self.get_curve_rev(release);
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
pub struct SVF {
    cutoff: Box<dyn Node>,
    resonance: Box<dyn Node>,
    input: Box<dyn Node>,
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

impl SVF {
    pub fn new(cutoff: Box<dyn Node>, resonance: Box<dyn Node>, input: Box<dyn Node>) -> SVF {
        let mut svf = Self {
            cutoff,
            resonance,
            input,
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
    fn update_coefficients(&mut self) {
        self.a1 = 1.0 / (1.0 + self.g * (self.g + self.k));
        self.a2 = self.g * self.a1;
        self.a3 = self.g * self.a2;
    }
}

impl Node for SVF {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        let cutoff = self.cutoff.tick();
        self.g = (std::f32::consts::PI * cutoff / SAMPLE_RATE).tan();
        let resonance = self.resonance.tick();
        let input = self.input.tick();
        self.k = 1.0 / resonance;
        self.update_coefficients();

        let v3 = input - self.ic2eq;
        let v1 = self.a1 * self.ic1eq + self.a2 * v3;
        let v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3;
        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;

        match self.mode {
            SVFMode::Lowpass => v2,
            SVFMode::Highpass => input - self.ic2eq - self.a2 * self.ic1eq,
            SVFMode::Bandpass => self.k * v1,
        }
    }
}

pub(crate) struct Seq {
    values: Vec<f32>,
    trig: Box<dyn Node>,
    step: usize,
}

impl Seq {
    pub(crate) fn new(values: Vec<f32>, trig: Box<dyn Node>) -> Self {
        Self {
            values,
            trig,
            step: 0,
        }
    }

    fn increment(&mut self) {
        self.step += 1;
        if self.step >= self.values.len() {
            self.step = 0;
        }
    }
}

impl Node for Seq {
    #[inline(always)]
    fn tick(&mut self) -> f32 {
        let trig = self.trig.tick();
        if trig > 0.0 {
            self.increment();
        }

        self.values[self.step]
    }
}
