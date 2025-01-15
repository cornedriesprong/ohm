use crate::utils::lerp;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::f32::consts::FRAC_PI_4;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};

pub const SAMPLE_RATE: f32 = 44100.;

pub(crate) trait Node: Send + Sync {
    fn tick(&mut self, input: f32, args: &[f32]) -> f32;
}

pub(crate) struct Constant {
    value: f32,
}

impl Constant {
    pub(crate) fn new(value: f32) -> Self {
        Self { value }
    }
}

impl Node for Constant {
    fn tick(&mut self, _: f32, _: &[f32]) -> f32 {
        self.value
    }
}

pub(crate) struct Sine {
    phase: f32,
}

impl Sine {
    pub(crate) fn new() -> Self {
        Self { phase: 0. }
    }
}

impl Node for Sine {
    #[inline]
    fn tick(&mut self, hz: f32, _: &[f32]) -> f32 {
        let y = (2. * PI * self.phase).sin();
        self.phase += hz / SAMPLE_RATE;
        if self.phase >= 1. {
            self.phase -= 1.;
        }
        y
    }
}

pub(crate) struct Square {
    phase: f32,
}

impl Square {
    pub(crate) fn new() -> Self {
        Self { phase: 0. }
    }
}

impl Node for Square {
    #[inline]
    fn tick(&mut self, hz: f32, _: &[f32]) -> f32 {
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
    pub fn new() -> Self {
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
    #[inline]
    fn tick(&mut self, hz: f32, _: &[f32]) -> f32 {
        self.period = self.sample_rate / hz;
        let sample = self.next_sample();
        self.saw = self.saw * 0.997 + sample;
        self.saw
    }
}

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
    #[inline]
    fn tick(&mut self, _: f32, _: &[f32]) -> f32 {
        self.rng.lock().unwrap().gen::<f32>() * 2. - 1.
    }
}

pub(crate) struct Pulse {
    phase: f32,
    prev_phase: f32,
}

impl Pulse {
    pub(crate) fn new() -> Self {
        Self {
            phase: 0.,
            prev_phase: 0.,
        }
    }
}

impl Node for Pulse {
    #[inline]
    fn tick(&mut self, hz: f32, _: &[f32]) -> f32 {
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

pub(crate) struct Gain {}

impl Gain {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Node for Gain {
    #[inline]
    fn tick(&mut self, a: f32, args: &[f32]) -> f32 {
        let b = args.first().expect("Mix node requires control input");
        a * b
    }
}

pub(crate) struct Mix {}

impl Mix {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Node for Mix {
    #[inline]
    fn tick(&mut self, a: f32, args: &[f32]) -> f32 {
        let b = args.first().expect("Mix node requires control input");
        a + b
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EnvelopeState {
    Attack,
    Release,
    Off,
}

pub(crate) struct AR {
    state: EnvelopeState,
    value: f32,
    time: f32,
}

impl AR {
    pub(crate) fn new() -> Self {
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

impl Node for AR {
    #[inline]
    fn tick(&mut self, trig: f32, args: &[f32]) -> f32 {
        let attack = args.get(0).unwrap_or(&1.0);
        let release = args.get(1).unwrap_or(&1.0);

        if trig > 0.0 {
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

#[derive(Debug, Clone, Copy)]
pub enum SVFMode {
    Lowpass,
    Highpass,
    Bandpass,
}

/// Cytomic (Andrew Simper) state-variable filter
#[derive(Debug, Clone, Copy)]
pub struct SVF {
    g: f32,
    k: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    ic1eq: f32,
    ic2eq: f32,
    sample_rate: f32,
    pub mode: SVFMode,
}

impl SVF {
    pub fn new() -> SVF {
        println!("new SVF");
        Self {
            g: 0.0,
            k: 0.0,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            ic1eq: 0.0,
            ic2eq: 0.0,
            sample_rate: SAMPLE_RATE,
            mode: SVFMode::Lowpass,
        }
    }

    #[inline]
    fn update_coefficients(&mut self) {
        self.a1 = 1.0 / (1.0 + self.g * (self.g + self.k));
        self.a2 = self.g * self.a1;
        self.a3 = self.g * self.a2;
    }
}

impl Node for SVF {
    #[inline]
    fn tick(&mut self, input: f32, args: &[f32]) -> f32 {
        let cutoff = args.get(0).unwrap_or(&1.0);
        self.g = (std::f32::consts::PI * cutoff / self.sample_rate).tan();
        let resonance = args.get(1).unwrap_or(&1.0);
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
    step: usize,
}

impl Seq {
    pub(crate) fn new(values: Vec<f32>) -> Self {
        Self { values, step: 0 }
    }

    fn increment(&mut self) {
        self.step += 1;
        if self.step >= self.values.len() {
            self.step = 0;
        }
    }
}

impl Node for Seq {
    #[inline]
    fn tick(&mut self, trig: f32, _: &[f32]) -> f32 {
        if trig > 0.0 {
            self.increment();
        }

        self.values[self.step]
    }
}
