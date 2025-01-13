use crate::utils::lerp;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};

pub const SAMPLE_RATE: f32 = 44100.;

pub(crate) trait Node: Send + Sync {
    fn process(&mut self, input: f32, args: &[f32]) -> f32;
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
    fn process(&mut self, _: f32, _: &[f32]) -> f32 {
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
    fn process(&mut self, hz: f32, _: &[f32]) -> f32 {
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
    fn process(&mut self, hz: f32, _: &[f32]) -> f32 {
        let inc = 2. * PI * hz / SAMPLE_RATE;
        let y = if self.phase < PI { 1. } else { -1. };
        self.phase += inc;

        if self.phase >= 2. * PI {
            self.phase -= 2. * PI;
        }

        y
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
    fn process(&mut self, _: f32, _: &[f32]) -> f32 {
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
    fn process(&mut self, hz: f32, _: &[f32]) -> f32 {
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
    fn process(&mut self, a: f32, args: &[f32]) -> f32 {
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
    fn process(&mut self, a: f32, args: &[f32]) -> f32 {
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
    fn process(&mut self, input: f32, args: &[f32]) -> f32 {
        let trig = args.get(0).unwrap_or(&0.0);
        let attack = args.get(1).unwrap_or(&1.0);
        let release = args.get(2).unwrap_or(&1.0);

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

        input * self.value
    }
}
