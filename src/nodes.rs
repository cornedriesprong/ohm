use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};

pub const SAMPLE_RATE: f32 = 44100.0;

pub(crate) trait Node: Send + Sync {
    fn process(&mut self, input: f32) -> f32;
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
    fn process(&mut self, _: f32) -> f32 {
        self.value
    }
}

pub(crate) struct Sine {
    phase: f32,
}

impl Sine {
    pub(crate) fn new() -> Self {
        Self { phase: 0.0 }
    }
}

impl Node for Sine {
    fn process(&mut self, hz: f32) -> f32 {
        let y = (2.0 * PI * self.phase).sin();
        self.phase += hz / SAMPLE_RATE;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        y
    }
}

pub(crate) struct Square {
    phase: f32,
}

impl Square {
    pub(crate) fn new() -> Self {
        Self { phase: 0.0 }
    }
}

impl Node for Square {
    fn process(&mut self, x: f32) -> f32 {
        let inc = 2.0 * PI * x / SAMPLE_RATE;
        let y = if self.phase < PI { 1.0 } else { -1.0 };
        self.phase += inc;

        if self.phase >= 2.0 * PI {
            self.phase -= 2.0 * PI;
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
    fn process(&mut self, _: f32) -> f32 {
        self.rng.lock().unwrap().gen::<f32>() * 2.0 - 1.0
    }
}

pub(crate) struct Gain {
    gain: f32,
}

impl Gain {
    pub(crate) fn new(gain: f32) -> Self {
        Self { gain }
    }
}

impl Node for Gain {
    fn process(&mut self, input: f32) -> f32 {
        input * self.gain
    }
}

pub(crate) struct Offset {
    offset: f32,
}

impl Offset {
    pub(crate) fn new(offset: f32) -> Self {
        Self { offset }
    }
}

impl Node for Offset {
    fn process(&mut self, input: f32) -> f32 {
        input + self.offset
    }
}
