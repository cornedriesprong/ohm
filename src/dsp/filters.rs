use crate::dsp::delay::{DelayLine, InterpolationType};

#[derive(Debug, Clone, Copy)]
pub enum SVFMode {
    Lowpass,
    Highpass,
    Bandpass,
}

/// Simple SVF for internal use (like reverb)
#[derive(Clone)]
pub struct SVF {
    filter:
        fundsp::hacker32::An<fundsp::hacker32::FixedSvf<f32, fundsp::hacker32::LowpassMode<f32>>>,
}

impl SVF {
    pub fn new(freq: f32, q: f32) -> Self {
        use fundsp::hacker32::*;
        Self {
            filter: lowpass_hz(freq, q),
        }
    }

    pub fn process(&mut self, input: f32, _freq_mod: f32) -> f32 {
        self.filter.tick(&[input].into())[0]
    }
}

/// Schroeder all-pass filter
#[derive(Clone)]
pub struct AllPass {
    delay_line: DelayLine,
    feedback: f32,
}

impl AllPass {
    pub fn new(length: usize) -> Self {
        return Self {
            delay_line: DelayLine::new(InterpolationType::Linear, length),
            feedback: 0.5,
        };
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let delayed = self.delay_line.read(None);
        let y = -x + delayed;
        self.delay_line
            .write_and_increment(x + (delayed * self.feedback));
        y
    }
}
