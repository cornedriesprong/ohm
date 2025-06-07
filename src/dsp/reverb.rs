use std::vec;

use crate::dsp::delay::{DelayLine, InterpolationType};
use crate::dsp::filters::{AllPass, SVF};
use rand::{thread_rng, Rng};

#[derive(Clone)]
struct ReverbPath {
    delay_line: DelayLine,
    svf: SVF,
    delay_time: i32,
    is_inverted: bool,
    feedback: f32,
}

impl ReverbPath {
    fn new() -> Self {
        let mut rng = thread_rng();
        let delay_time = rng.gen_range(10..10000);
        let is_inverted = rng.gen_bool(1.0 / 3.0);

        Self {
            delay_line: DelayLine::new(InterpolationType::None, delay_time),
            svf: SVF::new(5000.0, 0.707),
            delay_time: delay_time as i32,
            is_inverted,
            feedback: 0.95,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let mut read_pos = self.delay_line.index as f32 - self.delay_time as f32;
        while read_pos < 0.0 {
            read_pos += self.delay_line.buffer.len() as f32
        }

        let mut y = self.delay_line.read(Some(read_pos as usize));

        y = x + (y * self.feedback);

        // randomly invert the signal
        if self.is_inverted {
            y = -y
        };

        // low pass filter
        y = self.svf.process(y, 0.0);

        // write the signal back to the delay line
        self.delay_line.write_and_increment(y);

        y
    }
}

const ALLPASS_COUNT: usize = 8;
const DELAY_COUNT: usize = 32;
const ALLPASS_LENGTHS: [usize; ALLPASS_COUNT] = [861, 732, 642, 562, 410, 352, 285, 199];

#[derive(Clone)]
pub struct Reverb {
    allpasses: vec::Vec<AllPass>,
    paths: vec::Vec<ReverbPath>,
}

impl Reverb {
    pub fn new() -> Self {
        let allpasses = (0..ALLPASS_COUNT)
            .map(|i| AllPass::new(ALLPASS_LENGTHS[i]))
            .collect();
        let paths = (0..DELAY_COUNT).map(|_| ReverbPath::new()).collect();
        Self { allpasses, paths }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let x = self
            .allpasses
            .iter_mut()
            .fold(x, |acc, allpass| allpass.process(acc))
            / 8.0;

        let mut xs = [0.0; DELAY_COUNT];
        for (i, path) in self.paths.iter_mut().enumerate() {
            xs[i] += path.process(x);
        }

        Self::mix(&mut xs);

        xs.iter().fold(0.0, |acc, &x| acc + x) / DELAY_COUNT as f32
    }

    // Householder mixing matrix
    #[inline]
    fn mix(arr: &mut [f32; 32]) {
        let mut sum = 0.0;
        for i in 0..32 {
            sum += arr[i];
        }

        sum *= -2.0 / 32.0;

        for i in 0..32 {
            arr[i] += sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::IMPULSE_SIGNAL;

    #[test]
    fn test_reverb() {
        let mut reverb = Reverb::new();
        for i in IMPULSE_SIGNAL.iter() {
            let y = reverb.process(*i);
            println!("y: {}", y);
            assert!(y >= -1.0 && y <= 1.0);
        }
    }
}
