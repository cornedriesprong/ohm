// peak limiter on the output, linked across channels to keep the stereo image
const CEILING: f32 = 1.0;
const RELEASE_SECONDS: f32 = 0.1;

pub struct Limiter {
    gain: f32,
}

impl Default for Limiter {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl Limiter {
    // attacks instantly, so peaks never exceed the ceiling, and releases exponentially
    pub fn process(&mut self, frame: &mut [f32], sample_rate: u32) {
        let peak = frame.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
        let target = if peak > CEILING { CEILING / peak } else { 1.0 };
        if target < self.gain {
            self.gain = target;
        } else {
            let coef = 1.0 - (-1.0 / (RELEASE_SECONDS * sample_rate as f32)).exp();
            self.gain += (target - self.gain) * coef;
        }
        for sample in frame.iter_mut() {
            *sample = (*sample * self.gain).clamp(-CEILING, CEILING);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_peaks() {
        let mut limiter = Limiter::default();
        let mut frame = [2.0, -1.0];
        limiter.process(&mut frame, 48_000);
        assert_eq!(frame, [1.0, -0.5]);
    }

    #[test]
    fn passes_quiet_signals() {
        let mut limiter = Limiter::default();
        let mut frame = [0.5, -0.25];
        limiter.process(&mut frame, 48_000);
        assert_eq!(frame, [0.5, -0.25]);
    }

    #[test]
    fn releases() {
        let mut limiter = Limiter::default();
        limiter.process(&mut [4.0], 48_000);
        let mut frame = [0.5];
        for _ in 0..48_000 {
            frame = [0.5];
            limiter.process(&mut frame, 48_000);
        }
        assert!((frame[0] - 0.5).abs() < 1e-3);
    }
}
