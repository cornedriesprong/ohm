use std::vec;

// const BUFFER_LENGTH: usize = 48000; // 5 seconds at 48 Khz

#[derive(Clone)]
pub struct Delay {
    delay_line: DelayLine,
    time_samples: f32,
    // target_time: f32,
    // increment: f32,
    feedback: f32,
    // saturation: f32,
    // modulation_depth: f32,
    // mix: f32,
    // svf: SVF,
    // lfo: Oscillator,
}

impl Delay {
    pub fn new(time_samples: f32, feedback: f32) -> Self {
        Self {
            delay_line: DelayLine::new(InterpolationType::Cubic, time_samples as usize),
            time_samples,
            feedback,
        }
    }

    #[inline]
    pub fn process(&mut self, input: f32) -> f32 {
        let delayed = self.delay_line.read(None);
        let output = input + (delayed * self.feedback);
        self.delay_line.write_and_increment(output);

        output
    }

    pub fn set_delay_time(&mut self, time: f32) {
        self.time_samples = time;
    }

    pub fn set_feedback(&mut self, feedback: f32) {
        self.feedback = feedback;
    }

    fn cubic_interpolate(y0: f32, y1: f32, y2: f32, y3: f32, mu: f32) -> f32 {
        let mu2 = mu * mu;
        let a0 = y3 - y2 - y0 + y1;
        let a1 = y0 - y1 - a0;
        let a2 = y2 - y0;
        let a3 = y1;
        return a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3;
    }
}

#[derive(Clone)]
pub enum InterpolationType {
    None,
    Linear,
    Cubic,
}

#[derive(Clone)]
pub struct DelayLine {
    pub buffer: Vec<f32>,
    pub index: usize,
    length: usize,
    interpolation: InterpolationType,
}

impl DelayLine {
    pub fn new(interpolation: InterpolationType, length: usize) -> Self {
        Self {
            buffer: vec![0.0; length],
            index: 0,
            length,
            interpolation,
        }
    }

    pub fn read(&self, read_pos: Option<usize>) -> f32 {
        let mut read_pos = read_pos.unwrap_or(self.index) as f32;
        if read_pos < 0.0 {
            read_pos += self.length as f32;
        }

        match self.interpolation {
            InterpolationType::None => self.get_sample(read_pos as usize),
            InterpolationType::Linear => self.linear_interpolate(read_pos),
            InterpolationType::Cubic => self.cubic_interpolate(read_pos),
        }
    }

    pub fn write_and_increment(&mut self, value: f32) {
        self.buffer[self.index] = value;
        self.index = (self.index + 1) % self.length;
    }

    fn get_sample(&self, index: usize) -> f32 {
        self.buffer[index as usize]
    }

    fn linear_interpolate(&self, index: f32) -> f32 {
        let floor = index.floor() as usize;
        let frac = index - floor as f32;
        let s0 = self.get_sample(floor);
        let s1 = self.get_sample((floor + 1) % self.length);

        (1.0 - frac) * s0 + frac * s1
    }

    fn cubic_interpolate(&self, index: f32) -> f32 {
        let mut floor = index.floor() as usize;
        let frac = index - floor as f32;

        if floor == 0 {
            floor += 1;
        }

        let s0 = self.get_sample(floor - 1);
        let s1 = self.get_sample(floor);
        let s2 = self.get_sample((floor + 1) % self.length);
        let s3 = self.get_sample((floor + 2) % self.length);

        let a = (3.0 * (s1 - s2) - s0 + s3) / 2.0;
        let b = 2.0 * s2 + s0 - (5.0 / 2.0) * s1 - s3 / 2.0;
        let c = (s2 - s0) / 2.0;
        let d = s1;

        a * frac.powi(3) + b * frac.powi(2) + c * frac + d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_delay() {
        let delay = Delay::new(0.5, 0.5);
        assert_eq!(delay.time_samples, 0.5);
        assert_eq!(delay.feedback, 0.5);
    }

    #[test]
    fn new_creates_delay_line() {
        // let delay_line = DelayLine::new(InterpolationType::None, BUFFER_LENGTH);
        // assert_eq!(delay_line.index, 0);
        // assert_eq!(delay_line.buffer, [0.0; BUFFER_LENGTH]);
    }

    #[test]
    fn delay_line_write_and_increment() {
        // let mut delay_line = DelayLine::new(InterpolationType::None, BUFFER_LENGTH);
        // delay_line.write_and_increment(0.5);
        // assert_eq!(delay_line.index, 1);
        // assert_eq!(delay_line.buffer[0], 0.5);
    }

    #[test]
    fn delay_line_read() {
        // let mut delay_line = DelayLine::new(InterpolationType::None, BUFFER_LENGTH);
        // fill entire buffer
        // for _ in 0..BUFFER_LENGTH {
        //     delay_line.write_and_increment(0.5);
        // }
        // for i in 0..BUFFER_LENGTH {
        // assert_eq!(delay_line.read(i as f32), 0.5);
        // }
    }

    #[test]
    fn test_delay_line_linear_interpolate() {
        // let mut delay_line = DelayLine::new(InterpolationType::Linear, BUFFER_LENGTH);
        // delay_line.write_and_increment(1.0);
        // delay_line.write_and_increment(0.5);
        // assert_eq!(delay_line.read(0.5), 0.25);
        // assert_eq!(delay_line.read(1.0), 0.5);
        // assert_eq!(delay_line.read(1.5), 0.75);
        // assert_eq!(delay_line.read(2.0), 1.0);
        // assert_eq!(delay_line.read(2.5), 0.5);
    }

    #[test]
    fn test_delay_line_cubic_interpolate() {
        // let mut delay_line = DelayLine::new(InterpolationType::Cubic, BUFFER_LENGTH);
        // delay_line.write_and_increment(1.0);
        // delay_line.write_and_increment(0.75);
        // delay_line.write_and_increment(0.5);
        // delay_line.write_and_increment(0.25);
        // assert_eq!(delay_line.read(1.0), 0.25);
        // assert_eq!(delay_line.read(1.5), 0.375);
        // assert_eq!(delay_line.read(2.0), 0.5);
        // assert_eq!(delay_line.read(2.5), 0.625);
        // assert_eq!(delay_line.read(3.0), 0.75);
        // assert_eq!(delay_line.read(3.5), 0.625);
        // assert_eq!(delay_line.read(4.0), 0.75);
        // assert_eq!(delay_line.read(4.5), 0.515625);
        // assert_eq!(delay_line.read(5.0), 0.0);
    }
}
