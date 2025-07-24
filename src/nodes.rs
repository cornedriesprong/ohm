use crate::dsp::delay::Delay;
use crate::utils::freq_to_period;
use core::fmt;
use fmt::Debug;
use fundsp::hacker32::AudioUnit;
use rand::Rng;
use rtsan_standalone::nonblocking;
use std::f32::consts::PI;
use std::hash::Hash;

pub type Frame = [f32; 2];

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum NodeKind {
    Sin,
    Sqr,
    Saw,
    Tri,
    Ramp,
    Lp,
    Bp,
    Hp,
    Pulse,
    Print,
    Noise,
    Env,
    Seq,
    Pan,
    Pluck,
    Reverb,
    Delay,
    Moog,
    Pipe,
    BufferWriter { name: String },
    BufferReader { name: String },
}

pub(crate) trait Node: Send + Sync {
    fn tick(&mut self, _: &[Frame]) -> Frame {
        unimplemented!("This node is either a buffer reader or writer");
    }

    fn tick_read_buffer(&mut self, _: &[Frame], _: &[Frame]) -> Frame {
        unimplemented!("This node is not a buffer reader");
    }

    fn tick_write_buffer(&mut self, _: &[Frame], _: &mut [Frame]) {
        unimplemented!("This node is not a buffer writer");
    }

    fn clone_box(&self) -> Box<dyn Node>;
}

#[derive(Clone)]
pub struct FunDSPNode {
    node: Box<dyn AudioUnit>,
    is_stereo: bool,
}

impl FunDSPNode {
    pub fn mono(node: Box<dyn AudioUnit>) -> Self {
        Self {
            node,
            is_stereo: false,
        }
    }

    pub fn stereo(node: Box<dyn AudioUnit>) -> Self {
        Self {
            node,
            is_stereo: true,
        }
    }
}

impl Node for FunDSPNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        if self.is_stereo {
            let mut output = [0.0; 2];
            self.node.tick(inputs[0].as_slice(), &mut output);
            output
        } else {
            let input: Vec<f32> = inputs.iter().map(|[l, _]| *l).collect();
            let mut output = [0.0; 1];
            self.node.tick(input.as_slice(), &mut output);
            [output[0]; 2]
        }
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Node> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone, Debug)]
pub struct EnvSegment {
    pub duration: usize, // Duration in samples or ms (interpreted as needed)
    pub target: f32,     // Target value at end of this segment
}

#[derive(Clone)]
pub struct EnvNode {
    pub(crate) current_idx: usize,
    pub(crate) value: f32,
    pub(crate) time: usize,
    pub(crate) active: bool,
    pub(crate) prev_trig: f32,
}

impl EnvNode {
    pub(crate) fn new() -> Self {
        Self {
            current_idx: 0,
            value: 0.0,
            time: 0,
            active: false,
            prev_trig: 0.0,
        }
    }

    fn start(&mut self) {
        self.active = true;
        self.current_idx = 0;
        self.time = 0;
    }

    fn advance(&mut self, segments: &[EnvSegment]) {
        self.current_idx += 1;
        self.time = 0;

        if self.current_idx >= segments.len() {
            self.active = false;
            self.value = 0.0;
        }
    }

    fn interpolate_segment(&self, segments: &[EnvSegment]) -> f32 {
        let segment = &segments[self.current_idx];

        if segment.duration == 0 {
            return segment.target;
        }

        let pow = 3.0; // curve
        let raw_t = (self.time as f32 / segment.duration as f32).clamp(0.0, 1.0);

        // reverse the curve if we're going downward
        let curved_t = if segment.target < self.value {
            // ease out
            1.0 - (1.0 - raw_t).powf(pow)
        } else {
            // ease in
            raw_t.powf(pow)
        };

        let prev = if self.current_idx == 0 {
            0.0
        } else {
            segments[self.current_idx - 1].target
        };

        prev + curved_t * (segment.target - prev)
    }
}

impl Node for EnvNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let trig = inputs.last().expect("env: missing trigger input")[0];
        let segments = &inputs[0..inputs.len() - 1]
            .chunks_exact(2)
            .map(|pair| EnvSegment {
                target: pair[0][0],
                duration: pair[1][0] as usize,
            })
            .collect::<Vec<_>>();

        if trig > 0.0 && self.prev_trig <= 0.0 {
            self.start();
        }

        self.prev_trig = trig;

        if !self.active {
            return [0.0, 0.0];
        }

        let segment = &segments[self.current_idx];
        self.value = self.interpolate_segment(&segments);
        self.time += 1;

        if self.time >= segment.duration {
            self.advance(&segments);
        }

        [self.value, self.value]
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct SeqNode {
    pub(crate) step: usize,
}

impl SeqNode {
    pub(crate) fn new() -> Self {
        Self { step: 0 }
    }
}

impl Node for SeqNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let ramp = inputs.last().expect("seq: missing trigger input");
        let values = &inputs[0..inputs.len() - 1];

        let segment = 1.0 / values.len() as f32;
        self.step = (ramp[0] / segment).floor() as usize;

        values[self.step]
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

pub struct PulseNode {
    phase: f32,
    prev_phase: f32,
    sample_rate: u32,
}

impl Clone for PulseNode {
    fn clone(&self) -> Self {
        Self {
            phase: 0.0,
            prev_phase: 0.0,
            sample_rate: self.sample_rate,
        }
    }
}

impl PulseNode {
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            phase: 0.,
            prev_phase: 0.,
            sample_rate: sample_rate,
        }
    }
}

impl Node for PulseNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let freq = inputs.get(0).expect("pulse: missing freq input")[0];

        self.prev_phase = self.phase;
        self.phase += 2. * PI * freq / self.sample_rate as f32;

        if self.phase >= 2. * PI {
            self.phase -= 2. * PI;
            [1.0, 1.0]
        } else {
            [0.0, 0.0]
        }
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
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
    buffer: [f32; Self::BUFFER_SIZE],
    period: f32,
    read_pos: usize,
    pitch_track: f32,
    is_stopped: bool,
    sample_rate: u32,
}

impl PluckNode {
    const BUFFER_SIZE: usize = 2048; // 2048 samples

    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            // mode: Mode::String,
            buffer: [0.0; Self::BUFFER_SIZE],
            period: 1.0,
            read_pos: 0,
            pitch_track: 0.0,
            is_stopped: true,
            sample_rate,
        }
    }

    fn play(&mut self, freq: f32, tone: f32) {
        self.is_stopped = false;
        self.period = freq_to_period(self.sample_rate as f32, freq);
        self.read_pos = 0;

        self.pitch_track = (5.0 as f32).max(self.period / 7.0);
        assert!(self.period < Self::BUFFER_SIZE as f32);

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
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let freq = inputs.get(0).map_or(0.0, |arr| arr[0]);
        let tone = inputs.get(1).map_or(0.0, |arr| arr[0]);
        let damping = inputs.get(2).map_or(0.0, |arr| arr[0]);
        let trig = inputs.get(3).map_or(0.0, |arr| arr[0]);

        if trig > 0.0 {
            self.play(freq, tone);
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

        let output = self.buffer[self.read_pos];
        [output, output]
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct DelayNode {
    delay: Delay,
}

impl DelayNode {
    pub(crate) fn new() -> Self {
        Self {
            delay: Delay::new(15000.0, 0.5),
        }
    }
}

impl Node for DelayNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let input = inputs.get(0).expect("delay: missing input")[0];
        let output = self.delay.process(input);
        [output, output]
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct WavReaderNode {}

impl WavReaderNode {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Node for WavReaderNode {
    fn tick_read_buffer(&mut self, inputs: &[Frame], buffer: &[Frame]) -> Frame {
        let phase = inputs.get(0).expect("sampler: missing phase")[0];
        let len = buffer.len() as isize;
        let index = phase * len as f32;
        let index_floor = index.floor() as isize;
        let t = index - index.floor(); // fractional part

        // Get four neighboring samples with wrapping.
        let idx0 = (index_floor - 1).rem_euclid(len) as usize;
        let idx1 = index_floor.rem_euclid(len) as usize;
        let idx2 = (index_floor + 1).rem_euclid(len) as usize;
        let idx3 = (index_floor + 2).rem_euclid(len) as usize;

        let p0 = buffer[idx0][0];
        let p1 = buffer[idx1][0];
        let p2 = buffer[idx2][0];
        let p3 = buffer[idx3][0];

        // Catmull-Rom interpolation:
        // \\[
        // y(t) = 0.5 \\times (2P_1 + (-P_0+P_2)t + (2P_0-5P_1+4P_2-P_3)t^2 + (-P_0+3P_1-3P_2+P_3)t^3)
        // \\]
        let t2 = t * t;
        let t3 = t2 * t;
        let sample = 0.5
            * (2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);

        [sample, sample]
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct WavWriterNode {
    write_pos: usize,
}

impl WavWriterNode {
    pub(crate) fn new() -> Self {
        Self { write_pos: 0 }
    }
}

impl Node for WavWriterNode {
    fn tick_write_buffer(&mut self, inputs: &[Frame], buffer: &mut [Frame]) {
        let input = inputs.get(0).expect("wav writer: missing input");
        buffer[self.write_pos] = *input;
        self.write_pos = (self.write_pos + 1) % buffer.len();
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

// delay line
#[derive(Clone)]
pub(crate) struct PipeNode {
    buffer: [Frame; Self::BUFFER_SIZE],
    read_pos: usize,
    write_pos: usize,
}

impl PipeNode {
    pub const BUFFER_SIZE: usize = 48000;

    pub(crate) fn new() -> Self {
        Self {
            buffer: [[0.0, 0.0]; Self::BUFFER_SIZE],
            read_pos: 0,
            write_pos: 0,
        }
    }
}

impl Node for PipeNode {
    #[inline(always)]
    #[nonblocking]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let input = inputs.get(0).expect("pipe: missing input");
        self.buffer[self.write_pos] = *input;
        let delay = inputs.get(1).expect("pipe: missing delay")[0];
        self.read_pos = (self.read_pos + 1) % Self::BUFFER_SIZE;
        self.write_pos = (self.write_pos + 1) % Self::BUFFER_SIZE;

        // Read the value at the current read position
        self.buffer[(self.read_pos + delay as usize) % Self::BUFFER_SIZE]
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}
