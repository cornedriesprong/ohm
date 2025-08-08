use crate::utils::{cubic_interpolate, freq_to_period};
use core::fmt;
use fmt::Debug;
use fundsp::hacker32::AudioUnit;
use rand::random;
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
    BufferTap { id: usize },
    BufferWriter { id: usize },
    BufferReader { id: usize },
    BufferRef { id: usize },
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
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let phase = inputs.last().expect("seq: missing trigger input");
        let values = &inputs[0..inputs.len() - 1];

        let segment = 1.0 / values.len() as f32;
        self.step = (phase[0] / segment).floor() as usize;

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
            sample_rate,
        }
    }
}

impl Node for PulseNode {
    #[inline(always)]
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

            let noise = if random::<bool>() { 1.0 } else { -1.0 };
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
        window = (2.0f32).max(window * self.pitch_track);
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
pub struct BufReaderNode {}

impl BufReaderNode {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Node for BufReaderNode {
    fn tick_read_buffer(&mut self, inputs: &[Frame], buffer: &[Frame]) -> Frame {
        let phase = inputs.get(0).expect("missing phase")[0];
        let phase = phase.clamp(0.0, 1.0);
        let read_pos = phase * (buffer.len() as f32 - f32::EPSILON);

        cubic_interpolate(buffer, read_pos)
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

// a buf tap node is stateful in that it maintains it's own interal write position
// which makes it usable as a delay line
#[derive(Clone)]
pub struct BufTapNode {
    write_pos: usize,
}

impl BufTapNode {
    pub(crate) fn new() -> Self {
        Self { write_pos: 0 }
    }
}

impl Node for BufTapNode {
    fn tick_read_buffer(&mut self, inputs: &[Frame], buffer: &[Frame]) -> Frame {
        let offset = inputs.get(0).expect("missing offset")[0];
        let read_pos_f = self.write_pos as f32 - offset;
        let buffer_len = buffer.len() as f32;
        let read_pos = (read_pos_f % buffer_len + buffer_len) % buffer_len;
        let y = cubic_interpolate(&buffer, read_pos);

        self.write_pos = (self.write_pos + 1) % buffer.len();

        y
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct BufWriterNode {
    write_pos: usize,
}

impl BufWriterNode {
    pub(crate) fn new() -> Self {
        Self { write_pos: 0 }
    }
}

impl Node for BufWriterNode {
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
pub(crate) struct DelayNode {
    buffer: [Frame; Self::BUFFER_SIZE],
    write_pos: usize,
}

impl DelayNode {
    pub const BUFFER_SIZE: usize = 48000;

    pub(crate) fn new() -> Self {
        Self {
            buffer: [[0.0; 2]; Self::BUFFER_SIZE],
            write_pos: 0,
        }
    }
}

impl Node for DelayNode {
    #[inline(always)]
    fn tick(&mut self, inputs: &[Frame]) -> Frame {
        let input = inputs.get(0).expect("pipe: missing input");
        self.buffer[self.write_pos] = *input;

        let delay = inputs.get(1).expect("pipe: missing delay")[0];
        // clamp delay to buffer size
        let delay = delay.max(0.0).min((Self::BUFFER_SIZE - 1) as f32);
        // TODO: skip interpolation if delay is < 0

        let read_pos =
            (self.write_pos as f32 - delay + Self::BUFFER_SIZE as f32) % Self::BUFFER_SIZE as f32;

        let output = cubic_interpolate(&self.buffer, read_pos);

        self.write_pos = (self.write_pos + 1) % Self::BUFFER_SIZE;

        output
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}
