use crate::utils::cubic_interpolate;
use core::fmt;
use fmt::Debug;
use fundsp::hacker32::AudioUnit;
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
    Lfo,
    SampleAndHold,
    Svf,
    Moog,
    Pulse,
    Log,
    Noise,
    Seq,
    Pan,
    Reverb,
    Delay,
    BufferTap { id: usize },
    BufferWriter { id: usize },
    BufferReader { id: usize },
    BufferRef { id: usize },
}

pub(crate) trait Node: Send + Sync {
    #[inline(always)]
    fn process(&mut self, _inputs: &[&[Frame]], _outputs: &mut [Frame]) {
        unimplemented!("This node is either a buffer reader or writer");
    }

    #[inline(always)]
    fn process_read_buffer(
        &mut self,
        _inputs: &[&[Frame]],
        _buffer: &[Frame],
        _outputs: &mut [Frame],
    ) {
        unimplemented!("This node is not a buffer reader");
    }

    #[inline(always)]
    fn process_write_buffer(&mut self, _inputs: &[&[Frame]], _buffer: &mut [Frame]) {
        unimplemented!("This node is not a buffer writer");
    }

    fn clone_box(&self) -> Box<dyn Node>;
}

#[derive(Clone)]
pub struct LFONode {
    phase: f32,
    sample_rate: u32,
}

impl LFONode {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            phase: PI,
            sample_rate,
        }
    }
}

impl Node for LFONode {
    #[inline(always)]
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        for i in 0..outputs.len() {
            let freq = inputs
                .get(0)
                .and_then(|inp| inp.get(i))
                .map(|[l, _]| *l)
                .unwrap_or(440.0_f32);

            self.phase += 2.0 * PI * freq / self.sample_rate as f32;

            if self.phase >= 2.0 * PI {
                self.phase = self.phase % (2.0 * PI);
            }

            let y = self.phase.cos();
            let y = (y + 1.0) * 0.5; // map to unipolar

            outputs[i] = [y; 2];
        }
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct SampleAndHoldNode {
    value: Frame,
    prev: f32,
}

impl SampleAndHoldNode {
    pub fn new() -> Self {
        Self {
            value: [0.0; 2],
            prev: 1.0,
        }
    }
}

impl Node for SampleAndHoldNode {
    #[inline(always)]
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        for i in 0..outputs.len() {
            let input = inputs
                .get(0)
                .and_then(|inp| inp.get(i))
                .copied()
                .unwrap_or([0.0; 2]);
            let ramp = inputs
                .get(1)
                .and_then(|inp| inp.get(i))
                .map(|[l, _]| *l)
                .unwrap_or(0.0);

            // sample when ramp wraps around (decreases)
            if ramp < self.prev {
                self.value = input;
            }

            self.prev = ramp;
            outputs[i] = self.value;
        }
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct FunDSPNode {
    node: Box<dyn AudioUnit>,
    is_stereo: bool,
    input_buffer: Vec<f32>,
}

impl FunDSPNode {
    pub fn mono(node: Box<dyn AudioUnit>) -> Self {
        let num_inputs = node.inputs();
        Self {
            node,
            is_stereo: false,
            input_buffer: vec![0.0; num_inputs],
        }
    }

    pub fn stereo(node: Box<dyn AudioUnit>) -> Self {
        let num_inputs = node.inputs();
        Self {
            node,
            is_stereo: true,
            input_buffer: vec![0.0; num_inputs],
        }
    }
}

impl Node for FunDSPNode {
    #[inline(always)]
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        let chunk_size = outputs.len();
        let num_inputs = self.node.inputs();

        if self.is_stereo {
            for i in 0..chunk_size {
                for (input_idx, input_frames) in inputs.iter().enumerate().take(num_inputs) {
                    self.input_buffer[input_idx] = input_frames.get(i).map(|f| f[0]).unwrap_or(0.0);
                }

                let mut output_sample = [0.0; 2];
                self.node.tick(&self.input_buffer, &mut output_sample);
                outputs[i] = output_sample;
            }
        } else {
            for i in 0..chunk_size {
                for (input_idx, input_frames) in inputs.iter().enumerate().take(num_inputs) {
                    self.input_buffer[input_idx] = input_frames.get(i).map(|f| f[0]).unwrap_or(0.0);
                }

                let mut output_sample = [0.0; 1];
                self.node.tick(&self.input_buffer, &mut output_sample);
                outputs[i] = [output_sample[0]; 2];
            }
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

#[derive(Clone)]
pub struct SeqNode {}

impl SeqNode {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Node for SeqNode {
    #[inline(always)]
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        for i in 0..outputs.len() {
            let ramp = inputs
                .last()
                .and_then(|inp| inp.get(i))
                .copied()
                .unwrap_or([0.0; 2]);
            let values = &inputs[0..inputs.len() - 1];

            let segment = 1.0 / values.len() as f32;
            let step = (ramp[0] / segment).floor() as usize;

            // safety check since we once got a panic here
            if step < values.len() {
                outputs[i] = values[step].get(i).copied().unwrap_or([0.0; 2]);
            } else {
                outputs[i] = values[0].get(i).copied().unwrap_or([0.0; 2]);
            }
        }
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
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        for i in 0..outputs.len() {
            let freq = inputs
                .get(0)
                .and_then(|inp| inp.get(i))
                .map(|[l, _]| *l)
                .unwrap_or(440.0);

            self.prev_phase = self.phase;
            self.phase += 2. * PI * freq / self.sample_rate as f32;

            if self.phase >= 2. * PI {
                self.phase -= 2. * PI;
                outputs[i] = [1.0, 1.0];
            } else {
                outputs[i] = [0.0, 0.0];
            }
        }
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
    fn process_read_buffer(
        &mut self,
        inputs: &[&[Frame]],
        buffer: &[Frame],
        outputs: &mut [Frame],
    ) {
        for i in 0..outputs.len() {
            let phase = inputs
                .get(0)
                .and_then(|inp| inp.get(i))
                .map(|[l, _]| *l)
                .unwrap_or(0.0);
            let phase = phase.clamp(0.0, 1.0);
            let read_pos = phase * (buffer.len() as f32 - f32::EPSILON);

            outputs[i] = cubic_interpolate(buffer, read_pos);
        }
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
    fn process_read_buffer(
        &mut self,
        inputs: &[&[Frame]],
        buffer: &[Frame],
        outputs: &mut [Frame],
    ) {
        for i in 0..outputs.len() {
            let offset = inputs
                .get(0)
                .and_then(|inp| inp.get(i))
                .map(|[l, _]| *l)
                .unwrap_or(0.0);
            let read_pos_f = self.write_pos as f32 - offset;
            let buffer_len = buffer.len() as f32;
            let read_pos = (read_pos_f % buffer_len + buffer_len) % buffer_len;
            outputs[i] = cubic_interpolate(&buffer, read_pos);

            self.write_pos = (self.write_pos + 1) % buffer.len();
        }
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
    fn process_write_buffer(&mut self, inputs: &[&[Frame]], buffer: &mut [Frame]) {
        for i in 0..inputs.get(0).map(|inp| inp.len()).unwrap_or(0) {
            let input = inputs
                .get(0)
                .and_then(|inp| inp.get(i))
                .copied()
                .unwrap_or([0.0; 2]);
            buffer[self.write_pos] = input;
            self.write_pos = (self.write_pos + 1) % buffer.len();
        }
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
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        for i in 0..outputs.len() {
            let input = inputs
                .get(0)
                .and_then(|inp| inp.get(i))
                .copied()
                .unwrap_or([0.0; 2]);
            self.buffer[self.write_pos] = input;

            let delay = inputs
                .get(1)
                .and_then(|inp| inp.get(i))
                .map(|[l, _]| *l)
                .unwrap_or(0.0);
            // clamp delay to buffer size
            let delay = delay.max(0.0).min((Self::BUFFER_SIZE - 1) as f32);

            let read_pos = (self.write_pos as f32 - delay + Self::BUFFER_SIZE as f32)
                % Self::BUFFER_SIZE as f32;

            outputs[i] = cubic_interpolate(&self.buffer, read_pos);

            self.write_pos = (self.write_pos + 1) % Self::BUFFER_SIZE;
        }
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}
