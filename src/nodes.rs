use crate::utils::cubic_interpolate;
use fundsp::hacker32::AudioUnit;
use petgraph::graph::NodeIndex;
use std::f32::consts::PI;

pub type Frame = [f32; 2];

macro_rules! binary_op {
    ($inputs:expr, $outputs:expr, $op:tt) => {{
        let lhs = $inputs[0];
        let rhs = $inputs[1];
        for (out, (l, r)) in $outputs.iter_mut().zip(lhs.iter().zip(rhs.iter())) {
            *out = [l[0] $op r[0], l[1] $op r[1]];
        }
    }};
}

macro_rules! comparison_op {
    ($inputs:expr, $outputs:expr, $op:tt) => {{
        let lhs = $inputs[0];
        let rhs = $inputs[1];
        for (out, (l, r)) in $outputs.iter_mut().zip(lhs.iter().zip(rhs.iter())) {
            *out = [(l[0] $op r[0]) as u32 as f32, (l[1] $op r[1]) as u32 as f32];
        }
    }};
}

#[derive(Clone)]
pub(crate) enum Node {
    Constant(f32),
    Ramp {
        phase: f32,
        sample_rate: u32,
    },
    Lfo {
        phase: f32,
        sample_rate: u32,
    },
    Sum,
    Diff,
    Gain,
    Divide,
    Wrap,
    Power,
    Greater,
    Less,
    Equal,
    SampleAndHold {
        value: Frame,
        prev: f32,
    },
    FunDSP {
        audio_unit: Box<dyn AudioUnit>,
        is_stereo: bool,
        input_buffer: Vec<f32>,
    },
    Mix,
    Seq,
    Delay {
        buffer: Box<[Frame; 48000]>,
        write_pos: usize,
    },
    BufferTap {
        id: NodeIndex,
        write_pos: usize,
    },
    BufferWriter {
        id: NodeIndex,
        write_pos: usize,
    },
    BufferReader {
        id: NodeIndex,
    },
    BufferRef,
    Log,
}

impl Node {
    #[inline(always)]
    pub(crate) fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        match self {
            Node::Constant(value) => {
                let frame = [*value; 2];
                outputs.fill(frame);
            }
            Node::Ramp { phase, sample_rate } => {
                let freq = inputs[0];
                let sample_rate = *sample_rate as f32;
                let two_pi = 2.0 * PI;
                let inc = two_pi / sample_rate;

                for (out, freq_frame) in outputs.iter_mut().zip(freq.iter()) {
                    let freq = freq_frame[0];
                    *phase += freq * inc;
                    *phase %= two_pi;
                    let y = *phase / two_pi;
                    *out = [y; 2];
                }
            }
            Node::Lfo { phase, sample_rate } => {
                let freq_input = inputs[0];
                let sample_rate = *sample_rate as f32;
                let two_pi = 2.0 * PI;
                let inc = two_pi / sample_rate;

                for (out, freq_frame) in outputs.iter_mut().zip(freq_input.iter()) {
                    let freq = freq_frame[0];
                    *phase += freq * inc;
                    *phase %= two_pi;
                    let y = (phase.cos() + 1.0) * 0.5;
                    *out = [y; 2];
                }
            }
            Node::Sum => binary_op!(inputs, outputs, +),
            Node::Diff => binary_op!(inputs, outputs, -),
            Node::Gain => binary_op!(inputs, outputs, *),
            Node::Divide => binary_op!(inputs, outputs, /),
            Node::Wrap => binary_op!(inputs, outputs, %),
            Node::Power => {
                let lhs = inputs[0];
                let rhs = inputs[1];
                for (out, (l, r)) in outputs.iter_mut().zip(lhs.iter().zip(rhs.iter())) {
                    *out = [l[0].powf(r[0]), l[1].powf(r[1])];
                }
            }
            Node::Greater => comparison_op!(inputs, outputs, >),
            Node::Less => comparison_op!(inputs, outputs, <),
            Node::Equal => comparison_op!(inputs, outputs, ==),
            Node::SampleAndHold { value, prev, .. } => {
                let input = inputs[0];
                let ramp = inputs[1];
                for i in 0..outputs.len() {
                    let ramp_val = ramp.get(i).map(|f| f[0]).unwrap_or(0.0);
                    if (ramp_val - *prev).abs() > 0.5 {
                        *value = input.get(i).copied().unwrap_or([0.0; 2]);
                    }
                    *prev = ramp_val;
                    outputs[i] = *value;
                }
            }
            Node::FunDSP {
                audio_unit: node,
                is_stereo,
                input_buffer,
            } => {
                const EMPTY: &[Frame] = &[];
                let num_inputs = node.inputs();
                if *is_stereo {
                    for i in 0..outputs.len() {
                        for input_idx in 0..num_inputs {
                            let input_frames = inputs.get(input_idx).unwrap_or(&EMPTY);
                            input_buffer[input_idx] =
                                input_frames.get(i).map(|f| f[0]).unwrap_or(0.0);
                        }
                        let mut output_sample = [0.0; 2];
                        node.tick(input_buffer, &mut output_sample);
                        outputs[i] = output_sample;
                    }
                } else {
                    for i in 0..outputs.len() {
                        for input_idx in 0..num_inputs {
                            let input_frames = inputs.get(input_idx).unwrap_or(&EMPTY);
                            input_buffer[input_idx] =
                                input_frames.get(i).map(|f| f[0]).unwrap_or(0.0);
                        }
                        let mut output_sample = [0.0; 1];
                        node.tick(input_buffer, &mut output_sample);
                        outputs[i] = [output_sample[0]; 2];
                    }
                }
            }
            Node::Mix => {
                let inv_n = 1.0 / inputs.len() as f32;
                for (i, frame) in outputs.iter_mut().enumerate() {
                    let mut l = 0.0f32;
                    let mut r = 0.0f32;
                    for input in inputs.iter() {
                        let src = input.get(i).copied().unwrap_or([0.0; 2]);
                        l += src[0];
                        r += src[1];
                    }
                    frame[0] = l * inv_n;
                    frame[1] = r * inv_n;
                }
            }
            Node::Seq => {
                let ramp_input = inputs[0];
                let num_steps = inputs.len() - 1;
                for (i, (out, ramp_frame)) in outputs.iter_mut().zip(ramp_input.iter()).enumerate()
                {
                    let ramp = ramp_frame[0];
                    let step = (ramp * num_steps as f32).floor() as usize;
                    let step_input = inputs[step + 1];
                    *out = step_input.get(i).copied().unwrap_or([0.0; 2]);
                }
            }
            Node::Delay { buffer, write_pos } => {
                const BUFFER_SIZE: usize = 48000;
                let signal = inputs[0];
                let delay_input = inputs[1];
                let buffer_size_f32 = BUFFER_SIZE as f32;
                let max_delay = (BUFFER_SIZE - 1) as f32;

                for (i, out) in outputs.iter_mut().enumerate() {
                    let sig = signal.get(i).copied().unwrap_or([0.0; 2]);
                    buffer[*write_pos] = sig;

                    let delay = delay_input
                        .get(i)
                        .map(|f| f[0])
                        .unwrap_or(0.0)
                        .clamp(0.0, max_delay);
                    let read_pos = (*write_pos as f32 - delay + buffer_size_f32) % buffer_size_f32;

                    *out = cubic_interpolate(&buffer[..], read_pos);

                    *write_pos = (*write_pos + 1) % BUFFER_SIZE;
                }
            }
            Node::Log => {
                for (i, out) in outputs.iter_mut().enumerate() {
                    let input = inputs[0];
                    let frame = input.get(i).copied().unwrap_or([0.0; 2]);
                    println!("[{}]: L = {}, R = {}", i, frame[0], frame[1]);
                    *out = frame;
                }
            }
            Node::BufferTap { .. } => {}
            Node::BufferWriter { .. } => {}
            Node::BufferReader { .. } => {}
            Node::BufferRef { .. } => {}
        }
    }

    pub(crate) fn get_id(&self) -> String {
        match self {
            Node::Constant(v) => format!("Constant({})", v),
            Node::Ramp { .. } => "Ramp".to_string(),
            Node::Lfo { .. } => "Lfo".to_string(),
            Node::Sum => "Sum".to_string(),
            Node::Diff => "Diff".to_string(),
            Node::Gain => "Gain".to_string(),
            Node::Divide => "Divide".to_string(),
            Node::Wrap => "Wrap".to_string(),
            Node::Power => "Power".to_string(),
            Node::Greater => "Greater".to_string(),
            Node::Less => "Less".to_string(),
            Node::Equal => "Equal".to_string(),
            Node::SampleAndHold { .. } => "SampleAndHold".to_string(),
            Node::FunDSP { .. } => "FunDSP".to_string(),
            Node::Mix => "Mix".to_string(),
            Node::Seq => "Seq".to_string(),
            Node::Delay { .. } => "Delay".to_string(),
            Node::BufferTap { id, .. } => format!("BufferTap({:?})", id),
            Node::BufferWriter { id, .. } => format!("BufferWriter({:?})", id),
            Node::BufferReader { id } => format!("BufferReader({:?})", id),
            Node::BufferRef => format!("BufferRef"),
            Node::Log => "Log".to_string(),
        }
    }

    #[inline(always)]
    pub(crate) fn transfer_state(&mut self, old: &Node) {
        match (self, old) {
            (Node::Ramp { phase: new, .. }, Node::Ramp { phase: old, .. }) => *new = *old,
            (Node::Lfo { phase: new, .. }, Node::Lfo { phase: old, .. }) => *new = *old,
            (
                Node::SampleAndHold {
                    value: new_val,
                    prev: new_prev,
                },
                Node::SampleAndHold {
                    value: old_val,
                    prev: old_prev,
                },
            ) => {
                *new_val = *old_val;
                *new_prev = *old_prev;
            }
            (
                Node::Delay {
                    buffer: new_buf,
                    write_pos: new_pos,
                },
                Node::Delay {
                    buffer: old_buf,
                    write_pos: old_pos,
                },
            ) => {
                **new_buf = **old_buf;
                *new_pos = *old_pos;
            }
            (
                Node::FunDSP {
                    audio_unit: new_unit,
                    is_stereo: new_stereo,
                    input_buffer: new_buf,
                },
                Node::FunDSP {
                    audio_unit: old_unit,
                    is_stereo: old_stereo,
                    ..
                },
            ) => {
                if *new_stereo == *old_stereo
                    && new_unit.inputs() == old_unit.inputs()
                    && new_unit.outputs() == old_unit.outputs()
                    && new_unit.get_id() == old_unit.get_id()
                {
                    *new_unit = old_unit.clone();
                    new_buf.resize(new_unit.inputs(), 0.0);
                }
            }
            _ => {}
        }
    }

    #[inline(always)]
    pub(crate) fn process_read_buffer(
        &mut self,
        inputs: &[&[Frame]],
        buffer: &[Frame],
        outputs: &mut [Frame],
    ) {
        match self {
            Node::BufferReader { .. } => {
                for (i, out) in outputs.iter_mut().enumerate() {
                    let phase = inputs[0][i][0];
                    let read_pos = phase * (buffer.len() as f32 - f32::EPSILON);
                    *out = cubic_interpolate(buffer, read_pos);
                }
            }
            // TODO: buffer tap
            _ => unimplemented!(),
        }
    }

    #[inline(always)]
    pub(crate) fn process_write_buffer(&mut self, inputs: &[&[Frame]], buffer: &mut [Frame]) {
        match self {
            Node::BufferWriter { mut write_pos, .. } => {
                let buffer_len = buffer.len();
                for (i, frame) in &mut buffer.iter_mut().enumerate() {
                    let input = inputs[0][i];
                    *frame = input;
                    write_pos = (write_pos + 1) % buffer_len;
                }
            }
            _ => unimplemented!(),
        }
    }
}
