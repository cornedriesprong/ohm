use crate::container::Arena;
use crate::utils::cubic_interpolate;
use fundsp::hacker32::AudioUnit;
use std::any::Any;
use std::f32::consts::PI;

pub type Frame = [f32; 2];

macro_rules! define_osc_node {
    ($name:ident, $phase_to_output:expr) => {
        pub struct $name {
            inputs: Vec<usize>,
            phase: f32,
            sample_rate: u32,
            buffers: Vec<Vec<Frame>>,
        }

        impl $name {
            pub(crate) fn new(inputs: Vec<usize>, sample_rate: u32) -> Self {
                let num_inputs = inputs.len();
                Self {
                    inputs,
                    phase: 0.0,
                    sample_rate,
                    buffers: vec![Vec::new(); num_inputs],
                }
            }
        }

        impl Node for $name {
            #[inline(always)]
            fn process(&mut self, arena: &mut Arena, outputs: &mut [Frame]) {
                render_inputs(arena, &self.inputs, &mut self.buffers, outputs.len());

                let freq_input = &self.buffers[0];
                let sample_rate_f32 = self.sample_rate as f32;
                let max_freq = sample_rate_f32 * 0.5;
                let two_pi = 2.0 * PI;
                let phase_increment_scale = two_pi / sample_rate_f32;

                for (out, freq_frame) in outputs.iter_mut().zip(freq_input.iter()) {
                    let freq = freq_frame[0].clamp(0.0, max_freq);

                    self.phase += freq * phase_increment_scale;
                    self.phase = self.phase % two_pi;

                    let y = $phase_to_output(self.phase, two_pi);

                    *out = [y; 2];
                }
            }

            fn get_inputs(&self) -> &[usize] {
                &self.inputs
            }

            fn transfer_state(&mut self, old: &dyn Node) {
                if let Some(old) = (old as &dyn Any).downcast_ref::<$name>() {
                    self.phase = old.phase;
                }
            }
        }
    };
}

macro_rules! define_binary_op_node {
    ($name:ident, $op:expr) => {
        pub struct $name {
            inputs: Vec<usize>,
            buffers: Vec<Vec<Frame>>,
        }

        impl $name {
            pub(crate) fn new(inputs: Vec<usize>) -> Self {
                let num_inputs = inputs.len();
                Self {
                    inputs,
                    buffers: vec![Vec::new(); num_inputs],
                }
            }
        }

        impl Node for $name {
            #[inline(always)]
            fn process(&mut self, arena: &mut Arena, outputs: &mut [Frame]) {
                render_inputs(arena, &self.inputs, &mut self.buffers, outputs.len());

                let lhs = &self.buffers[0];
                let rhs = &self.buffers[1];

                for (out, (l, r)) in outputs.iter_mut().zip(lhs.iter().zip(rhs.iter())) {
                    *out = $op(*l, *r);
                }
            }

            fn get_inputs(&self) -> &[usize] {
                &self.inputs
            }
        }
    };
}

// helper function to render node inputs
#[inline(always)]
fn render_inputs(arena: &mut Arena, inputs: &[usize], buffers: &mut Vec<Vec<Frame>>, len: usize) {
    if buffers.len() != inputs.len() {
        buffers.resize_with(inputs.len(), Vec::new);
    }

    for (i, &input_id) in inputs.iter().enumerate() {
        if buffers[i].len() < len {
            buffers[i].resize(len, [0.0; 2]);
        }

        unsafe {
            let arena_ptr = arena as *mut Arena;
            (*arena_ptr)
                .get_mut(input_id)
                .process(&mut *arena_ptr, &mut buffers[i][..len]);
        }
    }
}

pub(crate) trait Node: Send + Sync + Any {
    #[inline(always)]
    fn process(&mut self, _arena: &mut Arena, _outputs: &mut [Frame]) {}

    fn get_id(&self) -> String {
        std::any::type_name::<Self>().to_string()
    }

    fn get_inputs(&self) -> &[usize] {
        &[]
    }

    fn transfer_state(&mut self, _old: &dyn Node) {
        // stateless nodes don't transfer state, so we default to no-op
    }

    fn get_buf_name(&self) -> Option<&str> {
        None
    }
}

define_binary_op_node!(MixNode, |lhs: Frame, rhs: Frame| {
    [lhs[0] + rhs[0], lhs[1] + rhs[0]]
});

define_binary_op_node!(SubtractNode, |lhs: Frame, rhs: Frame| {
    [lhs[0] - rhs[0], lhs[1] - rhs[0]]
});

define_binary_op_node!(GainNode, |lhs: Frame, rhs: Frame| {
    [lhs[0] * rhs[0], lhs[1] * rhs[0]]
});

define_binary_op_node!(DivideNode, |lhs: Frame, rhs: Frame| {
    [lhs[0] / rhs[0], lhs[1] / rhs[0]]
});

define_binary_op_node!(WrapNode, |lhs: Frame, rhs: Frame| {
    [lhs[0] % rhs[0], lhs[1] % rhs[0]]
});

define_binary_op_node!(PowerNode, |lhs: Frame, rhs: Frame| {
    [lhs[0].powf(rhs[0]), lhs[1].powf(rhs[0])]
});

define_binary_op_node!(GreaterNode, |lhs: Frame, rhs: Frame| {
    [
        (lhs[0] > rhs[0]) as u32 as f32,
        (lhs[1] > rhs[0]) as u32 as f32,
    ]
});

define_binary_op_node!(LessNode, |lhs: Frame, rhs: Frame| {
    [
        (lhs[0] < rhs[0]) as u32 as f32,
        (lhs[1] < rhs[0]) as u32 as f32,
    ]
});

define_binary_op_node!(EqualNode, |lhs: Frame, rhs: Frame| {
    [
        (lhs[0] == rhs[0]) as u32 as f32,
        (lhs[1] == rhs[0]) as u32 as f32,
    ]
});

define_osc_node!(RampNode, |phase: f32, two_pi: f32| { phase / two_pi });

define_osc_node!(LFONode, |phase: f32, _two_pi: f32| {
    (phase.cos() + 1.0) * 0.5
});

pub struct SampleAndHoldNode {
    inputs: Vec<usize>,
    value: Frame,
    prev: f32,
    buffers: Vec<Vec<Frame>>,
}

impl SampleAndHoldNode {
    pub(crate) fn new(inputs: Vec<usize>, sample_rate: u32) -> Self {
        _ = sample_rate;
        Self {
            inputs,
            value: [0.0; 2],
            prev: 1.0,
            buffers: vec![vec![], vec![]],
        }
    }
}

impl Node for SampleAndHoldNode {
    fn process(&mut self, arena: &mut Arena, outputs: &mut [Frame]) {
        render_inputs(arena, &self.inputs, &mut self.buffers, outputs.len());

        let signal = &self.buffers[0];
        let ramp = &self.buffers[1];

        for i in 0..outputs.len() {
            let ramp_val = ramp[i][0];
            if (ramp_val - self.prev).abs() > 0.5 {
                self.value = signal[i];
            }
            self.prev = ramp_val;
            outputs[i] = self.value;
        }
    }

    fn get_inputs(&self) -> &[usize] {
        &self.inputs
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old) = (old as &dyn Any).downcast_ref::<SampleAndHoldNode>() {
            self.value = old.value;
            self.prev = old.prev;
        }
    }
}

pub struct FunDSPNode {
    inputs: Vec<usize>,
    node: Box<dyn AudioUnit>,
    is_stereo: bool,
    input_buffer: Vec<f32>,
    buffers: Vec<Vec<Frame>>,
}

impl FunDSPNode {
    pub fn mono(inputs: Vec<usize>, node: Box<dyn AudioUnit>) -> Self {
        let num_inputs = inputs.len();
        let num_audio_inputs = node.inputs();
        Self {
            inputs,
            node,
            is_stereo: false,
            input_buffer: vec![0.0; num_audio_inputs],
            buffers: vec![Vec::new(); num_inputs],
        }
    }

    pub fn stereo(inputs: Vec<usize>, node: Box<dyn AudioUnit>) -> Self {
        let num_inputs = inputs.len();
        let num_audio_inputs = node.inputs();
        Self {
            inputs,
            node,
            is_stereo: true,
            input_buffer: vec![0.0; num_audio_inputs],
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for FunDSPNode {
    #[inline(always)]
    fn process(&mut self, arena: &mut Arena, outputs: &mut [Frame]) {
        let chunk_size = outputs.len();
        render_inputs(arena, &self.inputs, &mut self.buffers, chunk_size);

        let num_inputs = self.node.inputs();

        if self.is_stereo {
            for i in 0..chunk_size {
                for (input_idx, input_frames) in self.buffers.iter().enumerate().take(num_inputs) {
                    self.input_buffer[input_idx] = input_frames[i][0];
                }

                let mut output_sample = [0.0; 2];
                self.node.tick(&self.input_buffer, &mut output_sample);
                outputs[i] = output_sample;
            }
        } else {
            for i in 0..chunk_size {
                for (input_idx, input_frames) in self.buffers.iter().enumerate().take(num_inputs) {
                    self.input_buffer[input_idx] = input_frames[i][0];
                }

                let mut output_sample = [0.0; 1];
                self.node.tick(&self.input_buffer, &mut output_sample);
                outputs[i] = [output_sample[0]; 2];
            }
        }
    }

    fn get_inputs(&self) -> &[usize] {
        &self.inputs
    }
}

pub struct SeqNode {
    inputs: Vec<usize>,
    buffers: Vec<Vec<Frame>>,
}

impl SeqNode {
    pub(crate) fn new(inputs: Vec<usize>) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for SeqNode {
    #[inline(always)]
    fn process(&mut self, arena: &mut Arena, outputs: &mut [Frame]) {
        render_inputs(arena, &self.inputs, &mut self.buffers, outputs.len());

        let num_steps = self.buffers.len() - 1;
        let ramp_input = &self.buffers[0];

        for (i, (out, ramp_frame)) in outputs.iter_mut().zip(ramp_input.iter()).enumerate() {
            let ramp = ramp_frame[0];
            let step = (ramp * num_steps as f32).floor() as usize;
            *out = self.buffers[step + 1][i];
        }
    }

    fn get_inputs(&self) -> &[usize] {
        &self.inputs
    }
}

pub struct BufRefNode {
    name: String,
}

impl BufRefNode {
    pub(crate) fn new(name: String) -> Self {
        Self { name }
    }
}

impl Node for BufRefNode {
    fn get_id(&self) -> String {
        format!("BufRefNode {}", self.name)
    }

    fn get_buf_name(&self) -> Option<&str> {
        Some(&self.name)
    }
}

pub struct BufReaderNode {
    buf_name: String,
    inputs: Vec<usize>,
    buffers: Vec<Vec<Frame>>,
}

impl BufReaderNode {
    pub(crate) fn new(buf_name: String, inputs: Vec<usize>) -> Self {
        Self {
            buf_name,
            inputs,
            buffers: vec![Vec::new(); 1],
        }
    }
}

impl Node for BufReaderNode {
    fn process(&mut self, arena: &mut Arena, outputs: &mut [Frame]) {
        render_inputs(arena, &self.inputs, &mut self.buffers, outputs.len());

        let buffer = if let Some(buffer) = arena.get_buffer(&self.buf_name) {
            buffer
        } else {
            return;
        };

        let phase = self.buffers[0].as_slice();

        for (i, out) in outputs.iter_mut().enumerate() {
            let phase = phase[i][0];
            let read_pos = phase * (buffer.len() as f32 - f32::EPSILON);
            *out = cubic_interpolate(buffer, read_pos);
        }
    }

    fn get_inputs(&self) -> &[usize] {
        &self.inputs
    }
}

// a buf tap node is stateful in that it maintains it's own interal write position
// which makes it usable as a delay line
pub struct BufTapNode {
    buf_name: String,
    inputs: Vec<usize>,
    write_pos: usize,
    buffers: Vec<Vec<Frame>>,
}

impl BufTapNode {
    pub(crate) fn new(buf_name: String, inputs: Vec<usize>) -> Self {
        Self {
            buf_name,
            inputs,
            write_pos: 0,
            buffers: vec![Vec::new(); 1],
        }
    }
}

impl Node for BufTapNode {
    fn process(&mut self, arena: &mut Arena, outputs: &mut [Frame]) {
        render_inputs(arena, &self.inputs, &mut self.buffers, outputs.len());

        let buffer = if let Some(buffer) = arena.get_buffer(&self.buf_name) {
            buffer
        } else {
            return;
        };

        let offset = self.buffers[0].as_slice();

        for (out, offset) in outputs.iter_mut().zip(offset.iter()) {
            let read_pos_f = self.write_pos as f32 - offset[0];
            let buffer_len = buffer.len() as f32;
            let read_pos = (read_pos_f % buffer_len + buffer_len) % buffer_len;
            *out = cubic_interpolate(&buffer, read_pos);
        }
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old) = (old as &dyn Any).downcast_ref::<BufTapNode>() {
            self.write_pos = old.write_pos;
        }
    }
}

#[derive(Clone)]
pub struct BufWriterNode {
    buf_name: String,
    inputs: Vec<usize>,
    write_pos: usize,
    buffers: Vec<Vec<Frame>>,
}

impl BufWriterNode {
    pub(crate) fn new(buf_name: String, inputs: Vec<usize>) -> Self {
        Self {
            buf_name,
            inputs,
            write_pos: 0,
            buffers: vec![Vec::new(); 1],
        }
    }
}

impl Node for BufWriterNode {
    fn process(&mut self, arena: &mut Arena, outputs: &mut [Frame]) {
        render_inputs(arena, &self.inputs, &mut self.buffers, outputs.len());

        let buffer = if let Some(buffer) = arena.get_buffer_mut(&self.buf_name) {
            buffer
        } else {
            return;
        };

        let input = self.buffers[0].as_mut_slice();

        for input in input.iter_mut() {
            buffer[self.write_pos] = *input;
            self.write_pos = (self.write_pos + 1) % buffer.len();
        }
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old) = (old as &dyn Any).downcast_ref::<BufWriterNode>() {
            self.write_pos = old.write_pos;
        }
    }
}

// delay line
pub(crate) struct DelayNode {
    inputs: Vec<usize>,
    buffer: [Frame; Self::BUFFER_SIZE],
    write_pos: usize,
    buffers: Vec<Vec<Frame>>,
}

impl DelayNode {
    pub const BUFFER_SIZE: usize = 48000;

    pub(crate) fn new(inputs: Vec<usize>) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            buffer: [[0.0; 2]; Self::BUFFER_SIZE],
            write_pos: 0,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for DelayNode {
    #[inline(always)]
    fn process(&mut self, arena: &mut Arena, outputs: &mut [Frame]) {
        render_inputs(arena, &self.inputs, &mut self.buffers, outputs.len());

        let signal = &self.buffers[0];
        let delay_input = &self.buffers[1];
        let buffer_size_f32 = Self::BUFFER_SIZE as f32;
        let max_delay = (Self::BUFFER_SIZE - 1) as f32;

        for (out, (sig, delay_frame)) in outputs
            .iter_mut()
            .zip(signal.iter().zip(delay_input.iter()))
        {
            self.buffer[self.write_pos] = *sig;

            let delay = delay_frame[0].clamp(0.0, max_delay);
            let read_pos = (self.write_pos as f32 - delay + buffer_size_f32) % buffer_size_f32;

            *out = cubic_interpolate(&self.buffer, read_pos);

            self.write_pos = (self.write_pos + 1) % Self::BUFFER_SIZE;
        }
    }

    fn get_inputs(&self) -> &[usize] {
        &self.inputs
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old) = (old as &dyn Any).downcast_ref::<DelayNode>() {
            self.buffer = old.buffer;
            self.write_pos = old.write_pos;
        }
    }
}
