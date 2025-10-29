use crate::utils::cubic_interpolate;
use fundsp::hacker32::AudioUnit;
use std::any::Any;
use std::f32::consts::PI;

pub type Frame = [f32; 2];

macro_rules! define_osc_node {
    ($name:ident, $phase_to_output:expr) => {
        pub struct $name {
            phase: f32,
            sample_rate: u32,
        }

        impl $name {
            pub(crate) fn new(sample_rate: u32) -> Self {
                Self {
                    phase: 0.0,
                    sample_rate,
                }
            }
        }

        impl Node for $name {
            #[inline(always)]
            fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
                const EMPTY: &[Frame] = &[];
                let freq_input = inputs.get(0).unwrap_or(&EMPTY);
                let sample_rate = self.sample_rate as f32;
                let two_pi = 2.0 * PI;
                let inc = two_pi / sample_rate;

                for (out, freq_frame) in outputs.iter_mut().zip(freq_input.iter()) {
                    let freq = freq_frame[0];
                    self.phase += freq * inc;
                    self.phase %= two_pi;

                    let y = $phase_to_output(self.phase, two_pi);

                    *out = [y; 2];
                }
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
        pub struct $name;

        impl $name {
            pub(crate) fn new() -> Self {
                Self
            }
        }

        impl Node for $name {
            #[inline(always)]
            fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
                const EMPTY: &[Frame] = &[];
                let lhs = inputs.get(0).unwrap_or(&EMPTY);
                let rhs = inputs.get(1).unwrap_or(&EMPTY);

                for (out, (l, r)) in outputs.iter_mut().zip(lhs.iter().zip(rhs.iter())) {
                    *out = $op(*l, *r);
                }
            }
        }
    };
}

pub(crate) trait Node: Send + Sync + Any {
    #[inline(always)]
    fn process(&mut self, _inputs: &[&[Frame]], _outputs: &mut [Frame]) {}

    fn get_id(&self) -> String {
        std::any::type_name::<Self>().to_string()
    }

    fn transfer_state(&mut self, _old: &dyn Node) {
        // stateless nodes don't transfer state, so we default to no-op
    }

    fn get_buf_name(&self) -> Option<&str> {
        None
    }
}

define_binary_op_node!(SumNode, |lhs: Frame, rhs: Frame| {
    [lhs[0] + rhs[0], lhs[1] + rhs[0]]
});

define_binary_op_node!(DiffNode, |lhs: Frame, rhs: Frame| {
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

pub struct ConstantNode {
    value: f32,
}

impl ConstantNode {
    pub(crate) fn new(value: f32) -> Self {
        Self { value }
    }
}

impl Node for ConstantNode {
    fn process(&mut self, _inputs: &[&[Frame]], outputs: &mut [Frame]) {
        let frame = [self.value; 2];
        outputs.fill(frame);
    }

    fn get_id(&self) -> String {
        format!("{}", self.value)
    }
}

pub struct SampleAndHoldNode {
    value: Frame,
    prev: f32,
}

impl SampleAndHoldNode {
    pub(crate) fn new(sample_rate: u32) -> Self {
        _ = sample_rate;
        Self {
            value: [0.0; 2],
            prev: 1.0,
        }
    }
}

impl Node for SampleAndHoldNode {
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        const EMPTY: &[Frame] = &[];
        let signal = inputs.get(0).unwrap_or(&EMPTY);
        let ramp = inputs.get(1).unwrap_or(&EMPTY);
        let chunk_size = outputs.len();

        for i in 0..chunk_size {
            let ramp_val = ramp.get(i).map(|f| f[0]).unwrap_or(0.0);
            if (ramp_val - self.prev).abs() > 0.5 {
                self.value = signal.get(i).copied().unwrap_or([0.0; 2]);
            }
            self.prev = ramp_val;
            outputs[i] = self.value;
        }
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old) = (old as &dyn Any).downcast_ref::<SampleAndHoldNode>() {
            self.value = old.value;
            self.prev = old.prev;
        }
    }
}

pub struct FunDSPNode {
    node: Box<dyn AudioUnit>,
    is_stereo: bool,
    input_buffer: Vec<f32>,
}

impl FunDSPNode {
    pub fn mono(node: Box<dyn AudioUnit>) -> Self {
        let num_audio_inputs = node.inputs();
        Self {
            node,
            is_stereo: false,
            input_buffer: vec![0.0; num_audio_inputs],
        }
    }

    pub fn stereo(node: Box<dyn AudioUnit>) -> Self {
        let num_audio_inputs = node.inputs();
        Self {
            node,
            is_stereo: true,
            input_buffer: vec![0.0; num_audio_inputs],
        }
    }

    fn get_name(&self) -> String {
        match self.node.get_id() {
            6 => "Reverb".to_string(),
            21 => "Sin".to_string(),
            _ => self.node.get_id().to_string(),
        }
    }
}

impl Node for FunDSPNode {
    #[inline(always)]
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        const EMPTY: &[Frame] = &[];
        let chunk_size = outputs.len();
        let num_inputs = self.node.inputs();

        if self.is_stereo {
            for i in 0..chunk_size {
                for input_idx in 0..num_inputs {
                    let input_frames = inputs.get(input_idx).unwrap_or(&EMPTY);
                    self.input_buffer[input_idx] = input_frames.get(i).map(|f| f[0]).unwrap_or(0.0);
                }

                let mut output_sample = [0.0; 2];
                self.node.tick(&self.input_buffer, &mut output_sample);
                outputs[i] = output_sample;
            }
        } else {
            for i in 0..chunk_size {
                for input_idx in 0..num_inputs {
                    let input_frames = inputs.get(input_idx).unwrap_or(&EMPTY);
                    self.input_buffer[input_idx] = input_frames.get(i).map(|f| f[0]).unwrap_or(0.0);
                }

                let mut output_sample = [0.0; 1];
                self.node.tick(&self.input_buffer, &mut output_sample);
                outputs[i] = [output_sample[0]; 2];
            }
        }
    }

    fn get_id(&self) -> String {
        format!("FunDSP: {}", self.get_name())
    }
}

pub struct MixNode;

impl MixNode {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Node for MixNode {
    #[inline(always)]
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        let num_inputs = inputs.len();

        // return silence if there are no inputs
        if num_inputs == 0 {
            outputs.fill([0.0; 2]);
            return;
        }

        let inv_n = 1.0 / num_inputs as f32;

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
}

pub struct SeqNode;

impl SeqNode {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Node for SeqNode {
    #[inline(always)]
    fn process(&mut self, inputs: &[&[Frame]], outputs: &mut [Frame]) {
        if inputs.is_empty() {
            outputs.fill([0.0; 2]);
            return;
        }

        let ramp_input = inputs[0];
        let num_steps = inputs.len() - 1;

        for (i, (out, ramp_frame)) in outputs.iter_mut().zip(ramp_input.iter()).enumerate() {
            let ramp = ramp_frame[0];
            let step = (ramp * num_steps as f32).floor() as usize;
            let step_input = inputs[step + 1];
            *out = step_input.get(i).copied().unwrap_or([0.0; 2]);
        }
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
    fn process(&mut self, _inputs: &[&[Frame]], outputs: &mut [Frame]) {
        // TODO: implement buffer support
        outputs.fill([0.0; 2]);
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
    fn process(&mut self, _inputs: &[&[Frame]], outputs: &mut [Frame]) {
        // TODO: implement buffer support
        outputs.fill([0.0; 2]);
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
    fn process(&mut self, _inputs: &[&[Frame]], _outputs: &mut [Frame]) {
        // TODO: implement buffer support
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old) = (old as &dyn Any).downcast_ref::<BufWriterNode>() {
            self.write_pos = old.write_pos;
        }
    }
}

// delay line
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
        const EMPTY: &[Frame] = &[];
        let signal = inputs.get(0).unwrap_or(&EMPTY);
        let delay_input = inputs.get(1).unwrap_or(&EMPTY);
        let buffer_size_f32 = Self::BUFFER_SIZE as f32;
        let max_delay = (Self::BUFFER_SIZE - 1) as f32;

        for (i, out) in outputs.iter_mut().enumerate() {
            let sig = signal.get(i).copied().unwrap_or([0.0; 2]);
            self.buffer[self.write_pos] = sig;

            let delay = delay_input
                .get(i)
                .map(|f| f[0])
                .unwrap_or(0.0)
                .clamp(0.0, max_delay);
            let read_pos = (self.write_pos as f32 - delay + buffer_size_f32) % buffer_size_f32;

            *out = cubic_interpolate(&self.buffer, read_pos);

            self.write_pos = (self.write_pos + 1) % Self::BUFFER_SIZE;
        }
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old) = (old as &dyn Any).downcast_ref::<DelayNode>() {
            self.buffer = old.buffer;
            self.write_pos = old.write_pos;
        }
    }
}
