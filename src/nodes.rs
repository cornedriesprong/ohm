use crate::utils::cubic_interpolate;
use fundsp::hacker32::AudioUnit;
use std::f32::consts::PI;

pub type Frame = [f32; 2];

#[inline(always)]
fn render_inputs(inputs: &mut [Box<dyn Node>], buffers: &mut Vec<Vec<Frame>>, len: usize) {
    if buffers.len() != inputs.len() {
        buffers.resize_with(inputs.len(), Vec::new);
    }

    for (i, input_node) in inputs.iter_mut().enumerate() {
        if buffers[i].len() < len {
            buffers[i].resize(len, [0.0; 2]);
        }
        input_node.process(&mut buffers[i][..len]);
    }
}

pub(crate) trait Node: Send + Sync + std::any::Any {
    #[inline(always)]
    fn process(&mut self, _outputs: &mut [Frame]) {
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

    fn get_id(&self) -> String;

    fn clone_box(&self) -> Box<dyn Node>;

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &[]
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut []
    }

    fn transfer_state(&mut self, _old: &dyn Node) {}
}

#[derive(Clone)]
pub struct MixNode {
    inputs: Vec<Box<dyn Node>>,
    buffers: Vec<Vec<Frame>>,
}

impl MixNode {
    pub(crate) fn new(inputs: Vec<Box<dyn Node>>) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for MixNode {
    #[inline(always)]
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

        for i in 0..outputs.len() {
            let mut mix = [0.0; 2];
            for input_buffer in &self.buffers {
                mix[0] += input_buffer[i][0];
                mix[1] += input_buffer[i][1];
            }
            outputs[i] = mix;
        }
    }

    fn get_id(&self) -> String {
        "MixNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }
}

#[derive(Clone)]
pub struct GainNode {
    inputs: Vec<Box<dyn Node>>,
    buffers: Vec<Vec<Frame>>,
}

impl GainNode {
    pub(crate) fn new(inputs: Vec<Box<dyn Node>>) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for GainNode {
    #[inline(always)]
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

        let signal = &self.buffers[0];
        let gain_input = &self.buffers[1];

        for i in 0..outputs.len() {
            let gain = gain_input[i][0];
            outputs[i] = [signal[i][0] * gain, signal[i][1] * gain];
        }
    }

    fn get_id(&self) -> String {
        "GainNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }
}

#[derive(Clone)]
pub struct WrapNode {
    inputs: Vec<Box<dyn Node>>,
    buffers: Vec<Vec<Frame>>,
}

impl WrapNode {
    pub(crate) fn new(inputs: Vec<Box<dyn Node>>) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for WrapNode {
    #[inline(always)]
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

        let signal = &self.buffers[0];
        let modulo = &self.buffers[1];

        for i in 0..outputs.len() {
            outputs[i] = [signal[i][0] % modulo[i][0], signal[i][1] % modulo[i][0]];
        }
    }

    fn get_id(&self) -> String {
        "WrapNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }
}

#[derive(Clone)]
pub struct PowerNode {
    inputs: Vec<Box<dyn Node>>,
    buffers: Vec<Vec<Frame>>,
}

impl PowerNode {
    pub(crate) fn new(inputs: Vec<Box<dyn Node>>) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for PowerNode {
    #[inline(always)]
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

        let signal = &self.buffers[0];
        let power = &self.buffers[1];

        for i in 0..outputs.len() {
            outputs[i] = [
                signal[i][0].powf(power[i][0]),
                signal[i][1].powf(power[i][0]),
            ];
        }
    }

    fn get_id(&self) -> String {
        "PowerNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }
}

#[derive(Clone)]
pub struct GreaterNode {
    inputs: Vec<Box<dyn Node>>,
    buffers: Vec<Vec<Frame>>,
}

impl GreaterNode {
    pub(crate) fn new(inputs: Vec<Box<dyn Node>>) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for GreaterNode {
    #[inline(always)]
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

        let left = &self.buffers[0];
        let right = &self.buffers[1];

        for i in 0..outputs.len() {
            outputs[i] = [
                if left[i][0] > right[i][0] { 1.0 } else { 0.0 },
                if left[i][1] > right[i][0] { 1.0 } else { 0.0 },
            ];
        }
    }

    fn get_id(&self) -> String {
        "GreaterNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }
}

#[derive(Clone)]
pub struct LessNode {
    inputs: Vec<Box<dyn Node>>,
    buffers: Vec<Vec<Frame>>,
}

impl LessNode {
    pub(crate) fn new(inputs: Vec<Box<dyn Node>>) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for LessNode {
    #[inline(always)]
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

        let left = &self.buffers[0];
        let right = &self.buffers[1];

        for i in 0..outputs.len() {
            outputs[i] = [
                if left[i][0] < right[i][0] { 1.0 } else { 0.0 },
                if left[i][1] < right[i][0] { 1.0 } else { 0.0 },
            ];
        }
    }

    fn get_id(&self) -> String {
        "LessNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }
}

#[derive(Clone)]
pub struct EqualNode {
    inputs: Vec<Box<dyn Node>>,
    buffers: Vec<Vec<Frame>>,
}

impl EqualNode {
    pub(crate) fn new(inputs: Vec<Box<dyn Node>>) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for EqualNode {
    #[inline(always)]
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

        let left = &self.buffers[0];
        let right = &self.buffers[1];

        for i in 0..outputs.len() {
            outputs[i] = [
                if left[i][0] == right[i][0] { 1.0 } else { 0.0 },
                if left[i][1] == right[i][0] { 1.0 } else { 0.0 },
            ];
        }
    }

    fn get_id(&self) -> String {
        "EqualNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }
}

#[derive(Clone)]
pub struct LFONode {
    inputs: Vec<Box<dyn Node>>,
    phase: f32,
    sample_rate: u32,
    buffers: Vec<Vec<Frame>>,
}

impl LFONode {
    pub(crate) fn new(inputs: Vec<Box<dyn Node>>, sample_rate: u32) -> Self
    where
        Self: Sized,
    {
        let num_inputs = inputs.len();
        Self {
            inputs,
            phase: 0.0,
            sample_rate,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for LFONode {
    #[inline(always)]
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

        let freq_input = &self.buffers[0];

        for i in 0..outputs.len() {
            let freq = freq_input[i][0].clamp(0.0, self.sample_rate as f32 / 2.0);

            self.phase += 2.0 * PI * freq / self.sample_rate as f32;

            if self.phase >= 2.0 * PI {
                self.phase = self.phase % (2.0 * PI);
            }

            let y = self.phase.cos();
            let y = (y + 1.0) * 0.5; // map to unipolar

            outputs[i] = [y; 2];
        }
    }

    fn get_id(&self) -> String {
        "LFONode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old_lfo) = (old as &dyn std::any::Any).downcast_ref::<LFONode>() {
            self.phase = old_lfo.phase;
        }
    }
}

#[derive(Clone)]
pub struct SampleAndHoldNode {
    inputs: Vec<Box<dyn Node>>,
    value: Frame,
    prev: f32,
    buffers: Vec<Vec<Frame>>,
}

impl SampleAndHoldNode {
    pub(crate) fn new(inputs: Vec<Box<dyn Node>>, sample_rate: u32) -> Self {
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
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

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

    fn get_id(&self) -> String {
        "SampleAndHoldNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old_sh) = (old as &dyn std::any::Any).downcast_ref::<SampleAndHoldNode>() {
            self.value = old_sh.value;
            self.prev = old_sh.prev;
        }
    }
}

#[derive(Clone)]
pub struct FunDSPNode {
    inputs: Vec<Box<dyn Node>>,
    node: Box<dyn AudioUnit>,
    is_stereo: bool,
    input_buffer: Vec<f32>,
    buffers: Vec<Vec<Frame>>,
}

impl FunDSPNode {
    pub fn mono(inputs: Vec<Box<dyn Node>>, node: Box<dyn AudioUnit>) -> Self {
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

    pub fn stereo(inputs: Vec<Box<dyn Node>>, node: Box<dyn AudioUnit>) -> Self {
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
    fn process(&mut self, outputs: &mut [Frame]) {
        let chunk_size = outputs.len();
        render_inputs(&mut self.inputs, &mut self.buffers, chunk_size);

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

    fn get_id(&self) -> String {
        "FunDSPNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }
}

impl Clone for Box<dyn Node> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

#[derive(Clone)]
pub struct SeqNode {
    inputs: Vec<Box<dyn Node>>,
    buffers: Vec<Vec<Frame>>,
}

impl SeqNode {
    pub(crate) fn new(inputs: Vec<Box<dyn Node>>) -> Self {
        let num_inputs = inputs.len();
        Self {
            inputs,
            buffers: vec![Vec::new(); num_inputs],
        }
    }
}

impl Node for SeqNode {
    #[inline(always)]
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

        let num_values = self.buffers.len() - 1;
        let ramp_input = &self.buffers[num_values];

        for i in 0..outputs.len() {
            let ramp = ramp_input[i][0].clamp(0.0, 1.0);

            let segment = 1.0 / num_values as f32;
            let step = (ramp / segment).floor() as usize;

            // Safety check since we once got a panic here
            let step = step.min(num_values - 1);
            outputs[i] = self.buffers[step][i];
        }
    }

    fn get_id(&self) -> String {
        "SeqNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
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
                .map(|[l, _]| {
                    let val = *l;
                    ((val % 1.0) + 1.0) % 1.0 // wrap to 0.0, 1.0)
                })
                .unwrap_or(0.0);
            let read_pos = phase * (buffer.len() as f32 - f32::EPSILON);

            outputs[i] = cubic_interpolate(buffer, read_pos);
        }
    }

    fn get_id(&self) -> String {
        "BufReaderNode".to_string()
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
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let read_pos_f = self.write_pos as f32 - offset;
            let buffer_len = buffer.len() as f32;
            let read_pos = (read_pos_f % buffer_len + buffer_len) % buffer_len;
            outputs[i] = cubic_interpolate(&buffer, read_pos);

            self.write_pos = (self.write_pos + 1) % buffer.len();
        }
    }

    fn get_id(&self) -> String {
        "BufTapNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old_tap) = (old as &dyn std::any::Any).downcast_ref::<BufTapNode>() {
            self.write_pos = old_tap.write_pos;
        }
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

    fn get_id(&self) -> String {
        "BufWriterNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old_writer) = (old as &dyn std::any::Any).downcast_ref::<BufWriterNode>() {
            self.write_pos = old_writer.write_pos;
        }
    }
}

// delay line
#[derive(Clone)]
pub(crate) struct DelayNode {
    inputs: Vec<Box<dyn Node>>,
    buffer: [Frame; Self::BUFFER_SIZE],
    write_pos: usize,
    buffers: Vec<Vec<Frame>>,
}

impl DelayNode {
    pub const BUFFER_SIZE: usize = 48000;

    pub(crate) fn new(inputs: Vec<Box<dyn Node>>) -> Self {
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
    fn process(&mut self, outputs: &mut [Frame]) {
        render_inputs(&mut self.inputs, &mut self.buffers, outputs.len());

        let signal = &self.buffers[0];
        let delay_input = &self.buffers[1];

        for i in 0..outputs.len() {
            let input = signal[i];
            self.buffer[self.write_pos] = input;

            let delay = delay_input[i][0];
            // clamp delay to buffer size
            let delay = delay.max(0.0).min((Self::BUFFER_SIZE - 1) as f32);

            let read_pos = (self.write_pos as f32 - delay + Self::BUFFER_SIZE as f32)
                % Self::BUFFER_SIZE as f32;

            outputs[i] = cubic_interpolate(&self.buffer, read_pos);

            self.write_pos = (self.write_pos + 1) % Self::BUFFER_SIZE;
        }
    }

    fn get_id(&self) -> String {
        "DelayNode".to_string()
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn get_inputs(&self) -> &[Box<dyn Node>] {
        &self.inputs
    }

    fn get_inputs_mut(&mut self) -> &mut [Box<dyn Node>] {
        &mut self.inputs
    }

    fn transfer_state(&mut self, old: &dyn Node) {
        if let Some(old_delay) = (old as &dyn std::any::Any).downcast_ref::<DelayNode>() {
            self.buffer = old_delay.buffer;
            self.write_pos = old_delay.write_pos;
        }
    }
}
