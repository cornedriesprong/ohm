use crate::{
    ast::{BinaryOp, UnaryOp},
    stack::Stack,
};
use std::{collections::HashMap, f32::consts::PI};

pub(crate) const DEFAULT_BPM: f32 = 120.0;
pub(crate) const BEATS_PER_CYCLE: f32 = 4.0;
pub(crate) const RING_LEN: usize = 1 << 18;

pub struct Program {
    pub(crate) ops: Vec<OpCode>,
    pub slots: HashMap<String, usize>,
    pub channels: usize,
    pub bpm_slot: Option<usize>,
    state: Vec<f32>,
    buffer: Vec<f32>,
    regions: HashMap<String, (usize, usize)>,
    frame: usize,
    position: f64,
    stack: Stack,
    phases: Stack<f64>,
    seed: u32,
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            state: Vec::new(),
            channels: 0,
            slots: HashMap::new(),
            bpm_slot: None,
            buffer: Vec::new(),
            regions: HashMap::new(),
            frame: 0,
            position: 0.0,
            stack: Stack::new(),
            phases: Stack::new(),
            seed: 0x9e3779b9,
        }
    }

    pub fn copy_state_from(&mut self, old: &Self) {
        self.position = old.position;
        self.frame = old.frame;
        for (name, slot) in &self.slots {
            if let Some(old_slot) = old.slots.get(name) {
                self.state[*slot] = old.state[*old_slot];
            }
        }
        for (name, (start, len)) in &self.regions {
            if let Some((old_start, old_len)) = old.regions.get(name)
                && old_len == len
            {
                self.buffer[*start..*start + len]
                    .copy_from_slice(&old.buffer[*old_start..*old_start + len]);
            }
        }
    }

    pub(crate) fn insert_buffer(&mut self, name: String, len: usize) -> usize {
        let start = self.buffer.len();
        self.buffer.resize(start + len, 0.0);
        self.regions.insert(name, (start, len));
        start
    }

    // start and length
    pub(crate) fn region(&self, name: &str) -> Option<(usize, usize)> {
        self.regions.get(name).copied()
    }

    pub(crate) fn insert_state(&mut self, name: String) -> usize {
        let slot = self.state.len();
        self.state.push(0.0);
        self.slots.insert(name, slot);
        slot
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Waveform {
    Ramp,
    Sin,
    Saw,
    Square,
    Triangle,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FilterMode {
    Lp,
    Bp,
    Hp,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum OpCode {
    Num(f32),
    Unary(UnaryOp),
    Binary(BinaryOp),
    Osc {
        waveform: Waveform,
        phase_slot: usize,
    },
    Noise,
    Clock {
        rate: f32,
    },
    Mtof,
    PushPhase {
        length: usize,
    },
    Seq {
        length: usize,
    },
    Mix {
        length: usize,
    },
    Svf {
        mode: FilterMode,
        state_slot: usize,
    },
    Print,
    Store(usize),
    Load(usize),
    Write {
        start: usize,
        mask: usize,
    },
    Tap {
        start: usize,
        mask: usize,
    },
    SampleRate,
    Bar,
}

fn poly_blep(phase: f32, dt: f32) -> f32 {
    if phase < dt {
        let t = phase / dt;

        t + t - t * t - 1.0
    } else if phase > 1.0 - dt {
        let t = (phase - 1.0) / dt;
        t * t + t + t + 1.0
    } else {
        0.0
    }
}

fn poly_blamp(phase: f32, dt: f32) -> f32 {
    let t = (if phase < 0.5 { phase } else { phase - 1.0 }) / dt;
    if t.abs() < 1.0 {
        (1.0 - t.abs()).powi(3) / 3.0
    } else {
        0.0
    }
}

fn tempo(bpm_slot: Option<usize>, state: &[f32]) -> f32 {
    let bpm = bpm_slot.map_or(DEFAULT_BPM, |slot| state[slot]);
    finite_or(bpm, DEFAULT_BPM).clamp(1.0, 999.0)
}

fn finite_or(value: f32, default: f32) -> f32 {
    if value.is_finite() { value } else { default }
}

#[inline]
pub fn render(prog: &mut Program, output: &mut [f32], sample_rate: u32) {
    let stack = &mut prog.stack;
    stack.reset();
    prog.phases.reset();
    prog.phases.push(prog.position);
    let nyquist = sample_rate as f32 / 2.0;
    for op in &prog.ops {
        match op {
            OpCode::Num(num) => stack.push(*num),
            OpCode::Unary(op) => {
                let value = stack.top();
                *stack.top_mut() = match op {
                    UnaryOp::Negate => -value,
                    UnaryOp::Floor => value.floor(),
                    UnaryOp::Tanh => value.tanh(),
                    UnaryOp::Abs => value.abs(),
                };
            }
            OpCode::Binary(op) => {
                let rhs = stack.pop();
                let lhs = stack.top();
                *stack.top_mut() = match op {
                    BinaryOp::Add => lhs + rhs,
                    BinaryOp::Sub => lhs - rhs,
                    BinaryOp::Mul => lhs * rhs,
                    BinaryOp::Div => lhs / rhs,
                    BinaryOp::Equal => f32::from(lhs == rhs),
                    BinaryOp::NotEqual => f32::from(lhs != rhs),
                    BinaryOp::GreaterThan => f32::from(lhs > rhs),
                    BinaryOp::LessThan => f32::from(lhs < rhs),
                    BinaryOp::GreaterThanOrEqual => f32::from(lhs >= rhs),
                    BinaryOp::LessThanOrEqual => f32::from(lhs <= rhs),
                    BinaryOp::Pow => lhs.powf(rhs),
                    BinaryOp::Mod => lhs % rhs,
                    BinaryOp::Min => lhs.min(rhs),
                    BinaryOp::Max => lhs.max(rhs),
                };
            }
            OpCode::Osc {
                waveform: kind,
                phase_slot,
            } => {
                let offset = stack.pop();
                let freq = stack.top();
                // rests are silent
                if freq == 0.0 || freq.is_nan() {
                    *stack.top_mut() = 0.0;
                    prog.state[*phase_slot] = 0.0; // reset phase
                    continue;
                }
                let freq = freq.clamp(-nyquist, nyquist);
                let mut phase = prog.state[*phase_slot];
                let out_phase = (phase + offset).rem_euclid(1.0);
                let dt = freq.abs() / sample_rate as f32;
                *stack.top_mut() = match kind {
                    Waveform::Ramp => out_phase,
                    Waveform::Sin => (out_phase * PI * 2.0).sin(),
                    Waveform::Saw => 1.0 - out_phase * 2.0 + poly_blep(out_phase, dt),
                    Waveform::Square => {
                        let naive = if out_phase < 0.5 { 1.0 } else { -1.0 };
                        naive + poly_blep(out_phase, dt)
                            - poly_blep((out_phase + 0.5).rem_euclid(1.0), dt)
                    }
                    Waveform::Triangle => {
                        let naive = 1.0 - 4.0 * ((out_phase + 0.25).rem_euclid(1.0) - 0.5).abs();
                        naive
                            + 4.0
                                * dt
                                * (poly_blamp((out_phase - 0.75).rem_euclid(1.0), dt)
                                    - poly_blamp((out_phase - 0.25).rem_euclid(1.0), dt))
                    }
                };
                phase = (phase + freq / sample_rate as f32).rem_euclid(1.0);
                if phase >= 1.0 {
                    phase -= 1.0;
                }
                prog.state[*phase_slot] = phase;
            }
            OpCode::Print => {
                // TODO: send off audio thread
                println!("{}", stack.top())
            }
            OpCode::Noise => {
                let s = &mut prog.seed;
                *s ^= *s << 13;
                *s ^= *s >> 17;
                *s ^= *s << 5;
                stack.push(*s as f32 / u32::MAX as f32 * 2.0 - 1.0);
            }
            OpCode::Mtof => *stack.top_mut() = 440.0 * ((stack.top() - 69.0) / 12.0).exp2(),
            OpCode::Clock { rate } => {
                stack.push((prog.phases.top() * *rate as f64).rem_euclid(1.0) as f32)
            }
            OpCode::PushPhase { length } => prog.phases.push(stack.top() as f64 * *length as f64),
            OpCode::Svf { mode, state_slot } => {
                let q = stack.pop().clamp(0.5, 25.0);
                let cutoff = stack.pop().clamp(10.0, sample_rate as f32 * 0.45);
                let input = stack.top();

                // https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf
                let g = (PI * cutoff / sample_rate as f32).tan();
                let k = 1.0 / q;
                let a1 = 1.0 / (1.0 + g * (g + k));
                let a2 = g * a1;
                let a3 = g * a2;
                let ic1eq = prog.state[*state_slot];
                let ic2eq = prog.state[*state_slot + 1];
                let v3 = input - ic2eq;
                let v1 = a1 * ic1eq + a2 * v3;
                let v2 = ic2eq + a2 * ic1eq + a3 * v3;
                prog.state[*state_slot] = finite_or(2.0 * v1 - ic1eq, 0.0);
                prog.state[*state_slot + 1] = finite_or(2.0 * v2 - ic2eq, 0.0);

                *stack.top_mut() = match mode {
                    FilterMode::Lp => v2,
                    FilterMode::Bp => v1,
                    FilterMode::Hp => input - k * v1 - v2,
                };
            }
            OpCode::Mix { length } => {
                let mut sum = 0.0;
                for _ in 0..*length {
                    sum += stack.pop();
                }
                stack.push(sum);
            }
            OpCode::Seq { length } => {
                prog.phases.pop();
                if *length == 0 {
                    *stack.top_mut() = 0.0;
                } else {
                    let index = (stack.peek(*length) * *length as f32) as usize % *length;
                    let val = *stack.peek(index);
                    for _ in 0..*length {
                        stack.pop();
                    }
                    *stack.top_mut() = val;
                }
            }
            OpCode::Store(slot) => prog.state[*slot] = finite_or(stack.pop(), 0.0),
            OpCode::Load(slot) => stack.push(prog.state[*slot]),
            OpCode::Write { start, mask } => {
                prog.buffer[start + (prog.frame & mask)] = finite_or(stack.top(), 0.0)
            }
            OpCode::Tap { start, mask } => {
                let delay = finite_or(stack.top(), 1.0).clamp(1.0, (mask - 1) as f32);
                let whole = delay as usize;
                let read =
                    |delay: usize| prog.buffer[start + (prog.frame.wrapping_sub(delay) & mask)];
                let a = read(whole);
                *stack.top_mut() = a + (read(whole + 1) - a) * (delay - whole as f32);
            }
            OpCode::SampleRate => stack.push(sample_rate as f32),
            OpCode::Bar => stack.push(60.0 / tempo(prog.bpm_slot, &prog.state) * BEATS_PER_CYCLE),
        }
    }

    let bpm = tempo(prog.bpm_slot, &prog.state);
    prog.position += bpm as f64 / 60.0 / BEATS_PER_CYCLE as f64 / sample_rate as f64;
    prog.frame = prog.frame.wrapping_add(1);

    let n = prog.channels;
    for i in (0..n).rev() {
        let v = stack.pop();
        if i < output.len() {
            output[i] = finite_or(v, 0.0);
        }
    }
    for i in n..output.len() {
        output[i] = if n == 0 { 0.0 } else { output[i % n] };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{CompileError, tests::compile_src as run_src};

    static SAMPLE_RATE: u32 = 48_000;
    // sample rate at which the ambient phase takes 4 frames per cycle at the default bpm
    const FRAMES_PER_CYCLE: u32 = (DEFAULT_BPM / 60.0 / BEATS_PER_CYCLE * 4.0) as u32;

    fn render(prog: &mut Program, sample_rate: u32) -> f32 {
        let mut output = [0.0];
        super::render(prog, &mut output, sample_rate);
        output[0]
    }

    fn assert_approx_eq(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= f32::EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn run_prelude_src(src: &str) -> Result<Program, CompileError> {
        run_src(&format!("{}\n{src}", include_str!("../prelude.ohm")))
    }

    #[test]
    fn run_binop() -> Result<(), CompileError> {
        let mut prog = run_src("1 + 2")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 3.0);

        let mut prog = run_src("2 - 1")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 1.0);

        let mut prog = run_src("2 - 3")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), -1.0);

        let mut prog = run_src("1 * 2")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 2.0);

        let mut prog = run_src("4 / 2")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 2.0);

        Ok(())
    }

    #[test]
    fn run_negation() -> Result<(), CompileError> {
        let mut prog = run_src("-1")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), -1.0);

        let mut prog = run_src("x = 5\n-x")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), -5.0);

        Ok(())
    }

    #[test]
    fn run_equals() -> Result<(), CompileError> {
        for (src, expected) in [("b = 1\nb == 1", 1.0), ("b = 1\nb == 2", 0.0)] {
            let mut prog = run_src(src)?;
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        Ok(())
    }

    #[test]
    fn run_not_equals() -> Result<(), CompileError> {
        for (src, expected) in [("b = 1\nb != 1", 0.0), ("b = 1\nb != 2", 1.0)] {
            let mut prog = run_src(src)?;
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        Ok(())
    }

    #[test]
    fn run_greater_than() -> Result<(), CompileError> {
        for (src, expected) in [("1 > 2", 0.0), ("3 > 2", 1.0)] {
            let mut prog = run_src(src)?;
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        Ok(())
    }

    #[test]
    fn run_less_than() -> Result<(), CompileError> {
        for (src, expected) in [("1 < 2", 1.0), ("3 < 2", 0.0)] {
            let mut prog = run_src(src)?;
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        Ok(())
    }

    #[test]
    fn run_greater_than_or_equal() -> Result<(), CompileError> {
        for (src, expected) in [("1 >= 2", 0.0), ("2 >= 2", 1.0), ("3 >= 2", 1.0)] {
            let mut prog = run_src(src)?;
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        Ok(())
    }

    #[test]
    fn run_less_than_or_equal() -> Result<(), CompileError> {
        for (src, expected) in [("1 <= 2", 1.0), ("2 <= 2", 1.0), ("3 <= 2", 0.0)] {
            let mut prog = run_src(src)?;
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        Ok(())
    }

    #[test]
    fn run_pow() -> Result<(), CompileError> {
        assert_approx_eq(render(&mut run_src("2 ^ 2")?, SAMPLE_RATE), 4.0);
        assert_approx_eq(render(&mut run_src("2 ^ 3")?, SAMPLE_RATE), 8.0);
        assert_approx_eq(render(&mut run_src("3 ^ 2")?, SAMPLE_RATE), 9.0);
        assert_approx_eq(render(&mut run_src("3 ^ 3")?, SAMPLE_RATE), 27.0);

        Ok(())
    }

    #[test]
    fn run_mod() -> Result<(), CompileError> {
        assert_approx_eq(render(&mut run_src("2 % 2")?, SAMPLE_RATE), 0.0);
        assert_approx_eq(render(&mut run_src("2 % 3")?, SAMPLE_RATE), 2.0);
        assert_approx_eq(render(&mut run_src("3 % 2")?, SAMPLE_RATE), 1.0);

        Ok(())
    }

    #[test]
    fn run_multiple_statements() -> Result<(), CompileError> {
        let mut prog = run_src("x = 5\nadd = |y| x + y\nadd 3")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 8.0);

        Ok(())
    }

    #[test]
    fn run_cycle() -> Result<(), CompileError> {
        let mut prog = run_src("x = ~x + 1\nx")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 1.0);
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 2.0);
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 3.0);

        Ok(())
    }

    #[test]
    fn run_cycle_order() -> Result<(), CompileError> {
        // `~x` is the previous frame, whether `x` is stored before or after
        for src in ["x = ~x + 1\ny = ~x * 1\ny", "y = ~x * 1\nx = ~x + 1\ny"] {
            let mut prog = run_src(src)?;
            for expected in [0.0, 1.0, 2.0] {
                assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
            }
        }

        Ok(())
    }

    #[test]
    fn run_ramp() -> Result<(), CompileError> {
        let mut prog = run_src("ramp 1")?;
        for expected in [0.0, 0.25, 0.5, 0.75, 0.0] {
            assert_approx_eq(render(&mut prog, 4), expected);
        }

        Ok(())
    }

    #[test]
    fn run_sin() -> Result<(), CompileError> {
        let mut prog = run_src("sin 440 0.5")?;
        let freq = 440.0_f32;
        let sr = 48_000.0_f32;
        for i in 0_i16..4 {
            let phase = (f32::from(i) * freq / sr) + 0.5;
            let expected = (phase * std::f32::consts::PI * 2.0).sin();
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        // should wrap phase
        let mut prog = run_src(&format!("sin {}", SAMPLE_RATE / 2))?;
        render(&mut prog, SAMPLE_RATE);
        render(&mut prog, SAMPLE_RATE);
        assert_approx_eq(prog.state[0], 0.0);

        // frequency is clamped to nyquist
        let mut prog = run_src(&format!("sin {SAMPLE_RATE}"))?;
        render(&mut prog, SAMPLE_RATE);
        assert_approx_eq(prog.state[0], 0.5);

        Ok(())
    }

    #[test]
    fn run_saw() -> Result<(), CompileError> {
        // bandlimited: the step at the start of each cycle is smoothed to its midpoint
        let mut prog = run_src("saw 1")?;
        for expected in [0.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, 0.0] {
            assert_approx_eq(render(&mut prog, 8), expected);
        }

        Ok(())
    }

    #[test]
    fn run_square() -> Result<(), CompileError> {
        // bandlimited: both steps are smoothed to their midpoints
        let mut prog = run_src("square 1")?;
        for expected in [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0] {
            assert_approx_eq(render(&mut prog, 8), expected);
        }

        Ok(())
    }

    #[test]
    fn run_tri() -> Result<(), CompileError> {
        // bandlimited: the corners are rounded off
        let corner = 1.0 - 0.5 / 3.0;
        let mut prog = run_src("tri 1")?;
        for expected in [0.0, 0.5, corner, 0.5, 0.0, -0.5, -corner, -0.5, 0.0] {
            assert_approx_eq(render(&mut prog, 8), expected);
        }

        // naive at low frequencies
        let mut prog = run_src("tri 1 0.25")?;
        assert!((render(&mut prog, SAMPLE_RATE) - 1.0).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn run_seq_call() -> Result<(), CompileError> {
        let mut prog = run_src("ramp 1 -> <100 200>")?;
        for expected in [100.0, 100.0, 200.0, 200.0, 100.0] {
            assert_approx_eq(render(&mut prog, 4), expected);
        }

        Ok(())
    }

    #[test]
    fn run_seq_call_wrap() -> Result<(), CompileError> {
        let mut prog = run_src("1.75 -> <100 200>")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 200.0);
        Ok(())
    }

    #[test]
    fn run_empty_seq_call() -> Result<(), CompileError> {
        let mut prog = run_src("ramp 1 -> <>")?;
        assert_approx_eq(render(&mut prog, 4), 0.0);

        Ok(())
    }

    #[test]
    fn run_seq_call_stack() -> Result<(), CompileError> {
        let mut prog = run_src("(0 -> <100 300>) * 2")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 200.0);

        let mut prog = run_src("2 * (0 -> <100 300>)")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 200.0);

        Ok(())
    }

    #[test]
    fn run_seq_call_ref() -> Result<(), CompileError> {
        let mut prog = run_src("x = 100\n0 -> <~x 200>")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 0.0);
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 100.0);
        Ok(())
    }

    #[test]
    fn run_seq_call_rest() -> Result<(), CompileError> {
        // silent during the rest, restarts afterwards
        let mut prog = run_src("ramp 1 -> <1 _> -> square")?;
        for expected in [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] {
            assert_approx_eq(render(&mut prog, 8), expected);
        }

        Ok(())
    }

    #[test]
    fn run_stack() -> Result<(), CompileError> {
        let mut prog = run_src("{100 200}")?;
        let mut output = [0.0; 2];
        super::render(&mut prog, &mut output, 4);
        assert_eq!(output, [100.0, 200.0]);

        Ok(())
    }

    #[test]
    fn run_seq() -> Result<(), CompileError> {
        // uncalled seq is driven by the ambient phase
        let mut prog = run_src("<100 200>")?;
        let sample_rate = FRAMES_PER_CYCLE;
        for expected in [100.0, 100.0, 200.0, 200.0, 100.0] {
            assert_approx_eq(render(&mut prog, sample_rate), expected);
        }

        Ok(())
    }

    #[test]
    fn run_nested_seq() -> Result<(), CompileError> {
        // inner seq takes one step of the outer
        let mut prog = run_src("ramp 1 -> <100 <200 300>>")?;
        for expected in [100.0, 100.0, 200.0, 300.0, 100.0] {
            assert_approx_eq(render(&mut prog, 4), expected);
        }

        // nested uncalled seq
        let mut prog = run_src("<<100 200> 300>")?;
        for expected in [100.0, 200.0, 300.0, 300.0, 100.0] {
            assert_approx_eq(render(&mut prog, FRAMES_PER_CYCLE), expected);
        }

        Ok(())
    }

    #[test]
    fn run_clock() -> Result<(), CompileError> {
        // ambient phase at the top level
        let mut prog = run_src("clock")?;
        for expected in [0.0, 0.25, 0.5, 0.75, 0.0] {
            assert_approx_eq(render(&mut prog, FRAMES_PER_CYCLE), expected);
        }

        // phase of the step inside a seq
        let mut prog = run_src("ramp 1 -> <clock (1 - clock)>")?;
        for expected in [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25] {
            assert_approx_eq(render(&mut prog, 8), expected);
        }

        Ok(())
    }

    #[test]
    fn run_clock_rate() -> Result<(), CompileError> {
        // a quarter of the speed spans 4 bars
        let mut prog = run_src("clock / 4")?;
        for i in 0..17 {
            let expected = (i % 16) as f32 / 16.0;
            assert_approx_eq(render(&mut prog, FRAMES_PER_CYCLE), expected);
        }

        let mut prog = run_src("clock * 2 % 1")?;
        for expected in [0.0, 0.5, 0.0, 0.5] {
            assert_approx_eq(render(&mut prog, FRAMES_PER_CYCLE), expected);
        }

        // inside a seq, a slow clock spans multiple steps
        let mut prog = run_src("ramp 1 -> <(clock / 2) (clock / 2)>")?;
        for expected in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.0] {
            assert_approx_eq(render(&mut prog, 8), expected);
        }

        Ok(())
    }

    #[test]
    fn run_map() -> Result<(), CompileError> {
        // an envelope per step
        let mut prog = run_src("map <1 <2 _>> |n| n * (1 - clock)")?;
        for expected in [1.0, 0.75, 0.5, 0.25, 2.0, 1.0, 0.0, 0.0] {
            assert_approx_eq(render(&mut prog, FRAMES_PER_CYCLE * 2), expected);
        }

        Ok(())
    }

    #[test]
    fn run_filter_envelope() -> Result<(), CompileError> {
        let mut prog = run_src(
            "notes = <{100 500} {200 400} <{300 200} {500 800}>>\n\
            env = map notes |n| (1 - clock) ^ 3\n\
            osc = saw notes\n\
            osc -> mix -> lp (env * 1000 + 100) 2 * 0.5",
        )?;
        for _ in 0..SAMPLE_RATE * 2 {
            assert!(render(&mut prog, SAMPLE_RATE).is_finite());
        }

        Ok(())
    }

    #[test]
    fn run_block() -> Result<(), CompileError> {
        // each call has its own state
        let mut prog =
            run_src("counter = |step|\n  c = ~c + step\n  c\n{(counter 1) (counter 10)}")?;
        let mut output = [0.0; 2];
        for expected in [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]] {
            super::render(&mut prog, &mut output, SAMPLE_RATE);
            assert_eq!(output, expected);
        }

        // feedback fm
        let mut prog =
            run_src("fm_voice = |freq fb|\n  x = sin freq (~x * fb)\n  x\nfm_voice 100 0.5")?;
        for _ in 0..SAMPLE_RATE {
            assert!(render(&mut prog, SAMPLE_RATE).is_finite());
        }

        Ok(())
    }

    #[test]
    fn run_nan_safety() -> Result<(), CompileError> {
        // NaN from arithmetic doesn't persist in state or reach the output
        for src in [
            "x = ~x + 0 / 0\n~x",
            "1 -> lp (0 / 0)",
            "0 / 0 -> lp 100",
            "bpm = 0 / 0\n<1 2>",
            "0 / 0 -> rev (0 / 0) (0 / 0)",
            "0 / 0 -> bbd (0 / 0) (0 / 0)",
            "x = ~x[0 / 0] + 0 / 0\n~x[1]",
        ] {
            let mut prog = run_prelude_src(src)?;
            for _ in 0..4 {
                assert!(render(&mut prog, SAMPLE_RATE).is_finite(), "{src}");
            }
            assert!(prog.state.iter().all(|v| v.is_finite()), "{src}");
        }

        Ok(())
    }

    #[test]
    fn run_tap() -> Result<(), CompileError> {
        // a counter read 3 frames ago, starting from silence
        let mut prog = run_src("x = ~x + 1\n~x[3]")?;
        for expected in [0.0, 0.0, 0.0, 1.0, 2.0] {
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        // fractional delays are interpolated, delays below 1 frame are clamped
        let mut prog = run_src("x = ~x + 1\n{~x[1.5] ~x[0] ~x[-5]}")?;
        let mut output = [0.0; 3];
        for _ in 0..3 {
            super::render(&mut prog, &mut output, SAMPLE_RATE);
        }
        assert_eq!(output, [1.5, 2.0, 2.0]);

        // a stack of delays taps a single buffer
        let mut prog = run_src("x = ~x + 1\n{~x[1] ~x[2]}")?;
        for _ in 0..3 {
            super::render(&mut prog, &mut output[..2], SAMPLE_RATE);
        }
        assert_eq!(output[..2], [2.0, 1.0]);

        Ok(())
    }

    #[test]
    fn run_tap_voices() -> Result<(), CompileError> {
        // each voice and each call keeps its own buffer
        let mut prog = run_src("x = ~x + {1 10}\n~x[2]")?;
        let mut output = [0.0; 2];
        for _ in 0..3 {
            super::render(&mut prog, &mut output, SAMPLE_RATE);
        }
        assert_eq!(output, [1.0, 10.0]);

        let mut prog = run_src("f = |step|\n  c = ~c + step\n  ~c[2]\n{(f 1) (f 10)}")?;
        for _ in 0..3 {
            super::render(&mut prog, &mut output, SAMPLE_RATE);
        }
        assert_eq!(output, [1.0, 10.0]);

        Ok(())
    }

    #[test]
    fn run_tap_param() -> Result<(), CompileError> {
        // a stored name passed to a function refers to its buffer
        let mut prog = run_src("tap = |buf n| ~buf[n]\nx = ~x + 1\ntap x 2")?;
        for expected in [0.0, 0.0, 1.0, 2.0] {
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        // a stored argument keeps its own buffer
        let mut prog = run_src("tap = |x| ~x[2]\nx = ~x + 1\ntap (x * 10)")?;
        for expected in [0.0, 0.0, 10.0, 20.0] {
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        Ok(())
    }

    #[test]
    fn run_karplus_strong() -> Result<(), CompileError> {
        // a noise burst through a delay line of 100 frames and an averaging lowpass
        let src = "y = noise * (~n < 100) + (~y[100] + ~y[101]) * 0.5 * 0.99\nn = ~n + 1\ny";
        let mut prog = run_src(src)?;
        let frames: Vec<_> = (0..4000).map(|_| render(&mut prog, SAMPLE_RATE)).collect();
        let energy =
            |range: std::ops::Range<usize>| -> f32 { frames[range].iter().map(|v| v * v).sum() };
        // periodic with a period of 100.5 frames, and decaying
        let correlation: f32 = (2000..3000).map(|i| frames[i] * frames[i + 201]).sum();
        assert!(correlation > energy(2000..3000) * 0.5);
        assert!(energy(3000..4000) < energy(0..1000));
        assert!(energy(3000..4000) > 0.0);

        Ok(())
    }

    #[test]
    fn run_pluck() -> Result<(), CompileError> {
        // plucked when the gate opens, at the start of each step
        let mut prog = run_prelude_src("pluck <c3 e3 g3 c4> ((clock * 4 % 1) < 0.5) 0.996")?;
        let bar = SAMPLE_RATE as usize * 2;
        let frames: Vec<_> = (0..bar).map(|_| render(&mut prog, SAMPLE_RATE)).collect();
        let energy: Vec<f32> = frames
            .chunks(bar / 8)
            .map(|chunk| chunk.iter().map(|v| v * v).sum())
            .collect();
        for step in 1..4 {
            assert!(energy[step * 2] > energy[step * 2 - 1] * 2.0, "{energy:?}");
        }

        // silent while the gate is closed
        let mut prog = run_prelude_src("pluck c3 0 0.996")?;
        for _ in 0..1000 {
            assert_eq!(render(&mut prog, SAMPLE_RATE), 0.0);
        }

        Ok(())
    }

    #[test]
    fn run_math() -> Result<(), CompileError> {
        for (src, expected) in [
            ("floor 1.5", 1.0),
            ("floor (-1.5)", -2.0),
            ("tanh 0", 0.0),
            ("min 1 2", 1.0),
            ("max 1 2", 2.0),
            ("sr", SAMPLE_RATE as f32),
            ("bar", 2.0),
            ("bpm = 240\nbar", 1.0),
        ] {
            let mut prog = run_src(src)?;
            assert_approx_eq(render(&mut prog, SAMPLE_RATE), expected);
        }

        Ok(())
    }

    #[test]
    fn run_pitch() -> Result<(), CompileError> {
        for (src, expected) in [
            ("a4", 440.0),
            ("a4 + 12", 880.0),
            ("a5 - 12", 440.0),
            ("a4 * 2", 880.0),
            ("e4 - c4", 4.0),
        ] {
            let mut prog = run_src(src)?;
            assert!(
                (render(&mut prog, SAMPLE_RATE) - expected).abs() < 1e-3,
                "{src}"
            );
        }

        // transposed seqs keep their rests
        let mut prog = run_src("ramp 1 -> <a4 _> + 12")?;
        for expected in [880.0, 880.0, 0.0, 0.0] {
            assert!((render(&mut prog, 4) - expected).abs() < 1e-3);
        }

        // chords are stacks
        let mut prog = run_src("A")?;
        let mut output = [0.0; 3];
        super::render(&mut prog, &mut output, SAMPLE_RATE);
        assert!((output[0] - 220.0).abs() < 1e-3);

        Ok(())
    }

    #[test]
    fn run_bpm() -> Result<(), CompileError> {
        // twice the default bpm takes 2 frames per cycle
        let mut prog = run_src("bpm = 240\n<100 200>")?;
        for expected in [100.0, 200.0, 100.0] {
            assert_approx_eq(render(&mut prog, FRAMES_PER_CYCLE), expected);
        }

        Ok(())
    }

    #[test]
    fn run_destructuring() -> Result<(), CompileError> {
        let mut prog = run_src("{x y} = {1 2}\n{y x}")?;
        let mut output = [0.0; 2];
        super::render(&mut prog, &mut output, SAMPLE_RATE);
        assert_eq!(output, [2.0, 1.0]);

        // voices refer to each other's previous values
        let mut prog = run_src("{x y} = {(~y + 1) (~x + 2)}\n{x y}")?;
        for expected in [[1.0, 2.0], [3.0, 3.0], [4.0, 5.0]] {
            super::render(&mut prog, &mut output, SAMPLE_RATE);
            assert_eq!(output, expected);
        }

        Ok(())
    }

    #[test]
    fn run_mix() -> Result<(), CompileError> {
        let mut prog = run_src("mix {1 2 3}")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 6.0);

        // rests are silent
        let mut prog = run_src("mix {1 _ 3}")?;
        assert_approx_eq(render(&mut prog, SAMPLE_RATE), 4.0);

        Ok(())
    }

    #[test]
    fn run_pan() -> Result<(), CompileError> {
        let mut output = [0.0; 2];
        for (src, expected) in [
            ("1 -> pan (-1)", [1.0, 0.0]),
            ("1 -> pan 1", [0.0, 1.0]),
            ("1 -> pan 0", [0.5_f32.sqrt(), 0.5_f32.sqrt()]),
            // out of range positions are clamped
            ("1 -> pan 2", [0.0, 1.0]),
            // a position per voice, mixed into each side
            ("{1 2} -> pan {(-1) 1}", [1.0, 2.0]),
        ] {
            let mut prog = run_prelude_src(src)?;
            super::render(&mut prog, &mut output, SAMPLE_RATE);
            assert_eq!(output, expected, "{src}");
        }

        Ok(())
    }

    #[test]
    fn run_svf() -> Result<(), CompileError> {
        // lowpass passes dc, highpass and bandpass block it
        for (src, expected) in [
            ("1 -> lp 100", 1.0),
            ("1 -> hp 100", 0.0),
            ("1 -> bp 100", 0.0),
        ] {
            let mut prog = run_src(src)?;
            let mut out = 0.0;
            for _ in 0..SAMPLE_RATE {
                out = render(&mut prog, SAMPLE_RATE);
            }
            assert!((out - expected).abs() < 1e-3, "{src}: {out}");
        }

        // rests don't poison the filter state
        let mut prog = run_src("_ -> lp 100")?;
        render(&mut prog, SAMPLE_RATE);
        assert!(prog.state.iter().all(|v| v.is_finite()));

        Ok(())
    }

    #[test]
    fn run_stack_ref() -> Result<(), CompileError> {
        let mut prog = run_src("x = {1 2}\n~x")?;
        let mut output = [0.0; 2];
        super::render(&mut prog, &mut output, SAMPLE_RATE);
        assert_eq!(output, [0.0, 0.0]);
        super::render(&mut prog, &mut output, SAMPLE_RATE);
        assert_eq!(output, [1.0, 2.0]);

        // each voice refers to itself
        let mut prog = run_src("x = {1 2} + ~x\nx")?;
        for expected in [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]] {
            super::render(&mut prog, &mut output, SAMPLE_RATE);
            assert_eq!(output, expected);
        }

        Ok(())
    }

    #[test]
    fn render_channels() -> Result<(), CompileError> {
        let cases: [(&str, &[f32]); 7] = [
            // mono fills every channel
            ("1", &[1.0, 1.0]),
            ("{1 2}", &[1.0, 2.0]),
            // fewer voices than channels wrap
            ("{1 2}", &[1.0, 2.0, 1.0, 2.0]),
            ("{1 2 3}", &[1.0, 2.0, 3.0, 1.0]),
            // more voices than channels are dropped
            ("{1 2 3}", &[1.0, 2.0]),
            // rest outputs silence
            ("{_ 2}", &[0.0, 2.0]),
            // top level expressions are mixed
            ("1\n{2 3}", &[3.0, 4.0]),
        ];
        for (src, expected) in cases {
            let mut prog = run_src(src)?;
            let mut output = vec![f32::NAN; expected.len()];
            super::render(&mut prog, &mut output, SAMPLE_RATE);
            assert_eq!(output, expected, "{src}");
        }

        Ok(())
    }

    #[test]
    fn run_invalid_identifier() {
        match run_src("x = 1\ny + 1") {
            Err(CompileError::UnresolvedIdentifier(name)) => assert_eq!(name, "y"),
            _ => panic!(),
        }
    }

    #[test]
    fn preserve_state() -> Result<(), CompileError> {
        let mut old = run_src("x = ~x + 1\nx")?;
        assert_approx_eq(render(&mut old, SAMPLE_RATE), 1.0);
        let mut new = run_src("x = ~x + 1\nx")?;
        new.copy_state_from(&old);
        assert_eq!(old.slots, new.slots);

        Ok(())
    }

    #[test]
    fn preserve_ambient_phase() -> Result<(), CompileError> {
        let sample_rate = FRAMES_PER_CYCLE;
        let mut old = run_src("<100 200>")?;
        render(&mut old, sample_rate);
        render(&mut old, sample_rate);
        let mut new = run_src("<300 400>")?;
        new.copy_state_from(&old);
        assert_approx_eq(render(&mut new, sample_rate), 400.0);

        // position spans multiple bars
        let mut old = run_src("clock / 4")?;
        for _ in 0..6 {
            render(&mut old, sample_rate);
        }
        let mut new = run_src("clock / 4")?;
        new.copy_state_from(&old);
        assert_approx_eq(render(&mut new, sample_rate), 1.5 / 4.0);

        Ok(())
    }

    #[test]
    fn preserve_sin_phase() -> Result<(), CompileError> {
        let mut old = run_src("sin 440")?;
        assert_approx_eq(render(&mut old, SAMPLE_RATE), 0.0);
        let mut new = run_src("sin 440")?;
        new.copy_state_from(&old);
        assert_eq!(old.slots, new.slots);

        Ok(())
    }

    #[test]
    fn preserve_sin_phase_after_freq_change() -> Result<(), CompileError> {
        let mut old = run_src("sin 440")?;
        assert_approx_eq(render(&mut old, SAMPLE_RATE), 0.0);
        let mut new = run_src("sin 220")?;
        new.copy_state_from(&old);
        assert_eq!(old.slots, new.slots);

        Ok(())
    }

    #[test]
    fn preserve_sin_phase_in_graph() -> Result<(), CompileError> {
        // nb: this requires the AST paths in `compile_expr`
        let mut old = run_src(
            "freq = 1\n\
            lfo = sin freq\n\
            sin (440 + lfo * 100)",
        )?;
        assert_approx_eq(render(&mut old, SAMPLE_RATE), 0.0);
        let mut new = run_src(
            "freq = 2\n\
            lfo = sin freq\n\
            sin (440 + lfo * 100)",
        )?;
        new.copy_state_from(&old);
        assert_eq!(old.slots, new.slots);

        Ok(())
    }
}
