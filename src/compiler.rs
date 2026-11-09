use crate::{
    ast::{BinaryOp, Builtin, UnaryOp},
    env::Resolved,
    ir::IR,
    vm::{FilterMode, OpCode, Program, RING_LEN, Waveform},
};
use std::{
    collections::{HashMap, HashSet},
    fmt::{Display, Formatter},
};

#[derive(Debug)]
pub(crate) enum CompileError {
    UnresolvedIdentifier(String),
    UnexpectedExpression(String),
    NotCallable(String),
}

impl Display for CompileError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::UnresolvedIdentifier(name) => write!(f, "unresolved name: {name:?}"),
            CompileError::UnexpectedExpression(name) => write!(f, "unexpected expression: {name}"),
            CompileError::NotCallable(name) => write!(f, "not callable: {name}"),
        }
    }
}

impl std::error::Error for CompileError {}

pub(crate) fn compile(prog: &mut Program, resolved: &Resolved) -> Result<(), CompileError> {
    let mut refs = HashSet::from(["bpm".to_string()]);
    // ring buffer length of each tapped name
    let mut taps = HashMap::new();
    let stored = resolved.stores.iter().flat_map(|(_, voices)| voices);
    for expr in stored.chain(resolved.outputs.iter().flatten()) {
        collect_refs(&mut refs, &mut taps, expr);
    }

    // extra compiler pass to store cycles (~)
    for (name, voices) in &resolved.stores {
        if refs.contains(name) {
            for i in 0..voices.len() {
                let key = format!("{name}/{i}");
                if !prog.slots.contains_key(&key) {
                    prog.insert_state(key.clone());
                }
                if let Some(len) = taps.get(name)
                    && prog.region(&format!("~{key}")).is_none()
                {
                    prog.insert_buffer(format!("~{key}"), *len);
                }
            }
        }
    }

    for (name, voices) in &resolved.stores {
        if refs.contains(name) {
            for (i, expr) in voices.iter().enumerate() {
                let key = format!("{name}/{i}");
                let slot = prog.slots[&key];
                compile_expr(prog, expr, &key, i)?;
                if let Some((start, len)) = prog.region(&format!("~{key}")) {
                    prog.ops.push(OpCode::Write {
                        start,
                        mask: len - 1,
                    });
                }
                prog.ops.push(OpCode::Store(slot));
            }
        }
    }
    prog.bpm_slot = prog.slots.get("bpm/0").copied();

    let outputs: Vec<_> = resolved
        .outputs
        .iter()
        .filter(|voices| !voices.is_empty())
        .collect();
    let channels = outputs.iter().map(|voices| voices.len()).max().unwrap_or(0);
    for i in 0..channels {
        let mut voices: Vec<_> = outputs.iter().map(|v| v[i % v.len()].clone()).collect();
        let voice = match voices.len() {
            1 => voices.remove(0),
            _ => IR::Mix(voices),
        };
        compile_expr(prog, &voice, &format!("output/{i}"), i)?;
    }
    prog.channels = channels;

    Ok(())
}

fn collect_refs(refs: &mut HashSet<String>, taps: &mut HashMap<String, usize>, expr: &IR) {
    match expr {
        IR::Ref { name, .. } => {
            refs.insert(name.into());
        }
        IR::Tap { name, index, .. } => {
            refs.insert(name.into());
            // a constant delay only needs a ring buffer just long enough for it
            let len = match **index {
                IR::Num(delay) => (delay.max(1.0).ceil() as usize + 2)
                    .next_power_of_two()
                    .min(RING_LEN),
                _ => RING_LEN,
            };
            let max = taps.entry(name.into()).or_insert(len);
            *max = (*max).max(len);
            collect_refs(refs, taps, index);
        }
        IR::Builtin { args, .. } | IR::Mix(args) => {
            args.iter().for_each(|arg| collect_refs(refs, taps, arg))
        }
        IR::Unary(_, operand) => collect_refs(refs, taps, operand),
        IR::Binary(lhs, _, rhs) => {
            collect_refs(refs, taps, lhs);
            collect_refs(refs, taps, rhs);
        }
        IR::Seq { items, input } => {
            collect_refs(refs, taps, input);
            items.iter().for_each(|item| collect_refs(refs, taps, item));
        }
        IR::Pitch(pitch) => collect_refs(refs, taps, pitch),
        IR::Rest | IR::Num(..) | IR::Clock(..) => {}
    }
}

fn compile_expr(
    prog: &mut Program,
    expr: &IR,
    path: &str,
    voice: usize,
) -> Result<(), CompileError> {
    match expr {
        IR::Rest => prog.ops.push(OpCode::Num(0.0)),
        IR::Num(num) => prog.ops.push(OpCode::Num(*num)),
        IR::Clock(rate) => prog.ops.push(OpCode::Clock { rate: *rate }),
        IR::Pitch(pitch) => {
            compile_expr(prog, pitch, path, voice)?;
            prog.ops.push(OpCode::Mtof);
        }
        IR::Ref { name, voice: index } => {
            let key = format!("{name}/{}", index.unwrap_or(voice));
            if let Some(slot) = prog.slots.get(&key) {
                prog.ops.push(OpCode::Load(*slot));
            } else {
                return Err(CompileError::UnresolvedIdentifier(name.into()));
            };
        }
        IR::Tap {
            name,
            voice: index_voice,
            index,
        } => {
            compile_expr(prog, index, &format!("{path}/index"), voice)?;
            let key = format!("~{name}/{}", index_voice.unwrap_or(voice));
            let Some((start, len)) = prog.region(&key) else {
                return Err(CompileError::UnresolvedIdentifier(name.into()));
            };
            prog.ops.push(OpCode::Tap {
                start,
                mask: len - 1,
            });
        }
        IR::Unary(op, operand) => {
            compile_expr(prog, operand, &format!("{path}/operand"), voice)?;
            prog.ops.push(OpCode::Unary(*op));
        }
        IR::Binary(lhs, op, rhs) => {
            compile_expr(prog, lhs, &format!("{path}/lhs"), voice)?;
            compile_expr(prog, rhs, &format!("{path}/rhs"), voice)?;
            prog.ops.push(OpCode::Binary(*op));
        }
        IR::Builtin { kind, args } => compile_builtin(prog, kind, args, path, voice)?,
        IR::Seq { items, input } => {
            compile_expr(prog, input, &format!("{path}/input"), voice)?;
            prog.ops.push(OpCode::PushPhase {
                length: items.len(),
            });
            for (i, item) in items.iter().enumerate().rev() {
                compile_expr(prog, item, &format!("{path}/item/{i}"), voice)?;
            }
            prog.ops.push(OpCode::Seq {
                length: items.len(),
            });
        }
        IR::Mix(items) => {
            for (i, item) in items.iter().enumerate() {
                compile_expr(prog, item, &format!("{path}/item/{i}"), voice)?;
            }
            prog.ops.push(OpCode::Mix {
                length: items.len(),
            });
        }
    }
    Ok(())
}

fn compile_builtin(
    prog: &mut Program,
    kind: &Builtin,
    args: &[IR],
    path: &str,
    voice: usize,
) -> Result<(), CompileError> {
    for (i, default) in kind.defaults().iter().enumerate() {
        let arg = if let Some(arg) = args.get(i) {
            arg
        } else {
            &IR::Num(*default)
        };
        compile_expr(prog, arg, &format!("{path}/{i}"), voice)?;
    }

    match kind {
        Builtin::Ramp => compile_osc(prog, Waveform::Ramp, path),
        Builtin::Sin => compile_osc(prog, Waveform::Sin, path),
        Builtin::Saw => compile_osc(prog, Waveform::Saw, path),
        Builtin::Square => compile_osc(prog, Waveform::Square, path),
        Builtin::Tri => compile_osc(prog, Waveform::Triangle, path),
        Builtin::Noise => prog.ops.push(OpCode::Noise),
        Builtin::Print => prog.ops.push(OpCode::Print),
        Builtin::Lp => compile_svf(prog, FilterMode::Lp, path),
        Builtin::Bp => compile_svf(prog, FilterMode::Bp, path),
        Builtin::Hp => compile_svf(prog, FilterMode::Hp, path),
        Builtin::Floor => prog.ops.push(OpCode::Unary(UnaryOp::Floor)),
        Builtin::Tanh => prog.ops.push(OpCode::Unary(UnaryOp::Tanh)),
        Builtin::Abs => prog.ops.push(OpCode::Unary(UnaryOp::Abs)),
        Builtin::Min => prog.ops.push(OpCode::Binary(BinaryOp::Min)),
        Builtin::Max => prog.ops.push(OpCode::Binary(BinaryOp::Max)),
        Builtin::Sr => prog.ops.push(OpCode::SampleRate),
        Builtin::Bar => prog.ops.push(OpCode::Bar),
        Builtin::Mix | Builtin::Clock => {
            return Err(CompileError::UnexpectedExpression(kind.to_string()));
        }
    }

    Ok(())
}

fn compile_svf(prog: &mut Program, mode: FilterMode, path: &str) {
    let state_slot = prog.insert_state(format!("{path}/svf/ic1eq"));
    prog.insert_state(format!("{path}/svf/ic2eq"));
    prog.ops.push(OpCode::Svf { mode, state_slot });
}

fn compile_osc(prog: &mut Program, waveform: Waveform, path: &str) {
    let phase_slot = prog.insert_state(format!("{path}/osc"));
    let opcode = OpCode::Osc {
        waveform,
        phase_slot,
    };
    prog.ops.push(opcode);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{env::Env, lexer::Lexer, parser::Parser};

    pub(crate) fn compile_src(src: &str) -> Result<Program, CompileError> {
        let mut parser = Parser::new(Lexer::new(src));
        let items = parser.parse().unwrap();

        let mut env = Env::new();
        let items = env.resolve(&items)?;

        let mut prog = Program::new();
        compile(&mut prog, &items)?;

        Ok(prog)
    }

    #[test]
    fn compile_add() -> Result<(), CompileError> {
        let runtime = compile_src("1 + 2")?;
        assert_eq!(
            runtime.ops,
            vec![
                OpCode::Num(1.0),
                OpCode::Num(2.0),
                OpCode::Binary(BinaryOp::Add)
            ]
        );

        Ok(())
    }

    #[test]
    fn compile_sin() -> Result<(), CompileError> {
        let prog = compile_src("sin 300 0.5")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(300.0),
                OpCode::Num(0.5),
                OpCode::Osc {
                    waveform: Waveform::Sin,
                    phase_slot: 0
                }
            ]
        );

        Ok(())
    }

    #[test]
    fn compile_sin_default_args() -> Result<(), CompileError> {
        let prog = compile_src("sin")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(100.0),
                OpCode::Num(0.0),
                OpCode::Osc {
                    waveform: Waveform::Sin,
                    phase_slot: 0
                }
            ]
        );
        let prog = compile_src("sin 200")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(200.0),
                OpCode::Num(0.0),
                OpCode::Osc {
                    waveform: Waveform::Sin,
                    phase_slot: 0
                }
            ]
        );

        Ok(())
    }

    #[test]
    fn compile_print() -> Result<(), CompileError> {
        let prog = compile_src("print 100")?;
        assert_eq!(prog.ops, vec![OpCode::Num(100.0), OpCode::Print]);

        // missing print argument defaults to 0
        let prog = compile_src("print")?;
        assert_eq!(prog.ops, vec![OpCode::Num(0.0), OpCode::Print]);

        Ok(())
    }

    #[test]
    fn compile_seq_call() -> Result<(), CompileError> {
        let prog = compile_src("0 -> <100 200>")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(0.0),
                OpCode::PushPhase { length: 2 },
                OpCode::Num(200.0),
                OpCode::Num(100.0),
                OpCode::Seq { length: 2 }
            ]
        );

        let prog = compile_src("0 -> <100>")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(0.0),
                OpCode::PushPhase { length: 1 },
                OpCode::Num(100.0),
                OpCode::Seq { length: 1 }
            ]
        );

        let prog = compile_src("1 -> <>")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(1.0),
                OpCode::PushPhase { length: 0 },
                OpCode::Seq { length: 0 }
            ]
        );

        Ok(())
    }

    #[test]
    fn compile_cycle() -> Result<(), CompileError> {
        let prog = compile_src("x = 5\nadd = |y| ~x + y\nadd 5")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(5.0),
                OpCode::Write { start: 0, mask: 3 },
                OpCode::Store(0),
                OpCode::Num(1.0),
                OpCode::Tap { start: 0, mask: 3 },
                OpCode::Num(5.0),
                OpCode::Binary(BinaryOp::Add)
            ]
        );

        Ok(())
    }

    #[test]
    fn compile_reassignment() -> Result<(), CompileError> {
        let prog = compile_src("x = 1\nx = 2\nx")?;
        assert_eq!(prog.ops, vec![OpCode::Num(2.0),]);

        Ok(())
    }

    #[test]
    fn compile_seq() -> Result<(), CompileError> {
        // uncalled seq is driven by the ambient phase
        let prog = compile_src("<100 200>")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Clock { rate: 1.0 },
                OpCode::PushPhase { length: 2 },
                OpCode::Num(200.0),
                OpCode::Num(100.0),
                OpCode::Seq { length: 2 }
            ]
        );

        let prog = compile_src("<(100 + 200) 200>")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Clock { rate: 1.0 },
                OpCode::PushPhase { length: 2 },
                OpCode::Num(200.0),
                OpCode::Num(100.0),
                OpCode::Num(200.0),
                OpCode::Binary(BinaryOp::Add),
                OpCode::Seq { length: 2 }
            ]
        );

        Ok(())
    }

    #[test]
    fn compile_stack() -> Result<(), CompileError> {
        let prog = compile_src("{100 200}")?;
        assert_eq!(prog.ops, vec![OpCode::Num(100.0), OpCode::Num(200.0)]);

        let prog = compile_src("{(100 + 200) 200}")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(100.0),
                OpCode::Num(200.0),
                OpCode::Binary(BinaryOp::Add),
                OpCode::Num(200.0),
            ]
        );

        Ok(())
    }

    #[test]
    fn compile_nested_seq() -> Result<(), CompileError> {
        let prog = compile_src("0 -> <100 <200 300>>")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(0.0),
                OpCode::PushPhase { length: 2 },
                // phase of the outer seq's step
                OpCode::Clock { rate: 1.0 },
                OpCode::PushPhase { length: 2 },
                OpCode::Num(300.0),
                OpCode::Num(200.0),
                OpCode::Seq { length: 2 },
                OpCode::Num(100.0),
                OpCode::Seq { length: 2 }
            ]
        );

        Ok(())
    }

    #[test]
    fn compile_bpm() -> Result<(), CompileError> {
        // bpm is always stored, even when not referenced
        let prog = compile_src("bpm = 140")?;
        assert_eq!(prog.ops, vec![OpCode::Num(140.0), OpCode::Store(0)]);
        assert_eq!(prog.bpm_slot, Some(0));

        Ok(())
    }

    #[test]
    fn compile_svf() -> Result<(), CompileError> {
        let prog = compile_src("0 -> lp 800 2")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(0.0),
                OpCode::Num(800.0),
                OpCode::Num(2.0),
                OpCode::Svf {
                    mode: FilterMode::Lp,
                    state_slot: 0
                }
            ]
        );

        Ok(())
    }

    #[test]
    fn compile_stack_ref() -> Result<(), CompileError> {
        // a slot per voice
        let prog = compile_src("x = {1 2}\n~x")?;
        assert_eq!(
            prog.ops,
            vec![
                OpCode::Num(1.0),
                OpCode::Write { start: 0, mask: 3 },
                OpCode::Store(0),
                OpCode::Num(2.0),
                OpCode::Write { start: 4, mask: 3 },
                OpCode::Store(1),
                OpCode::Num(1.0),
                OpCode::Tap { start: 0, mask: 3 },
                OpCode::Num(1.0),
                OpCode::Tap { start: 4, mask: 3 }
            ]
        );

        Ok(())
    }
}
