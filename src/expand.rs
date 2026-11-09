use crate::{
    ast::{BinaryOp, Builtin, Expr},
    compiler::CompileError,
    ir::IR,
};
use std::collections::HashMap;

pub(crate) fn expand(expr: Expr, widths: &HashMap<String, usize>) -> Result<Vec<IR>, CompileError> {
    Ok(match expr {
        Expr::Rest => vec![IR::Rest],
        Expr::Pitch(pitch) => vec![IR::Pitch(IR::Num(pitch).into())],
        Expr::Num(num) => vec![IR::Num(num)],
        Expr::Ref(name) => match widths.get(&name) {
            Some(width) => (0..*width)
                .map(|i| IR::Ref {
                    name: name.clone(),
                    voice: Some(i),
                })
                .collect(),
            None => vec![IR::Ref { name, voice: None }],
        },
        Expr::Tap { name, index } => {
            zip(&[expand(Expr::Ref(name), widths)?, expand(*index, widths)?])
                .into_iter()
                .map(|row| {
                    let [IR::Ref { name, voice }, index]: [IR; 2] = row.try_into().unwrap() else {
                        unreachable!("a ref expands into refs")
                    };
                    IR::Tap {
                        name,
                        voice,
                        index: index.into(),
                    }
                })
                .collect()
        }
        Expr::Ident(..) | Expr::Fn { .. } => {
            return Err(CompileError::UnexpectedExpression(format!("{expr:?}")));
        }
        Expr::Stack(items) => expand_all(items, widths)?.concat(),
        Expr::Unary(op, operand) => expand(*operand, widths)?
            .into_iter()
            .map(|operand| IR::Unary(op, operand.into()))
            .collect(),
        Expr::Binary(lhs, op, rhs) => zip(&[expand(*lhs, widths)?, expand(*rhs, widths)?])
            .into_iter()
            .map(|row| {
                let [lhs, rhs] = row.try_into().unwrap();
                match (lhs, op, rhs) {
                    (IR::Clock(rate), BinaryOp::Div, IR::Num(num)) if num != 0.0 => {
                        IR::Clock(rate / num)
                    }
                    (IR::Pitch(lhs), BinaryOp::Sub, IR::Pitch(rhs)) => {
                        IR::Binary(lhs, BinaryOp::Sub, rhs)
                    }
                    (lhs, op @ (BinaryOp::Add | BinaryOp::Sub), rhs)
                        if is_pitch(&lhs) && !is_pitch(&rhs) =>
                    {
                        transpose(lhs, op, &rhs)
                    }
                    (lhs, BinaryOp::Add, rhs) if is_pitch(&rhs) && !is_pitch(&lhs) => {
                        transpose(rhs, BinaryOp::Add, &lhs)
                    }
                    (lhs, op, rhs) => IR::Binary(lhs.into(), op, rhs.into()),
                }
            })
            .collect(),
        Expr::Seq(items) => expand_seq(items, widths)?
            .into_iter()
            .map(|items| IR::Seq {
                items,
                input: IR::Clock(1.0).into(),
            })
            .collect(),
        Expr::Builtin {
            kind: Builtin::Clock,
            ..
        } => vec![IR::Clock(1.0)],
        Expr::Builtin {
            kind: Builtin::Mix,
            args,
        } => vec![IR::Mix(expand_all(args, widths)?.concat())],
        Expr::Builtin { kind, args } => zip(&expand_all(args, widths)?)
            .into_iter()
            .map(|args| IR::Builtin { kind, args })
            .collect(),
        Expr::Call { callee, args } => {
            let [input]: [Expr; 1] = args
                .try_into()
                .map_err(|args| CompileError::UnexpectedExpression(format!("{args:?}")))?;
            let callees = expand_callee(*callee, widths)?;
            let inputs = expand(input, widths)?;
            if callees.is_empty() || inputs.is_empty() {
                return Ok(vec![]);
            }
            (0..callees.len().max(inputs.len()))
                .map(|i| IR::Seq {
                    items: callees[i % callees.len()].clone(),
                    input: inputs[i % inputs.len()].clone().into(),
                })
                .collect()
        }
    })
}

pub(crate) fn storable(voices: &[IR]) -> bool {
    !voices.iter().all(is_trivial)
        && !voices
            .iter()
            .any(|voice| is_pitch(voice) || is_contextual(voice))
}

fn is_trivial(voice: &IR) -> bool {
    match voice {
        IR::Rest | IR::Num(..) | IR::Ref { .. } => true,
        IR::Unary(_, operand) => is_trivial(operand),
        _ => false,
    }
}

fn is_contextual(voice: &IR) -> bool {
    match voice {
        IR::Clock(..) | IR::Seq { .. } => true,
        IR::Rest | IR::Num(..) | IR::Ref { .. } => false,
        IR::Pitch(operand) | IR::Unary(_, operand) => is_contextual(operand),
        IR::Binary(lhs, _, rhs) => is_contextual(lhs) || is_contextual(rhs),
        IR::Builtin { args: items, .. } | IR::Mix(items) => items.iter().any(is_contextual),
        IR::Tap { index, .. } => is_contextual(index),
    }
}

fn is_pitch(voice: &IR) -> bool {
    match voice {
        IR::Pitch(..) => true,
        IR::Seq { items, .. } => {
            items.iter().all(|item| *item == IR::Rest || is_pitch(item))
                && items.iter().any(is_pitch)
        }
        _ => false,
    }
}

fn transpose(voice: IR, op: BinaryOp, interval: &IR) -> IR {
    match voice {
        IR::Pitch(pitch) => IR::Pitch(IR::Binary(pitch, op, interval.clone().into()).into()),
        IR::Seq { items, input } => IR::Seq {
            items: items
                .into_iter()
                .map(|item| transpose(item, op, interval))
                .collect(),
            input,
        },
        voice => voice,
    }
}

fn expand_all(
    exprs: Vec<Expr>,
    widths: &HashMap<String, usize>,
) -> Result<Vec<Vec<IR>>, CompileError> {
    exprs.into_iter().map(|expr| expand(expr, widths)).collect()
}

fn expand_seq(
    items: Vec<Expr>,
    widths: &HashMap<String, usize>,
) -> Result<Vec<Vec<IR>>, CompileError> {
    let cols = expand_all(items, widths)?;
    let voices = cols.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut cols: Vec<_> = cols.into_iter().map(Vec::into_iter).collect();
    Ok((0..voices)
        .map(|_| {
            cols.iter_mut()
                .map(|col| col.next().unwrap_or(IR::Rest))
                .collect()
        })
        .collect())
}

fn expand_callee(
    expr: Expr,
    widths: &HashMap<String, usize>,
) -> Result<Vec<Vec<IR>>, CompileError> {
    match expr {
        Expr::Seq(items) => expand_seq(items, widths),
        Expr::Stack(items) => Ok(items
            .into_iter()
            .map(|item| expand_callee(item, widths))
            .collect::<Result<Vec<_>, _>>()?
            .concat()),
        callee => Err(CompileError::NotCallable(format!("{callee:?}"))),
    }
}

fn zip(cols: &[Vec<IR>]) -> Vec<Vec<IR>> {
    if cols.iter().any(Vec::is_empty) {
        return vec![];
    }
    let voices = cols.iter().map(Vec::len).max().unwrap_or(1);
    (0..voices)
        .map(|i| cols.iter().map(|col| col[i % col.len()].clone()).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::UnaryOp;

    fn expand(expr: Expr) -> Result<Vec<IR>, CompileError> {
        super::expand(expr, &HashMap::new())
    }

    #[test]
    fn expand_rest() {
        assert_eq!(expand(Expr::Rest).unwrap(), vec![IR::Rest]);
    }

    #[test]
    fn expand_num() {
        assert_eq!(expand(Expr::Num(0.0)).unwrap(), vec![IR::Num(0.0)]);
    }

    #[test]
    fn expand_ident() {
        let expr = Expr::Ident("x".into());
        assert!(matches!(
            expand(expr),
            Err(CompileError::UnexpectedExpression(_))
        ));
    }

    #[test]
    fn expand_ref() {
        // unknown width refers to the enclosing voice
        assert_eq!(
            expand(Expr::Ref("x".into())).unwrap(),
            vec![IR::Ref {
                name: "x".into(),
                voice: None
            }]
        );

        // known width expands into a voice per stored voice
        let widths = HashMap::from([("x".into(), 2)]);
        assert_eq!(
            super::expand(Expr::Ref("x".into()), &widths).unwrap(),
            vec![
                IR::Ref {
                    name: "x".into(),
                    voice: Some(0)
                },
                IR::Ref {
                    name: "x".into(),
                    voice: Some(1)
                }
            ]
        );
    }

    #[test]
    fn expand_fn() {
        let expr = Expr::Fn {
            params: vec![],
            defaults: vec![],
            assigns: vec![],
            body: Expr::Num(0.0).into(),
        };
        assert!(matches!(
            expand(expr),
            Err(CompileError::UnexpectedExpression(_))
        ));
    }

    #[test]
    fn expand_stack() {
        // stack should be removed from expanded result
        let expr = Expr::Stack(vec![Expr::Num(0.0)]);
        assert_eq!(expand(expr).unwrap(), vec![IR::Num(0.0)]);
    }

    #[test]
    fn expand_nested_stack() {
        // nested stack should be flattened and removed from expanded result
        let expr = Expr::Stack(vec![Expr::Stack(vec![Expr::Num(0.0)])]);
        assert_eq!(expand(expr).unwrap(), vec![IR::Num(0.0)]);
    }

    #[test]
    fn expand_unary() {
        let expr = Expr::Unary(UnaryOp::Negate, Expr::Num(0.0).into());
        assert_eq!(
            expand(expr).unwrap(),
            vec![IR::Unary(UnaryOp::Negate, IR::Num(0.0).into())]
        );

        // remove stack from unary operand
        let expr = Expr::Unary(UnaryOp::Negate, Expr::Stack(vec![Expr::Num(0.0)]).into());
        assert_eq!(
            expand(expr).unwrap(),
            vec![IR::Unary(UnaryOp::Negate, IR::Num(0.0).into())]
        );

        // flatten stack as operand into multiple operations
        let expr = Expr::Unary(
            UnaryOp::Negate,
            Expr::Stack(vec![Expr::Num(0.0), Expr::Num(1.0)]).into(),
        );
        assert_eq!(
            expand(expr).unwrap(),
            vec![
                IR::Unary(UnaryOp::Negate, IR::Num(0.0).into()),
                IR::Unary(UnaryOp::Negate, IR::Num(1.0).into())
            ]
        );
    }

    #[test]
    fn expand_binary() {
        let expr = Expr::Binary(Expr::Num(1.0).into(), BinaryOp::Add, Expr::Num(1.0).into());
        assert_eq!(
            expand(expr).unwrap(),
            vec![IR::Binary(
                IR::Num(1.0).into(),
                BinaryOp::Add,
                IR::Num(1.0).into()
            )]
        );

        // flatten stacks as binary operands into multiple operations
        let expr = Expr::Binary(
            Expr::Stack(vec![Expr::Num(1.0), Expr::Num(2.0)]).into(),
            BinaryOp::Add,
            Expr::Stack(vec![Expr::Num(1.0)]).into(),
        );
        assert_eq!(
            expand(expr).unwrap(),
            vec![
                IR::Binary(IR::Num(1.0).into(), BinaryOp::Add, IR::Num(1.0).into()),
                IR::Binary(IR::Num(2.0).into(), BinaryOp::Add, IR::Num(1.0).into()),
            ]
        );

        // if one of the arms is an empty stack, return nothing
        let expr = Expr::Binary(
            Expr::Stack(vec![]).into(),
            BinaryOp::Add,
            Expr::Stack(vec![Expr::Num(1.0)]).into(),
        );
        assert_eq!(expand(expr).unwrap(), vec![]);
    }

    #[test]
    fn expand_seq() {
        // uncalled seq is driven by the ambient phase
        let expr = Expr::Seq(vec![Expr::Num(0.0)]);
        assert_eq!(
            expand(expr).unwrap(),
            vec![IR::Seq {
                items: vec![IR::Num(0.0)],
                input: IR::Clock(1.0).into()
            }]
        );

        // stacks in a seq split it into voices, padded with rests
        let expr = Expr::Seq(vec![
            Expr::Stack(vec![Expr::Num(1.0)]),
            Expr::Stack(vec![Expr::Num(2.0), Expr::Num(3.0)]),
        ]);
        assert_eq!(
            expand(expr).unwrap(),
            vec![
                IR::Seq {
                    items: vec![IR::Num(1.0), IR::Num(2.0)],
                    input: IR::Clock(1.0).into()
                },
                IR::Seq {
                    items: vec![IR::Rest, IR::Num(3.0)],
                    input: IR::Clock(1.0).into()
                }
            ]
        );
    }

    #[test]
    fn expand_nested_seq() {
        // inner seq is driven by the phase of the enclosing seq, which is resolved at runtime
        let expr = Expr::Seq(vec![
            Expr::Num(100.0),
            Expr::Seq(vec![Expr::Num(200.0), Expr::Num(300.0)]),
        ]);
        assert_eq!(
            expand(expr).unwrap(),
            vec![IR::Seq {
                items: vec![
                    IR::Num(100.0),
                    IR::Seq {
                        items: vec![IR::Num(200.0), IR::Num(300.0)],
                        input: IR::Clock(1.0).into()
                    }
                ],
                input: IR::Clock(1.0).into()
            }]
        );
    }

    #[test]
    fn expand_clock_rate() {
        let clock = || Expr::Builtin {
            kind: Builtin::Clock,
            args: vec![],
        };
        let scale = |lhs, op, rhs| expand(Expr::Binary(Box::new(lhs), op, Box::new(rhs))).unwrap();

        assert_eq!(
            scale(clock(), BinaryOp::Div, Expr::Num(4.0)),
            vec![IR::Clock(0.25)]
        );
        assert_eq!(
            scale(clock(), BinaryOp::Mul, Expr::Num(2.0)),
            vec![IR::Binary(
                IR::Clock(1.0).into(),
                BinaryOp::Mul,
                IR::Num(2.0).into()
            )]
        );
        assert_eq!(
            scale(clock(), BinaryOp::Add, Expr::Num(2.0)),
            vec![IR::Binary(
                IR::Clock(1.0).into(),
                BinaryOp::Add,
                IR::Num(2.0).into()
            )]
        );
        assert_eq!(
            scale(clock(), BinaryOp::Div, Expr::Num(0.0)),
            vec![IR::Binary(
                IR::Clock(1.0).into(),
                BinaryOp::Div,
                IR::Num(0.0).into()
            )]
        );
    }

    #[test]
    fn expand_pitch() {
        let pitch = |n: f32| IR::Pitch(IR::Num(n).into());
        let binary = |lhs, op, rhs| expand(Expr::Binary(Box::new(lhs), op, Box::new(rhs))).unwrap();
        let transposed = |n: f32, op, interval: f32| {
            IR::Pitch(IR::Binary(IR::Num(n).into(), op, IR::Num(interval).into()).into())
        };

        // transposing
        assert_eq!(
            binary(Expr::Pitch(69.0), BinaryOp::Add, Expr::Num(12.0)),
            vec![transposed(69.0, BinaryOp::Add, 12.0)]
        );
        assert_eq!(
            binary(Expr::Num(12.0), BinaryOp::Add, Expr::Pitch(69.0)),
            vec![transposed(69.0, BinaryOp::Add, 12.0)]
        );
        assert_eq!(
            binary(Expr::Pitch(69.0), BinaryOp::Sub, Expr::Num(12.0)),
            vec![transposed(69.0, BinaryOp::Sub, 12.0)]
        );

        // intervals
        assert_eq!(
            binary(Expr::Pitch(64.0), BinaryOp::Sub, Expr::Pitch(60.0)),
            vec![IR::Binary(
                IR::Num(64.0).into(),
                BinaryOp::Sub,
                IR::Num(60.0).into()
            )]
        );

        // other operations act on frequencies
        assert_eq!(
            binary(Expr::Pitch(69.0), BinaryOp::Mul, Expr::Num(2.0)),
            vec![IR::Binary(
                pitch(69.0).into(),
                BinaryOp::Mul,
                IR::Num(2.0).into()
            )]
        );

        // seqs of pitches are transposed item by item, keeping rests
        assert_eq!(
            binary(
                Expr::Seq(vec![Expr::Pitch(60.0), Expr::Rest]),
                BinaryOp::Add,
                Expr::Num(7.0)
            ),
            vec![IR::Seq {
                items: vec![transposed(60.0, BinaryOp::Add, 7.0), IR::Rest],
                input: IR::Clock(1.0).into()
            }]
        );
    }

    #[test]
    fn expand_mix() {
        let expr = Expr::Builtin {
            kind: Builtin::Mix,
            args: vec![Expr::Stack(vec![
                Expr::Num(1.0),
                Expr::Num(2.0),
                Expr::Num(3.0),
            ])],
        };
        assert_eq!(
            expand(expr).unwrap(),
            vec![IR::Mix(vec![IR::Num(1.0), IR::Num(2.0), IR::Num(3.0)])]
        );

        // mixing nothing is silence
        let expr = Expr::Builtin {
            kind: Builtin::Mix,
            args: vec![Expr::Stack(vec![])],
        };
        assert_eq!(expand(expr).unwrap(), vec![IR::Mix(vec![])]);
    }

    #[test]
    fn expand_builtin() {
        let expr = Expr::Builtin {
            kind: Builtin::Sin,
            args: vec![Expr::Num(0.0)],
        };
        assert_eq!(
            expand(expr).unwrap(),
            vec![IR::Builtin {
                kind: Builtin::Sin,
                args: vec![IR::Num(0.0)]
            }]
        );

        let expr = Expr::Builtin {
            kind: Builtin::Sin,
            args: vec![Expr::Stack(vec![Expr::Num(100.0), Expr::Num(200.0)])],
        };
        assert_eq!(
            expand(expr).unwrap(),
            vec![
                IR::Builtin {
                    kind: Builtin::Sin,
                    args: vec![IR::Num(100.0)]
                },
                IR::Builtin {
                    kind: Builtin::Sin,
                    args: vec![IR::Num(200.0)]
                }
            ]
        );
    }

    #[test]
    fn expand_call() {
        let expr = Expr::Call {
            callee: Expr::Seq(vec![Expr::Num(0.0)]).into(),
            args: vec![Expr::Num(1.0)],
        };
        assert_eq!(
            expand(expr).unwrap(),
            vec![IR::Seq {
                items: vec![IR::Num(0.0)],
                input: IR::Num(1.0).into()
            }]
        );

        // stack of seqs as callee
        let expr = Expr::Call {
            callee: Expr::Stack(vec![
                Expr::Seq(vec![Expr::Num(0.0)]),
                Expr::Seq(vec![Expr::Num(1.0)]),
            ])
            .into(),
            args: vec![Expr::Num(0.5)],
        };
        assert_eq!(
            expand(expr).unwrap(),
            vec![
                IR::Seq {
                    items: vec![IR::Num(0.0)],
                    input: IR::Num(0.5).into()
                },
                IR::Seq {
                    items: vec![IR::Num(1.0)],
                    input: IR::Num(0.5).into()
                }
            ]
        );

        // stack as input
        let expr = Expr::Call {
            callee: Expr::Seq(vec![Expr::Num(0.0)]).into(),
            args: vec![Expr::Stack(vec![Expr::Num(0.0), Expr::Num(0.5)])],
        };
        assert_eq!(
            expand(expr).unwrap(),
            vec![
                IR::Seq {
                    items: vec![IR::Num(0.0)],
                    input: IR::Num(0.0).into()
                },
                IR::Seq {
                    items: vec![IR::Num(0.0)],
                    input: IR::Num(0.5).into()
                }
            ]
        );

        // stack callee
        let expr = Expr::Call {
            callee: Expr::Stack(vec![Expr::Num(0.0)]).into(),
            args: vec![Expr::Num(0.0)],
        };
        assert!(matches!(expand(expr), Err(CompileError::NotCallable(_))));
    }
}
