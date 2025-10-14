use crate::audio_graph::Container;
use crate::nodes::{
    self, BufReaderNode, BufTapNode, BufWriterNode, DelayNode, FunDSPNode, LFONode, Node, NodeKind,
    SampleAndHoldNode, SeqNode,
};
use crate::op::Op;
use koto::runtime::KList;
use koto::{
    runtime::{unexpected_type, KValue},
    Koto,
};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
enum ParamType {
    Node { default: f32 },
}

struct NodeDescriptor {
    name: &'static str,
    kind: NodeKind,
    params: &'static [ParamType],
    node: Box<dyn Node>,
}

pub fn create_env(koto: &Koto, container: Arc<Mutex<Container>>, sample_rate: u32) {
    use fundsp::hacker32::*;

    koto.prelude().insert("bpm", 120.0);
    koto.prelude().insert("sr", sample_rate);

    let nodes = vec![
        NodeDescriptor {
            name: "ramp",
            kind: NodeKind::Ramp,
            params: &[ParamType::Node { default: 100.0 }],
            node: Box::new(FunDSPNode::mono(Box::new(ramp()))),
        },
        NodeDescriptor {
            name: "sin",
            kind: NodeKind::Sin,
            params: &[ParamType::Node { default: 100.0 }],
            node: Box::new(FunDSPNode::mono(Box::new(sine()))),
        },
        NodeDescriptor {
            name: "sqr",
            kind: NodeKind::Sqr,
            params: &[ParamType::Node { default: 100.0 }],
            node: Box::new(FunDSPNode::mono(Box::new(square()))),
        },
        NodeDescriptor {
            name: "saw",
            kind: NodeKind::Saw,
            params: &[ParamType::Node { default: 100.0 }],
            node: Box::new(FunDSPNode::mono(Box::new(saw()))),
        },
        NodeDescriptor {
            name: "tri",
            kind: NodeKind::Tri,
            params: &[ParamType::Node { default: 100.0 }],
            node: Box::new(FunDSPNode::mono(Box::new(triangle()))),
        },
        NodeDescriptor {
            name: "clip",
            kind: NodeKind::Clip,
            params: &[ParamType::Node { default: 100.0 }],
            node: Box::new(FunDSPNode::mono(Box::new(clip()))),
        },
        NodeDescriptor {
            name: "lfo",
            kind: NodeKind::Lfo,
            params: &[ParamType::Node { default: 100.0 }],
            node: Box::new(LFONode::new(sample_rate)),
        },
        NodeDescriptor {
            name: "sh",
            kind: NodeKind::SampleAndHold,
            params: &[
                ParamType::Node { default: 0.0 },
                ParamType::Node { default: 0.0 },
            ],
            node: Box::new(SampleAndHoldNode::new()),
        },
        NodeDescriptor {
            name: "onepole",
            kind: NodeKind::Onepole,
            params: &[
                ParamType::Node { default: 0.0 },
                ParamType::Node { default: 20.0 },
            ],
            node: Box::new(FunDSPNode::mono(Box::new(lowpole()))),
        },
        NodeDescriptor {
            name: "log",
            kind: NodeKind::Log,
            params: &[ParamType::Node { default: 0.0 }],
            node: Box::new(FunDSPNode::mono(Box::new(sink()))),
        },
        NodeDescriptor {
            name: "noise",
            kind: NodeKind::Noise,
            params: &[],
            node: Box::new(FunDSPNode::mono(Box::new(noise()))),
        },
        NodeDescriptor {
            name: "pan",
            kind: NodeKind::Pan,
            params: &[
                ParamType::Node { default: 0.0 },
                ParamType::Node { default: 0.5 },
            ],
            node: Box::new(FunDSPNode::stereo(Box::new(panner()))),
        },
        NodeDescriptor {
            name: "reverb",
            kind: NodeKind::Reverb,
            params: &[ParamType::Node { default: 0.0 }],
            node: Box::new(FunDSPNode::stereo(Box::new(reverb2_stereo(
                10.0,
                2.0,
                0.9,
                1.0,
                lowpole_hz(18000.0),
            )))),
        },
        NodeDescriptor {
            name: "delay",
            kind: NodeKind::Delay,
            params: &[
                ParamType::Node { default: 0.0 },
                ParamType::Node { default: 1.0 },
            ],
            node: Box::new(DelayNode::new()),
        },
        NodeDescriptor {
            name: "moog",
            kind: NodeKind::Moog,
            params: &[
                ParamType::Node { default: 0.0 },
                ParamType::Node { default: 500.0 },
                ParamType::Node { default: 0.0 },
            ],
            node: Box::new(FunDSPNode::mono(Box::new(moog()))),
        },
    ];

    for node in nodes {
        let params = node.params;
        let kind = node.kind.clone();
        let node_impl = node.node.clone();

        koto.prelude().add_fn(node.name, move |ctx| {
            let args = ctx.args();
            let mut inputs = Vec::new();

            for (i, param_type) in params.iter().enumerate() {
                match param_type {
                    ParamType::Node { default } => {
                        let default = KValue::Number((*default).into());
                        let input = node_from_kvalue(args.get(i).unwrap_or(&default))?;
                        inputs.push(input);
                    }
                };
            }

            Ok(KValue::Object(
                Op::Node {
                    kind: kind.clone(),
                    inputs,
                    node: node_impl.clone(),
                }
                .into(),
            ))
        });
    }

    koto.prelude().add_fn("svf", move |ctx| {
        use fundsp::hacker32::{allpass, bandpass, highpass, lowpass, notch, peak};

        let args = ctx.args();
        let input = node_from_kvalue(args.get(0).unwrap_or(&KValue::Number(0.0.into())))?;
        let filter_type = str_from_kvalue(args.get(1).unwrap_or(&KValue::Str("lp".into())))?;
        let cutoff = node_from_kvalue(args.get(2).unwrap_or(&KValue::Number(500.0.into())))?;
        let resonance = node_from_kvalue(args.get(3).unwrap_or(&KValue::Number(0.707.into())))?;

        let audio_unit: Box<dyn AudioUnit + Send> = match filter_type.as_str() {
            "lp" | "lowpass" => Box::new(lowpass()),
            "bp" | "bandpass" => Box::new(bandpass()),
            "hp" | "highpass" => Box::new(highpass()),
            "notch" => Box::new(notch()),
            "peak" => Box::new(peak()),
            "ap" | "allpass" => Box::new(allpass()),
            _ => return Err("Missing filter type".into()),
        };

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Svf,
                inputs: vec![input, cutoff, resonance],
                node: Box::new(FunDSPNode::mono(audio_unit)),
            }
            .into(),
        ))
    });

    koto.prelude().add_fn("seq", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(args.get(0).unwrap_or(&KValue::Number(0.0.into())))?;
        let values = list_from_value(
            args.get(1)
                .unwrap_or(&KValue::List(KList::with_capacity(0))),
        )?;

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Seq,
                inputs: values
                    .iter()
                    .map(|value| value.clone())
                    .chain(std::iter::once(input.clone()))
                    .collect(),
                node: Box::new(SeqNode::new()),
            }
            .into(),
        ))
    });

    // Special cases: Buffer operations (need container and special buffer ID handling)
    let buf_container = Arc::clone(&container);
    koto.prelude().add_fn("buf", move |ctx| {
        let args = ctx.args();
        let default_length = KValue::Number(sample_rate.into());
        let length = num_from_kvalue(args.get(0).unwrap_or(&default_length))?;

        let mut container = buf_container
            .lock()
            .map_err(|e| format!("Failed to lock buffer container: {e}"))?;

        let id = container.add_buffer(length as usize);

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::BufferRef { id },
                inputs: vec![],
                node: Box::new(FunDSPNode::mono(Box::new(sink()))),
            }
            .into(),
        ))
    });
    let file_container = Arc::clone(&container);
    koto.prelude().add_fn("file", move |ctx| {
        let args = ctx.args();
        let name = str_from_kvalue(args.get(0).unwrap_or(&KValue::Str("".into())))?;

        let mut container = file_container
            .lock()
            .map_err(|e| format!("Failed to lock buffer container: {e}"))?;

        let filename = if name.ends_with(".wav") {
            format!("samples/{}", name)
        } else {
            format!("samples/{}.wav", name)
        };

        let wave = Wave::load(filename).map_err(|e| format!("Failed to load '{name}': {e}"))?;

        let frames: Vec<nodes::Frame> = match wave.channels() {
            1 => (0..wave.len())
                .map(|i| {
                    let sample = wave.at(0, i);
                    [sample, sample]
                })
                .collect(),
            _ => (0..wave.len())
                .map(|i| [wave.at(0, i), wave.at(1, i)])
                .collect(),
        };

        let id = container.load_frames_to_buffer(frames);

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::BufferRef { id },
                inputs: vec![],
                node: Box::new(FunDSPNode::mono(Box::new(sink()))),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("rec", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        let buf_ref = node_from_kvalue(&args[1])?;

        return match buf_ref {
            Op::Node { kind, .. } => match kind {
                NodeKind::BufferRef { id } => Ok(KValue::Object(
                    Op::Node {
                        kind: NodeKind::BufferWriter { id },
                        inputs: vec![input],
                        node: Box::new(BufWriterNode::new()),
                    }
                    .into(),
                )),
                _ => Err("Expected a buffer".into()),
            },
            _ => Err("Expected a buffer".into()),
        };
    });
    koto.prelude().add_fn("play", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(args.get(0).unwrap_or(&KValue::Number(0.0.into())))?;
        let buf_ref = node_from_kvalue(&args[1])?;

        return match buf_ref {
            Op::Node { kind, .. } => match kind {
                NodeKind::BufferRef { id } => Ok(KValue::Object(
                    Op::Node {
                        kind: NodeKind::BufferReader { id },
                        inputs: vec![input],
                        node: Box::new(BufReaderNode::new()),
                    }
                    .into(),
                )),
                _ => Err("Expected a buffer".into()),
            },
            _ => Err("Expected a buffer".into()),
        };
    });
    koto.prelude().add_fn("tap", move |ctx| {
        let args = ctx.args();
        let buf_ref = node_from_kvalue(&args[0])?;
        let offset = node_from_kvalue(args.get(1).unwrap_or(&KValue::Number(0.0.into())))?;

        return match buf_ref {
            Op::Node { kind, .. } => match kind {
                NodeKind::BufferRef { id } => Ok(KValue::Object(
                    Op::Node {
                        kind: NodeKind::BufferTap { id },
                        inputs: vec![offset],
                        node: Box::new(BufTapNode::new()),
                    }
                    .into(),
                )),
                _ => Err("Expected a buffer".into()),
            },
            _ => Err("Expected a buffer".into()),
        };
    });
    // greater than
    koto.prelude().add_fn("gt", move |ctx| {
        let args = ctx.args();
        let lhs = node_from_kvalue(args.get(0).unwrap_or(&KValue::Number(0.0.into())))?;
        let rhs = node_from_kvalue(args.get(1).unwrap_or(&KValue::Number(0.0.into())))?;
        Ok(KValue::Object(
            Op::Greater(Box::new(lhs), Box::new(rhs)).into(),
        ))
    });
    // less than
    koto.prelude().add_fn("lt", move |ctx| {
        let args = ctx.args();
        let lhs = node_from_kvalue(args.get(0).unwrap_or(&KValue::Number(0.0.into())))?;
        let rhs = node_from_kvalue(args.get(1).unwrap_or(&KValue::Number(0.0.into())))?;
        Ok(KValue::Object(
            Op::Less(Box::new(lhs), Box::new(rhs)).into(),
        ))
    });
    // equal
    koto.prelude().add_fn("eq", move |ctx| {
        let args = ctx.args();
        let lhs = node_from_kvalue(args.get(0).unwrap_or(&KValue::Number(0.0.into())))?;
        let rhs = node_from_kvalue(args.get(1).unwrap_or(&KValue::Number(0.0.into())))?;
        Ok(KValue::Object(
            Op::Equal(Box::new(lhs), Box::new(rhs)).into(),
        ))
    });
}

fn node_from_kvalue(value: &KValue) -> Result<Op, koto::runtime::Error> {
    use fundsp::hacker32::sink;
    match value {
        KValue::Number(n) => Ok(Op::Constant(n.into())),
        KValue::Object(obj) if obj.is_a::<Op>() => Ok(obj.cast::<Op>()?.to_owned()),
        KValue::Str(_) => Ok(Op::Node {
            kind: NodeKind::Log,
            inputs: vec![],
            node: Box::new(FunDSPNode::mono(Box::new(sink()))),
        }),
        unexpected => unexpected_type("number, expr, or list", unexpected)?,
    }
}

fn num_from_kvalue(value: &KValue) -> Result<f32, koto::runtime::Error> {
    match value {
        KValue::Number(n) => Ok(n.into()),
        unexpected => unexpected_type("number", unexpected)?,
    }
}

fn str_from_kvalue(value: &KValue) -> Result<String, koto::runtime::Error> {
    match value {
        KValue::Str(s) => Ok(s.to_string()),
        unexpected => unexpected_type("string", unexpected)?,
    }
}

fn list_from_value(value: &KValue) -> Result<Vec<Op>, koto::runtime::Error> {
    match value {
        KValue::List(list) => Ok(list
            .data()
            .iter()
            .map(node_from_kvalue)
            .collect::<Result<Vec<_>, _>>()?),
        KValue::Tuple(t) => Ok(t
            .iter()
            .map(node_from_kvalue)
            .collect::<Result<Vec<_>, _>>()?),
        unexpected => unexpected_type("list", unexpected),
    }
}
