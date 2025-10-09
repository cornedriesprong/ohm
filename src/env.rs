use crate::audio_graph::Container;
use crate::nodes::{
    self, BufReaderNode, BufTapNode, BufWriterNode, DelayNode, FunDSPNode, LFONode, NodeKind,
    SampleAndHoldNode, SeqNode,
};
use crate::op::Op;
use koto::{
    runtime::{unexpected_type, KValue, KotoFunction},
    Koto,
};
use std::sync::{Arc, Mutex};

pub fn create_env(koto: &Koto, container: Arc<Mutex<Container>>, sample_rate: u32) {
    use fundsp::hacker32::*;

    koto.prelude().insert("bpm", 120.0);
    koto.prelude().insert("sr", sample_rate);

    add_osc(koto, "sin".to_string(), || Box::new(sine()));
    add_osc(koto, "sqr".to_string(), || Box::new(square()));
    add_osc(koto, "saw".to_string(), || Box::new(saw()));
    add_osc(koto, "tri".to_string(), || Box::new(triangle()));

    koto.prelude().add_fn("ramp", move |ctx| {
        let freq = node_from_kvalue(ctx.args().get(0).unwrap_or(&KValue::Number(1.0.into())))?;
        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Ramp,
                inputs: vec![freq],
                node: Box::new(FunDSPNode::mono(Box::new(ramp()))),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("clip", move |ctx| {
        let freq = node_from_kvalue(ctx.args().get(0).unwrap_or(&KValue::Number(1.0.into())))?;
        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Ramp,
                inputs: vec![freq],
                node: Box::new(FunDSPNode::mono(Box::new(clip()))),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("lfo", move |ctx| {
        let freq = node_from_kvalue(ctx.args().get(0).unwrap_or(&KValue::Number(1000.0.into())))?;
        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Lfo,
                inputs: vec![freq],
                node: Box::new(LFONode::new(sample_rate)),
            }
            .into(),
        ))
    });
    // sample and hold
    koto.prelude()
        .add_fn("sh", move |ctx| -> Result<KValue, _> {
            let args = ctx.args();
            let input = node_from_kvalue(&args[0])?;
            let trig = node_from_kvalue(&args[1])?;
            Ok(KValue::Object(koto::prelude::KObject::from(Op::Node {
                kind: NodeKind::SampleAndHold,
                inputs: vec![input, trig],
                node: Box::new(SampleAndHoldNode::new()),
            })))
        });
    koto.prelude().add_fn("onepole", move |ctx| {
        use fundsp::hacker32::lowpole;

        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        let cutoff = node_from_kvalue(&args[1])?;

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Onepole,
                inputs: vec![input, cutoff],
                node: Box::new(FunDSPNode::mono(Box::new(lowpole()))),
            }
            .into(),
        ))
    });
    // state variable filter
    koto.prelude().add_fn("svf", move |ctx| {
        use fundsp::hacker32::{allpass, bandpass, highpass, lowpass, notch, peak};

        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        let filter_type = str_from_kvalue(&args[1])?;
        let cutoff = node_from_kvalue(&args[2])?;
        let resonance = node_from_kvalue(&args[3])?;

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
    koto.prelude().add_fn("moog", move |ctx| {
        use fundsp::hacker32::moog;

        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        let cutoff = node_from_kvalue(args.get(1).unwrap_or(&KValue::Number(500.0.into())))?;
        let q = node_from_kvalue(args.get(2).unwrap_or(&KValue::Number(0.0.into())))?;

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Moog,
                inputs: vec![input, cutoff, q],
                node: Box::new(FunDSPNode::mono(Box::new(moog()))),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn(
        "log",
        make_expr_node(|args| Op::Node {
            kind: NodeKind::Log,
            inputs: vec![args[0].clone()],
            node: Box::new(FunDSPNode::mono(Box::new(sink()))),
        }),
    );
    koto.prelude().add_fn(
        "noise",
        make_expr_node(|_| {
            Op::Node {
                kind: NodeKind::Noise,
                inputs: vec![],
                node: Box::new(FunDSPNode::mono(Box::new(noise()))),
            }
            .into()
        }),
    );
    koto.prelude().add_fn("seq", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        let values = list_from_value(&args[1])?;

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
    koto.prelude().add_fn("pan", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        let pan = node_from_kvalue(args.get(1).unwrap_or(&KValue::Number(0.5.into())))?;

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Pan,
                inputs: vec![input, pan],
                node: Box::new(FunDSPNode::stereo(Box::new(panner()))),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("reverb", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Reverb,
                inputs: vec![input],
                node: Box::new(FunDSPNode::stereo(Box::new(reverb2_stereo(
                    10.0,
                    2.0,
                    0.9,
                    1.0,
                    lowpole_hz(18000.0),
                )))),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("delay", move |ctx| {
        let args = ctx.args();

        let input = node_from_kvalue(&args[0])?;
        let delay = node_from_kvalue(args.get(1).unwrap_or(&KValue::Number(1.into())))?;

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Delay,
                inputs: vec![input, delay],
                node: Box::new(DelayNode::new()),
            }
            .into(),
        ))
    });
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
        let name = str_from_kvalue(&args[0])?;

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
        let buf = node_from_kvalue(&args[1])?;
        return match buf {
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
        let input = node_from_kvalue(&args[0])?;
        let buf = node_from_kvalue(&args[1])?;
        return match buf {
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
        let buf = node_from_kvalue(&args[0])?;
        let offset = node_from_kvalue(&args[1])?;
        return match buf {
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
        let lhs = node_from_kvalue(&args[0])?;
        let rhs = node_from_kvalue(&args[1])?;
        Ok(KValue::Object(
            Op::Greater(Box::new(lhs), Box::new(rhs)).into(),
        ))
    });
    // less than
    koto.prelude().add_fn("lt", move |ctx| {
        let args = ctx.args();
        let lhs = node_from_kvalue(&args[0])?;
        let rhs = node_from_kvalue(&args[1])?;
        Ok(KValue::Object(
            Op::Less(Box::new(lhs), Box::new(rhs)).into(),
        ))
    });
    // equal
    koto.prelude().add_fn("eq", move |ctx| {
        let args = ctx.args();
        let lhs = node_from_kvalue(&args[0])?;
        let rhs = node_from_kvalue(&args[1])?;
        Ok(KValue::Object(
            Op::Equal(Box::new(lhs), Box::new(rhs)).into(),
        ))
    });
}

fn make_expr_node<F>(node_constructor: F) -> impl KotoFunction
where
    F: Fn(Vec<Op>) -> Op + 'static,
{
    move |ctx| {
        let args: Result<Vec<Op>, _> = ctx.args().iter().map(node_from_kvalue).collect();
        Ok(KValue::Object(node_constructor(args?).into()))
    }
}

fn add_osc<F>(koto: &Koto, name: String, osc_fn: F)
where
    F: Fn() -> Box<dyn fundsp::hacker32::AudioUnit> + 'static,
{
    koto.prelude().add_fn(
        name.clone().as_str(),
        make_expr_node(move |args| Op::Node {
            kind: match name.as_str() {
                "sin" => NodeKind::Sin,
                "sqr" => NodeKind::Sqr,
                "saw" => NodeKind::Saw,
                "tri" => NodeKind::Tri,
                "ramp" => NodeKind::Ramp,
                _ => panic!("Unknown oscillator type: {}", name),
            },
            inputs: vec![args.get(0).cloned().unwrap_or(Op::Constant(100.0))],
            node: Box::new(FunDSPNode::mono(osc_fn())),
        }),
    );
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
