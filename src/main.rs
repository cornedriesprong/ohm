use crate::nodes::{
    BufReaderNode, BufTapNode, BufWriterNode, EnvNode, FunDSPNode, LFONode, NodeKind, PulseNode,
    SeqNode,
};
use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use koto::prelude::*;
use nodes::DelayNode;
use notify::{event::ModifyKind, EventKind, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{
    fs,
    time::{Duration, Instant},
};

mod nodes;
mod utils;

mod op;
use crate::op::Op;

mod audio_graph;
use audio_graph::*;

fn main() -> anyhow::Result<()> {
    // let args: Vec<String> = std::env::args().collect();
    // if args.len() != 2 {
    //     eprintln!("Usage: {} <filename>", args[0]);
    //     std::process::exit(1);
    // }

    // let filename = &args[1];
    let filename = "patch.koto";

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");
    let config = device.default_output_config()?;

    run::<f32>(&device, &config.into(), filename)
}

pub fn run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    filename: &str,
) -> Result<(), anyhow::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let container = Container::new();
    let container: Arc<Mutex<Container>> = Arc::new(Mutex::new(container));
    let container_clone1 = Arc::clone(&container);
    let container_clone2 = Arc::clone(&container);

    let mut koto = Koto::default();
    create_env(&koto, container_clone1, config.sample_rate.0);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for frame in data.chunks_mut(2) {
                let out = container_clone2.lock().unwrap().tick();
                frame[0] = out[0];
                frame[1] = out[1];
            }
        },
        err_fn,
        None,
    )?;

    stream.play()?;

    let mut update_audio_graph = |path: &Path| -> Result<(), anyhow::Error> {
        let src = fs::read_to_string(path)?;
        match koto.compile_and_run(CompileArgs {
            script: &src,
            script_path: Some(KString::from(path.to_str().unwrap())),
            compiler_settings: CompilerSettings {
                export_top_level_ids: true,
                ..Default::default()
            },
        })? {
            KValue::Object(obj) if obj.is_a::<Op>() => match obj.cast::<Op>() {
                Ok(expr) => {
                    let graph = parse_to_graph(expr.to_owned());
                    let mut guard = container.lock().unwrap();
                    guard.update_graph(graph);
                    Ok(())
                }
                Err(e) => bail!("Failed to cast to Expr: {}", e),
            },
            KValue::Str(str) => {
                println!("{}", str);

                Ok(())
            }
            other => bail!("Expected a Map, found '{}'", other.type_as_string(),),
        }
    };

    update_audio_graph(Path::new(filename))?;

    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(tx)?;

    let path = Path::new(filename);
    watcher.watch(path, RecursiveMode::NonRecursive)?;

    let mut last_update = Instant::now();
    let debounce_duration = Duration::from_millis(100); // Adjust as needed

    for res in rx {
        match res {
            Ok(event) => match event.kind {
                EventKind::Modify(ModifyKind::Data(_)) => {
                    let now = Instant::now();
                    if now.duration_since(last_update) >= debounce_duration {
                        if let Err(e) = update_audio_graph(path) {
                            eprintln!("Error updating audio graph: {}", e);
                        }
                        last_update = now;
                    }
                }
                EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(_)) => {
                    // re-watch the file
                    watcher.unwatch(path).ok();
                    watcher.watch(path, RecursiveMode::NonRecursive).ok();

                    let now = Instant::now();
                    if now.duration_since(last_update) >= debounce_duration {
                        if let Err(e) = update_audio_graph(path) {
                            eprintln!("Error updating audio graph: {}", e);
                        }
                        last_update = now;
                    }
                }
                _ => {}
            },
            Err(e) => eprintln!("Watch error: {}", e),
        }
    }

    Ok(())
}

fn create_env(koto: &Koto, container: Arc<Mutex<Container>>, sample_rate: u32) {
    use fundsp::hacker32::*;

    koto.prelude().insert("bpm", 120.0);
    koto.prelude().insert("sr", sample_rate);

    add_osc(koto, "sin".to_string(), || Box::new(sine()));
    add_osc(koto, "sqr".to_string(), || Box::new(square()));
    add_osc(koto, "saw".to_string(), || Box::new(saw()));
    add_osc(koto, "tri".to_string(), || Box::new(triangle()));
    add_osc(koto, "ramp".to_string(), || Box::new(ramp()));

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
    koto.prelude().add_fn("svf", move |ctx| {
        use fundsp::hacker32::{allpass, bandpass, highpass, lowpass, notch, peak};

        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        let filter_type = str_from_kvalue(&args[1])?;
        let hz = node_from_kvalue(&args[2])?;
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
                inputs: vec![input, hz, resonance],
                node: Box::new(FunDSPNode::mono(audio_unit)),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("moog", move |ctx| {
        use fundsp::hacker32::moog;

        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        let cutoff = node_from_kvalue(args.get(1).unwrap_or(&KValue::Number(1000.0.into())))?;
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
        "pulse",
        make_expr_node(move |args| Op::Node {
            kind: NodeKind::Pulse,
            inputs: vec![args[0].clone()],
            node: Box::new(PulseNode::new(sample_rate)),
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
    koto.prelude().add_fn("ftop", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Ftop,
                inputs: vec![input],
                node: Box::new(FunDSPNode::mono(Box::new(sink()))),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("ptof", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Ptof,
                inputs: vec![input],
                node: Box::new(FunDSPNode::mono(Box::new(sink()))),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("env", move |ctx| {
        let args = ctx.args();
        let ramp = node_from_kvalue(&args[0])?;
        let segments = list_of_tuples_from_value(&args[1])?;

        let inputs = segments
            .iter()
            .flat_map(|(value, duration)| vec![value.clone(), duration.clone()])
            .chain(std::iter::once(ramp))
            .collect::<Vec<_>>();

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Env,
                inputs,
                node: Box::new(EnvNode::new()),
            }
            .into(),
        ))
    });
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
        let pan = node_from_kvalue(&args[1])?;

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

fn list_of_tuples_from_value(value: &KValue) -> Result<Vec<(Op, Op)>, koto::runtime::Error> {
    match value {
        KValue::List(list) => list
            .data()
            .iter()
            .map(|item| match item {
                KValue::Tuple(t) => {
                    let target = node_from_kvalue(&t[0])?;
                    let duration = node_from_kvalue(&t[1])?;
                    Ok((target, duration))
                }
                unexpected => unexpected_type("tuple", unexpected),
            })
            .collect(),
        unexpected => unexpected_type("list", unexpected),
    }
}
