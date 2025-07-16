use crate::nodes::{DelayNode, EnvNode, FunDSPNode, NodeKind, Op, PluckNode, PulseNode, SeqNode};
use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use koto::prelude::*;
use notify::{event::ModifyKind, EventKind, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{
    fs,
    time::{Duration, Instant},
};

mod consts;
mod dsp;
mod nodes;
mod utils;

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

    let mut koto = Koto::default();
    create_env(&koto);

    let graph: Arc<Mutex<Option<AudioGraph>>> = Arc::new(Mutex::new(None));
    let graph_clone = Arc::clone(&graph);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            if let Some(graph) = graph_clone.lock().unwrap().as_mut() {
                for frame in data.chunks_mut(2) {
                    let out = graph.tick();
                    frame[0] = out[0];
                    frame[1] = out[1];
                }
            }
        },
        err_fn,
        None,
    )?;

    stream.play()?;

    let mut update_audio_graph = |path: &Path| -> Result<(), anyhow::Error> {
        let src = fs::read_to_string(path)?;
        match koto.compile_and_run(&src)? {
            KValue::Object(obj) if obj.is_a::<Op>() => match obj.cast::<Op>() {
                Ok(expr) => {
                    let new_graph = parse_to_audio_graph(expr.to_owned());
                    let mut guard = graph.lock().unwrap();

                    match guard.as_mut() {
                        Some(existing_graph) => {
                            // Apply diff to preserve state
                            existing_graph.apply_diff(new_graph);
                        }
                        None => {
                            // First time - just set the graph
                            *guard = Some(new_graph);
                        }
                    }

                    Ok(())
                }
                Err(e) => bail!("Failed to cast to Expr: {}", e),
            },
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
                    // re watch the file
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

fn create_env(koto: &Koto) {
    use fundsp::hacker32::*;

    add_osc(koto, "sin".to_string(), || Box::new(sine()));
    add_osc(koto, "sqr".to_string(), || Box::new(square()));
    add_osc(koto, "saw".to_string(), || Box::new(saw()));
    add_osc(koto, "tri".to_string(), || Box::new(triangle()));
    add_osc(koto, "ramp".to_string(), || Box::new(ramp()));

    add_filter(koto, "lp".to_string(), || Box::new(lowpass()));
    add_filter(koto, "bp".to_string(), || Box::new(bandpass()));
    add_filter(koto, "hp".to_string(), || Box::new(highpass()));

    koto.prelude().add_fn(
        "print",
        make_expr_node(|args| Op::Node {
            kind: NodeKind::Print,
            inputs: vec![args[0].clone()],
            node: Box::new(FunDSPNode::mono(Box::new(sink()))),
        }),
    );
    koto.prelude().add_fn(
        "pulse",
        make_expr_node(|args| Op::Node {
            kind: NodeKind::Pulse,
            inputs: vec![args[0].clone()],
            node: Box::new(PulseNode::new()),
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
    koto.prelude().add_fn("env", move |ctx| {
        let args = ctx.args();
        let trig = node_from_kvalue(&args[0])?;
        let segments = list_of_tuples_from_value(&args[1])?;

        let inputs = segments
            .iter()
            .flat_map(|(value, duration)| vec![value.clone(), duration.clone()])
            .chain(std::iter::once(trig))
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
    koto.prelude().add_fn("pluck", move |ctx| {
        let args = ctx.args();
        let trig = node_from_kvalue(&args[0])?;
        let freq = node_from_kvalue(&args[1])?;
        let tone = node_from_kvalue(&args[2])?;
        let damping = node_from_kvalue(&args[3])?;

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Pluck,
                inputs: vec![freq, tone, damping, trig],
                node: Box::new(PluckNode::new()),
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

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Delay,
                inputs: vec![input],
                node: Box::new(DelayNode::new()),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("moog", move |ctx| {
        let args = ctx.args();

        let input = node_from_kvalue(&args[0])?;
        let cutoff = node_from_kvalue(&args[1])?;
        let resonance = node_from_kvalue(args.get(2).unwrap_or(&KValue::Number(0.into())))?;

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Moog,
                inputs: vec![input, cutoff, resonance],
                node: Box::new(FunDSPNode::mono(Box::new(moog()))),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("wav", move |ctx| {
        let args = ctx.args();
        let filename = format!("samples/{}", str_from_kvalue(&args[0])?);
        let wave = Wave::load(&filename).expect("Could not load wave.");
        let file = Arc::new(wave);

        Ok(KValue::Object(
            Op::Node {
                kind: NodeKind::Wav,
                inputs: vec![],
                node: Box::new(FunDSPNode::mono(Box::new(wavech(&file, 0, Some(0))))),
            }
            .into(),
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
            inputs: vec![args[0].clone()],
            node: Box::new(FunDSPNode::mono(osc_fn())),
        }),
    );
}

fn add_filter<F>(koto: &Koto, name: String, filter_fn: F)
where
    F: Fn() -> Box<dyn fundsp::hacker32::AudioUnit> + 'static,
{
    koto.prelude().add_fn(name.clone().as_str(), move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;
        let cutoff = node_from_kvalue(args.get(1).unwrap_or(&KValue::Number(500.into())))?;
        let resonance = node_from_kvalue(args.get(2).unwrap_or(&KValue::Number(0.717.into())))?;

        Ok(KValue::Object(
            Op::Node {
                kind: match name.as_str() {
                    "lp" => NodeKind::Lp,
                    "bp" => NodeKind::Bp,
                    "hp" => NodeKind::Hp,
                    _ => panic!("Unknown filter type: {}", name),
                },
                inputs: vec![input, cutoff, resonance],
                node: Box::new(FunDSPNode::mono(filter_fn())),
            }
            .into(),
        ))
    });
}

fn node_from_kvalue(value: &KValue) -> Result<Op, koto::runtime::Error> {
    match value {
        KValue::Number(n) => Ok(Op::Constant(n.into())),
        KValue::Object(obj) if obj.is_a::<Op>() => Ok(obj.cast::<Op>()?.to_owned()),
        unexpected => unexpected_type("number, expr, or list", unexpected)?,
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
