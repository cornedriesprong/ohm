use crate::nodes::{DelayNode, EnvNode, NodeKind, PluckNode, PulseNode, SeqNode};
use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use fundsp::hacker::wavech;
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
    let args: Vec<String> = std::env::args().collect();
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
            KValue::Object(obj) if obj.is_a::<NodeKind>() => match obj.cast::<NodeKind>() {
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
    // add node creation functions to Koto environment
    koto.prelude().add_fn(
        "sin",
        make_expr_node(|args| {
            use fundsp::hacker32::sine;
            NodeKind::Sine {
                freq: Box::new(args[0].clone()),
                node: Box::new(sine()),
            }
            .into()
        }),
    );
    koto.prelude().add_fn(
        "sqr",
        make_expr_node(|args| {
            use fundsp::hacker32::square;
            NodeKind::Square {
                freq: Box::new(args[0].clone()),
                node: Box::new(square()),
            }
            .into()
        }),
    );
    koto.prelude().add_fn(
        "saw",
        make_expr_node(|args| {
            use fundsp::hacker32::saw;
            NodeKind::Saw {
                freq: Box::new(args[0].clone()),
                node: Box::new(saw()),
            }
            .into()
        }),
    );
    koto.prelude().add_fn(
        "tri",
        make_expr_node(|args| {
            use fundsp::hacker32::triangle;
            NodeKind::Triangle {
                freq: Box::new(args[0].clone()),
                node: Box::new(triangle()),
            }
            .into()
        }),
    );
    koto.prelude().add_fn(
        "pulse",
        make_expr_node(|args| {
            NodeKind::Pulse {
                freq: Box::new(args[0].clone()),
                node: PulseNode::new(),
            }
            .into()
        }),
    );
    koto.prelude().add_fn(
        "noise",
        make_expr_node(|_| {
            use fundsp::hacker32::noise;
            NodeKind::Noise(Box::new(noise())).into()
        }),
    );
    koto.prelude().add_fn("env", move |ctx| {
        let args = ctx.args();
        if args.len() != 2 {
            return unexpected_args("expected 2 arguments: list, trig", args);
        }
        let segments = list_of_tuples_from_value(&args[0])?;
        let trig = node_from_kvalue(&args[1])?;

        Ok(KValue::Object(
            NodeKind::Env {
                trig: Box::new(trig),
                segments,
                node: EnvNode::new(),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("lp", move |ctx| {
        use fundsp::hacker32::lowpass;
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("expected 3 arguments: cutoff, resonance, input", args);
        }

        let cutoff = node_from_kvalue(&args[0])?;
        let resonance = node_from_kvalue(&args[1])?;
        let input = node_from_kvalue(&args[2])?;

        Ok(KValue::Object(
            NodeKind::SVF {
                input: Box::new(input),
                cutoff: Box::new(cutoff),
                resonance: Box::new(resonance),
                node: Box::new(lowpass()),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("bp", move |ctx| {
        use fundsp::hacker32::bandpass;
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("expected 3 arguments: cutoff, resonance, input", args);
        }

        let cutoff = node_from_kvalue(&args[0])?;
        let resonance = node_from_kvalue(&args[1])?;
        let input = node_from_kvalue(&args[2])?;

        Ok(KValue::Object(
            NodeKind::SVF {
                input: Box::new(input),
                cutoff: Box::new(cutoff),
                resonance: Box::new(resonance),
                node: Box::new(bandpass()),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("hp", move |ctx| {
        use fundsp::hacker32::highpass;
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("expected 3 arguments: cutoff, resonance, input", args);
        }

        let cutoff = node_from_kvalue(&args[0])?;
        let resonance = node_from_kvalue(&args[1])?;
        let input = node_from_kvalue(&args[2])?;

        Ok(KValue::Object(
            NodeKind::SVF {
                input: Box::new(input),
                cutoff: Box::new(cutoff),
                resonance: Box::new(resonance),
                node: Box::new(highpass()),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("seq", move |ctx| {
        let args = ctx.args();
        if args.len() != 2 {
            return unexpected_args("expected 2 arguments: list, trig", args);
        }

        let values = list_from_value(&args[0])?;
        let trig = node_from_kvalue(&args[1])?;

        Ok(KValue::Object(
            NodeKind::Seq {
                trig: Box::new(trig),
                values,
                node: SeqNode::new(),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("pan", move |ctx| {
        use fundsp::hacker32::panner;
        let args = ctx.args();
        if args.len() != 2 {
            return unexpected_args("expected 2 arguments: pan, input", args);
        }

        let input = node_from_kvalue(&args[0])?;
        let value = node_from_kvalue(&args[1])?;

        Ok(KValue::Object(
            NodeKind::Pan {
                input: Box::new(input),
                value: Box::new(value),
                node: Box::new(panner()),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("pluck", move |ctx| {
        let args = ctx.args();
        if args.len() != 4 {
            return unexpected_args("expected 4 arguments: frequency, tone, damping, trig", args);
        }

        let freq = node_from_kvalue(&args[0])?;
        let tone = node_from_kvalue(&args[1])?;
        let damping = node_from_kvalue(&args[2])?;
        let trig = node_from_kvalue(&args[3])?;

        Ok(KValue::Object(
            NodeKind::Pluck {
                freq: Box::new(freq),
                tone: Box::new(tone),
                damping: Box::new(damping),
                trig: Box::new(trig),
                node: PluckNode::new(),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("reverb", move |ctx| {
        use fundsp::hacker32::{lowpole_hz, reverb2_stereo};
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;

        Ok(KValue::Object(
            NodeKind::Reverb {
                input: Box::new(input),
                node: Box::new(reverb2_stereo(10.0, 2.0, 0.9, 1.0, lowpole_hz(18000.0))),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("delay", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;

        Ok(KValue::Object(
            NodeKind::Delay {
                input: Box::new(input),
                node: DelayNode::new(),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("moog", move |ctx| {
        use fundsp::hacker32::moog;
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("expected 3 arguments: cutoff, resonance, input", args);
        }

        let cutoff = node_from_kvalue(&args[0])?;
        let resonance = node_from_kvalue(&args[1])?;
        let input = node_from_kvalue(&args[2])?;

        Ok(KValue::Object(
            NodeKind::Moog {
                cutoff: Box::new(cutoff),
                resonance: Box::new(resonance),
                input: Box::new(input),
                node: Box::new(moog()),
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("wav", move |ctx| {
        use fundsp::hacker32::Wave;
        let args = ctx.args();
        if args.len() != 1 {
            return unexpected_args("expected 1 argument: filename", args);
        }

        let filename = format!("samples/{}", str_from_kvalue(&args[0])?);
        let wave = Wave::load(&filename).expect("Could not load wave.");
        let file = Arc::new(wave);

        Ok(KValue::Object(
            NodeKind::Sampler {
                node: Box::new(wavech(&file, 0, Some(0))),
            }
            .into(),
        ))
    });
}

fn make_expr_node<F>(node_constructor: F) -> impl KotoFunction
where
    F: Fn(Vec<NodeKind>) -> NodeKind + 'static,
{
    move |ctx| {
        let args: Result<Vec<NodeKind>, _> = ctx.args().iter().map(node_from_kvalue).collect();
        Ok(KValue::Object(node_constructor(args?).into()))
    }
}

fn node_from_kvalue(value: &KValue) -> Result<NodeKind, koto::runtime::Error> {
    match value {
        KValue::Number(n) => Ok(NodeKind::Constant(n.into())),
        KValue::Object(obj) if obj.is_a::<NodeKind>() => Ok(obj.cast::<NodeKind>()?.to_owned()),
        unexpected => unexpected_type("number, expr, or list", unexpected)?,
    }
}

fn str_from_kvalue(value: &KValue) -> Result<String, koto::runtime::Error> {
    match value {
        KValue::Str(s) => Ok(s.to_string()),
        unexpected => unexpected_type("string", unexpected)?,
    }
}

fn list_from_value(value: &KValue) -> Result<Vec<NodeKind>, koto::runtime::Error> {
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

fn list_of_tuples_from_value(
    value: &KValue,
) -> Result<Vec<(NodeKind, NodeKind)>, koto::runtime::Error> {
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
