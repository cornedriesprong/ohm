use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use koto::prelude::*;
use nodes::*;
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
    koto.prelude()
        .add_fn("sin", make_expr_node(|args| sine(args[0].clone())));
    koto.prelude()
        .add_fn("sqr", make_expr_node(|args| square(args[0].clone())));
    koto.prelude()
        .add_fn("saw", make_expr_node(|args| saw(args[0].clone())));
    koto.prelude()
        .add_fn("tri", make_expr_node(|args| triangle(args[0].clone())));
    koto.prelude()
        .add_fn("pulse", make_expr_node(|args| pulse(args[0].clone())));
    koto.prelude().add_fn("noise", make_expr_node(|_| noise()));
    koto.prelude().add_fn("env", move |ctx| {
        let args = ctx.args();
        if args.len() != 2 {
            return unexpected_args("expected 2 arguments: list, trig", args);
        }
        let segments = list_of_tuples_from_value(&args[0])?;
        let trig = node_from_kvalue(&args[1])?;

        Ok(KValue::Object(env(segments, trig).into()))
    });
    koto.prelude().add_fn("lp", move |ctx| {
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("expected 3 arguments: cutoff, resonance, input", args);
        }

        let cutoff = node_from_kvalue(&args[0])?;
        let resonance = node_from_kvalue(&args[1])?;
        let input = node_from_kvalue(&args[2])?;

        Ok(KValue::Object(
            svf(input, cutoff, resonance, FilterMode::Lowpass).into(),
        ))
    });
    koto.prelude().add_fn("bp", move |ctx| {
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("expected 3 arguments: cutoff, resonance, input", args);
        }

        let cutoff = node_from_kvalue(&args[0])?;
        let resonance = node_from_kvalue(&args[1])?;
        let input = node_from_kvalue(&args[2])?;

        Ok(KValue::Object(
            svf(input, cutoff, resonance, FilterMode::Bandpass).into(),
        ))
    });
    koto.prelude().add_fn("hp", move |ctx| {
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("expected 3 arguments: cutoff, resonance, input", args);
        }

        let cutoff = node_from_kvalue(&args[0])?;
        let resonance = node_from_kvalue(&args[1])?;
        let input = node_from_kvalue(&args[2])?;

        Ok(KValue::Object(
            svf(input, cutoff, resonance, FilterMode::Highpass).into(),
        ))
    });
    koto.prelude().add_fn("seq", move |ctx| {
        let args = ctx.args();
        if args.len() != 2 {
            return unexpected_args("expected 2 arguments: list, trig", args);
        }

        let values = list_from_value(&args[0])?;
        let trig = node_from_kvalue(&args[1])?;

        Ok(KValue::Object(seq(values, trig).into()))
    });
    koto.prelude().add_fn("pan", move |ctx| {
        let args = ctx.args();
        if args.len() != 2 {
            return unexpected_args("expected 2 arguments: pan, input", args);
        }

        let value = node_from_kvalue(&args[0])?;
        let input = node_from_kvalue(&args[1])?;

        Ok(KValue::Object(pan(input, value).into()))
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

        Ok(KValue::Object(pluck(freq, tone, damping, trig).into()))
    });
    koto.prelude().add_fn("reverb", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;

        Ok(KValue::Object(reverb(input).into()))
    });
    koto.prelude().add_fn("delay", move |ctx| {
        let args = ctx.args();
        let input = node_from_kvalue(&args[0])?;

        Ok(KValue::Object(delay(input).into()))
    });
    koto.prelude().add_fn("moog", move |ctx| {
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("expected 3 arguments: cutoff, resonance, input", args);
        }

        let cutoff = node_from_kvalue(&args[0])?;
        let resonance = node_from_kvalue(&args[1])?;
        let input = node_from_kvalue(&args[2])?;

        Ok(KValue::Object(moog(input, cutoff, resonance).into()))
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
