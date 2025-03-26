use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use koto::prelude::*;
use nodes::*;
use notify::{Event, RecursiveMode, Watcher};
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
    if args.len() != 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        std::process::exit(1);
    }

    let filename = &args[1];

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
                    let s = graph.tick();
                    for sample in frame.iter_mut() {
                        *sample = s;
                    }
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
                    let new = parse_to_audio_graph(expr.to_owned());
                    let mut guard = graph.lock().unwrap();
                    *guard = Some(new);

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
    watcher.watch(Path::new(filename), RecursiveMode::NonRecursive)?;

    let mut last_update = Instant::now();
    let debounce_duration = Duration::from_millis(100); // Adjust as needed

    for res in rx {
        match res {
            Ok(Event { kind, .. }) if kind.is_modify() => {
                let now = Instant::now();
                if now.duration_since(last_update) >= debounce_duration {
                    if let Err(e) = update_audio_graph(Path::new(filename)) {
                        eprintln!("Error updating audio graph: {}", e);
                    }
                    last_update = now;
                }
            }
            Err(e) => eprintln!("Watch error: {}", e),
            _ => {}
        }
    }
    Ok(())
}

fn create_env(koto: &Koto) {
    // add node creation functions to Koto environment
    koto.prelude()
        .add_fn("sine", make_expr_node(|args| sine(args[0].clone())));
    koto.prelude()
        .add_fn("square", make_expr_node(|args| square(args[0].clone())));
    koto.prelude()
        .add_fn("saw", make_expr_node(|args| saw(args[0].clone())));
    koto.prelude()
        .add_fn("pulse", make_expr_node(|args| pulse(args[0].clone())));
    koto.prelude().add_fn("noise", make_expr_node(|_| noise()));
    koto.prelude().add_fn("ar", move |ctx| {
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("3 arguments: attack, release, trig", args);
        }

        let attack = expr_from_kvalue(&args[0])?;
        let release = expr_from_kvalue(&args[1])?;
        let trig = expr_from_kvalue(&args[2])?;

        Ok(KValue::Object(ar(attack, release, trig).into()))
    });
    koto.prelude().add_fn("svf", move |ctx| {
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("3 arguments: cutoff, resonance, input", args);
        }

        let cutoff = expr_from_kvalue(&args[0])?;
        let resonance = expr_from_kvalue(&args[1])?;
        let input = expr_from_kvalue(&args[2])?;

        Ok(KValue::Object(svf(cutoff, resonance, input).into()))
    });
    koto.prelude().add_fn("seq", move |ctx| {
        let args = ctx.args();
        if args.len() != 2 {
            return unexpected_args("2 arguments: list, trig", args);
        }

        let values: Vec<_> = if let KValue::List(list) = &args[0] {
            list.data()
                .iter()
                .map(|v| {
                    if let KValue::Number(n) = v {
                        Ok((*n).into())
                    } else {
                        unexpected_type("number", v)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            unexpected_type("list", &args[0])?
        };
        let trig = expr_from_kvalue(&args[1])?;

        Ok(KValue::Object(seq(values, trig).into()))
    });
    koto.prelude().add_fn("pipe", move |ctx| {
        let args = ctx.args();
        if args.len() != 2 {
            return unexpected_args("2 arguments: delay (in samples), input", args);
        }

        let delay = expr_from_kvalue(&args[0])?;
        let input = expr_from_kvalue(&args[1])?;

        Ok(KValue::Object(pipe(delay, input).into()))
    });
    koto.prelude().add_fn("pluck", move |ctx| {
        let args = ctx.args();
        if args.len() != 4 {
            return unexpected_args("4 arguments: frequency, tone, damping, trig", args);
        }

        let freq = expr_from_kvalue(&args[0])?;
        let tone = expr_from_kvalue(&args[1])?;
        let damping = expr_from_kvalue(&args[2])?;
        let trig = expr_from_kvalue(&args[3])?;

        Ok(KValue::Object(pluck(freq, tone, damping, trig).into()))
    });
    koto.prelude().add_fn("reverb", move |ctx| {
        let args = ctx.args();
        let input = expr_from_kvalue(&args[0])?;

        Ok(KValue::Object(reverb(input).into()))
    });
    koto.prelude().add_fn("delay", move |ctx| {
        let args = ctx.args();
        let input = expr_from_kvalue(&args[0])?;

        Ok(KValue::Object(delay(input).into()))
    });
}

fn make_expr_node<F>(node_constructor: F) -> impl KotoFunction
where
    F: Fn(Vec<NodeKind>) -> NodeKind + 'static,
{
    move |ctx| {
        let args: Result<Vec<NodeKind>, _> = ctx.args().iter().map(expr_from_kvalue).collect();
        Ok(KValue::Object(node_constructor(args?).into()))
    }
}
fn expr_from_kvalue(value: &KValue) -> Result<NodeKind, koto::runtime::Error> {
    match value {
        KValue::Number(n) => Ok(constant(n.into())),
        KValue::Object(obj) if obj.is_a::<NodeKind>() => Ok(obj.cast::<NodeKind>()?.to_owned()),
        unexpected => unexpected_type("number, expr, or list", unexpected)?,
    }
}
