use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use koto::prelude::*;
use notify::{Event, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::{
    fs,
    time::{Duration, Instant},
};

mod nodes;
mod utils;

mod audio_graph;
use audio_graph::*;

mod parser;
use parser::*;

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

    let audio_graph: Arc<Mutex<Option<AudioGraph>>> = Arc::new(Mutex::new(None));
    let audio_graph_clone = audio_graph.clone();

    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            if let Some(graph) = audio_graph_clone.lock().unwrap().as_mut() {
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
            KValue::Object(obj) if obj.is_a::<Expr>() => match obj.cast::<Expr>() {
                Ok(expr) => {
                    let new = parse_to_audio_graph(expr.to_owned());
                    let mut guard = audio_graph.lock().unwrap();

                    if let Some(old) = guard.as_mut() {
                        // apply a diff between the old and new graphs to avoid
                        // discontinuities in the audio
                        let (update, add, remove) = diff_graph(&old, &new);
                        println!("--------------");
                        println!("update: {:?}", update);
                        println!("add: {:?}", add);
                        println!("remove: {:?}", remove);

                        for (id, node) in update {
                            old.replace_node(id, node);
                        }

                        for (_, node) in add {
                            old.add_node(node);
                        }

                        for id in remove {
                            old.remove_node(id);
                        }

                        old.reconnect_edges(&new);
                    } else {
                        // first time creating the graph, no diff needed
                        println!("new graph");
                        *guard = Some(new);
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
    koto.prelude().add_fn("sine", add_fn(OperatorType::Sine));
    koto.prelude()
        .add_fn("square", add_fn(OperatorType::Square));
    koto.prelude().add_fn("saw", add_fn(OperatorType::Saw));
    koto.prelude().add_fn("pulse", add_fn(OperatorType::Pulse));
    koto.prelude().add_fn("noise", add_noise_fn());
    koto.prelude().add_fn("ar", move |ctx| {
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("3 arguments: attack, release, trig", args);
        }

        let attack = expr_from_kvalue(&args[0])?;
        let release = expr_from_kvalue(&args[1])?;
        let trig = expr_from_kvalue(&args[2])?;

        Ok(KValue::Object(
            Expr::Operator {
                kind: OperatorType::AR,
                input: Box::new(trig),
                args: vec![attack, release],
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("svf", move |ctx| {
        let args = ctx.args();
        if args.len() != 3 {
            return unexpected_args("3 arguments: cutoff, resonance, input", args);
        }

        let cutoff = expr_from_kvalue(&args[0])?;
        let resonance = expr_from_kvalue(&args[1])?;
        let input = expr_from_kvalue(&args[2])?;

        Ok(KValue::Object(
            Expr::Operator {
                kind: OperatorType::SVF,
                input: Box::new(input),
                args: vec![cutoff, resonance],
            }
            .into(),
        ))
    });
    koto.prelude().add_fn("seq", move |ctx| {
        let args = ctx.args();
        if args.len() != 2 {
            return unexpected_args("2 arguments: list, trig", args);
        }

        let list = expr_from_kvalue(&args[0])?;
        let trig = expr_from_kvalue(&args[1])?;

        Ok(KValue::Object(
            Expr::Operator {
                kind: OperatorType::Seq,
                input: Box::new(trig),
                args: vec![list],
            }
            .into(),
        ))
    });
}

// TODO: define separate function for noise operator, which doesn't take any arguments
fn add_fn(op_type: OperatorType) -> impl KotoFunction {
    move |ctx| {
        let args = ctx.args();
        let input = match args.len() {
            0 => Expr::Number(440.0),
            1 => expr_from_kvalue(&args[0])?,
            _ => return unexpected_args("zero or one argument", &args),
        };

        Ok(KValue::Object(
            Expr::Operator {
                kind: op_type,
                input: Box::new(input),
                args: vec![],
            }
            .into(),
        ))
    }
}

fn add_noise_fn() -> impl KotoFunction {
    move |ctx| match ctx.args() {
        [] => Ok(KValue::Object(
            Expr::Operator {
                kind: OperatorType::Noise,
                input: Box::new(Expr::Number(0.)), // TODO: fix
                args: vec![],
            }
            .into(),
        )),
        unexpected => unexpected_args("no arguments", &unexpected),
    }
}

fn expr_from_kvalue(value: &KValue) -> Result<Expr, koto::runtime::Error> {
    match value {
        KValue::Number(n) => Ok(Expr::Number(n.into())),
        KValue::Object(obj) if obj.is_a::<Expr>() => Ok(obj.cast::<Expr>()?.to_owned()),
        KValue::List(list) => {
            let vec: Result<Vec<f32>, _> = list
                .data()
                .iter()
                .map(|v| match v {
                    KValue::Number(n) => Ok((*n).into()),
                    _ => unexpected_type("number", v),
                })
                .collect();
            Ok(Expr::List(vec?))
        }
        unexpected => unexpected_type("number, expr, or list", unexpected)?,
    }
}
