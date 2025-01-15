use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use koto::prelude::*;
use std::fs;
use std::time::Duration;

mod nodes;
mod utils;

mod audio_graph;
use audio_graph::*;

mod parser;
use parser::*;

fn main() -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("Failed to get default output device");
    let config = device.default_output_config().unwrap();

    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()),
        sample_format => panic!("Unsupported sample format '{sample_format}'"),
    }
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
            return type_error_with_slice("3 arguments: attack, release, trig", args);
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
            return type_error_with_slice("3 arguments: cutoff, resonance, input", args);
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
            return type_error_with_slice("2 arguments: list, trig", args);
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

pub fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> Result<(), anyhow::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);
    let src = fs::read_to_string("./script.koto").unwrap();
    let mut koto = Koto::default();
    create_env(&koto);

    let mut audio_graph = match koto.compile_and_run(&src)? {
        KValue::Object(obj) if obj.is_a::<Expr>() => {
            parse_to_audio_graph(obj.cast::<Expr>()?.clone())
        }
        other => bail!(
            "Expected a Map, found '{}': ({})",
            other.type_as_string(),
            koto.value_to_string(other)?
        ),
    };

    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for frame in data.chunks_mut(2) {
                let s = audio_graph.tick();
                for sample in frame.iter_mut() {
                    *sample = s;
                }
            }
        },
        err_fn,
        None,
    )?;
    stream.play()?;

    std::thread::sleep(Duration::from_millis(360000));

    Ok(())
}

// TODO: define separate function for noise operator, which doesn't take any arguments
fn add_fn(op_type: OperatorType) -> impl KotoFunction {
    move |ctx| {
        let args = ctx.args();
        let input = match args.len() {
            0 => Expr::Number(440.0),
            1 => expr_from_kvalue(&args[0])?,
            _ => return type_error_with_slice("zero or one argument", &args),
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
        unexpected => type_error_with_slice("no arguments", unexpected),
    }
}

fn expr_from_kvalue(value: &KValue) -> Result<Expr, koto::Error> {
    match value {
        KValue::Number(n) => Ok(Expr::Number(n.into())),
        KValue::Object(obj) if obj.is_a::<Expr>() => Ok(obj.cast::<Expr>()?.to_owned()),
        KValue::List(list) => {
            let vec: Result<Vec<f32>, _> = list
                .data()
                .iter()
                .map(|v| match v {
                    KValue::Number(n) => Ok((*n).into()),
                    _ => type_error("number", v),
                })
                .collect();
            Ok(Expr::List(vec?))
        }
        unexpected => type_error("number, expr, or list", unexpected),
    }
}
