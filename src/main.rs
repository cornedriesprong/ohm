use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use koto::prelude::*;
use std::fs;
use std::time::Duration;

mod nodes;

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

pub fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> Result<(), anyhow::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let src = fs::read_to_string("./script.koto").unwrap();
    let mut koto = Koto::default();

    // add node creation functions to Koto env
    koto.prelude().add_fn("sine", add_fn(OperatorType::Sine));
    koto.prelude()
        .add_fn("square", add_fn(OperatorType::Square));
    koto.prelude().add_fn("noise", add_noise_fn());
    koto.prelude().add_fn("skew", add_fn(OperatorType::Skew));

    koto.prelude().add_fn("mix", move |ctx| match ctx.args() {
        [KValue::Object(lhs), KValue::Object(rhs)] => Ok(KValue::Object(
            Expr::Operator {
                kind: OperatorType::Mix,
                input: Box::new(lhs.cast::<Expr>()?.to_owned()),
                args: vec![rhs.cast::<Expr>()?.to_owned()],
            }
            .into(),
        )),
        unexpected => type_error_with_slice("a number", unexpected),
    });

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
                let s = audio_graph.process();
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
    move |ctx| match ctx.args() {
        [] => Ok(KValue::Object(
            Expr::Operator {
                kind: op_type,
                input: Box::new(Expr::Number(440.0)),
                args: vec![],
            }
            .into(),
        )),
        [KValue::Number(hz)] => Ok(KValue::Object(
            Expr::Operator {
                kind: op_type,
                input: Box::new(Expr::Number(hz.into())),
                args: vec![],
            }
            .into(),
        )),
        [KValue::Object(obj)] if obj.is_a::<Expr>() => Ok(KValue::Object(
            Expr::Operator {
                kind: op_type,
                input: Box::new(obj.cast::<Expr>()?.to_owned()),
                args: vec![],
            }
            .into(),
        )),
        unexpected => type_error_with_slice("a number", unexpected),
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
