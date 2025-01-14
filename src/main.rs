use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use fundsp::hacker::{AudioUnit, BufferRef};
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
    koto.prelude().add_fn("pulse", add_fn(OperatorType::Pulse));
    koto.prelude().add_fn("noise", add_noise_fn());
    koto.prelude().add_fn("ar", move |ctx| {
        let args = ctx.args();
        if args.len() != 4 {
            return type_error_with_slice("four arguments: trig, attack, release, input", args);
        }

        let trig = ExprInput::from_kvalue(&args[0])?;
        let attack = ExprInput::from_kvalue(&args[1])?;
        let release = ExprInput::from_kvalue(&args[2])?;
        let input = ExprInput::from_kvalue(&args[3])?;

        Ok(KValue::Object(
            Expr::Operator {
                kind: OperatorType::AR,
                input: Box::new(input.clone().into_expr()),
                args: vec![trig.into_expr(), attack.into_expr(), release.into_expr()],
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

    let mut net = match koto.compile_and_run(&src)? {
        KValue::Object(obj) if obj.is_a::<Expr>() => {
            // parse_to_audio_graph(obj.cast::<Expr>()?.clone())
            parse_to_net(obj.cast::<Expr>()?.clone())
        }
        other => bail!(
            "Expected a Map, found '{}': ({})",
            other.type_as_string(),
            koto.value_to_string(other)?
        ),
    };

    let mut next_value = move || net.get_mono();

    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for frame in data.chunks_mut(2) {
                for sample in frame.iter_mut() {
                    *sample = next_value();
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
            1 => ExprInput::from_kvalue(&args[0])?.into_expr(),
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

#[derive(Clone)]
pub(crate) enum ExprInput {
    Number(f32),
    Expression(Expr),
}

impl ExprInput {
    pub fn from_kvalue(value: &KValue) -> Result<Self, koto::Error> {
        match value {
            KValue::Number(n) => Ok(ExprInput::Number(n.into())),
            KValue::Object(obj) if obj.is_a::<Expr>() => {
                Ok(ExprInput::Expression(obj.cast::<Expr>()?.to_owned()))
            }
            unexpected => type_error("number or expr", unexpected)?,
        }
    }

    pub fn into_expr(self) -> Expr {
        match self {
            ExprInput::Number(n) => Expr::Number(n),
            ExprInput::Expression(e) => e,
        }
    }
}
