use anyhow::Context;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ohm_rs::{Limiter, Program, build, render};
use std::{fs, sync::mpsc, thread, time::Duration};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let filename: String = if args.len() >= 2 {
        args[1].clone()
    } else {
        "patch.ohm".to_string()
    };

    let src = fs::read_to_string(&filename)?;
    let (tx, rx) = mpsc::channel();
    let mut prog = compile_and_run(&src)?;
    thread::spawn(move || watch_file(&filename, src, tx));

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("no output device available")?;
    let supported_config = device.default_output_config()?;
    let channels = supported_config.channels() as usize;
    let config = supported_config.config();
    let mut limiter = Limiter::default();
    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                while let Ok(mut new) = rx.try_recv() {
                    new.copy_state_from(&prog);
                    prog = new;
                }
                for frame in data.chunks_mut(channels) {
                    render(&mut prog, frame, config.sample_rate);
                    limiter.process(frame, config.sample_rate);
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .context("failed to build output stream")?;

    stream.play().context("failed to start stream")?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();

    Ok(())
}

fn watch_file(name: &str, mut prev: String, tx: mpsc::Sender<Program>) {
    loop {
        thread::sleep(Duration::from_millis(100));

        if let Ok(src) = fs::read_to_string(name) {
            if src == prev {
                continue;
            }
            let result = compile_and_run(&src);
            prev = src;

            match result {
                Ok(prog) => {
                    println!("reloaded {name}");
                    _ = tx.send(prog);
                }
                Err(error) => eprintln!("{name} {error:#}"),
            }
        }
    }
}

fn compile_and_run(src: &str) -> anyhow::Result<Program> {
    build(
        &fs::read_to_string("prelude.ohm").context("prelude.ohm not found")?,
        src,
    )
}
