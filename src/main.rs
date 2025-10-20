use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use notify::{event::ModifyKind, EventKind, RecursiveMode, Watcher};
use std::{
    fs,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

mod audio_graph;
mod nodes;
mod utils;
use audio_graph::*;

mod parser;
use crate::parser::{tokenize, Parser};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let filename = if args.len() >= 2 {
        &args[1]
    } else {
        "patch.ohm"
    };

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
    let container_clone = Arc::clone(&container);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            container_clone.lock().unwrap().process_interleaved(data);
        },
        err_fn,
        None,
    )?;

    stream.play()?;

    let update_audio_graph = |path: &Path| -> Result<(), anyhow::Error> {
        let src = fs::read_to_string(path)?;
        let tokens = tokenize(src);
        let mut parser = Parser::new(tokens, config.sample_rate.0);

        match parser.parse() {
            Some(expr) => {
                let mut guard = container.lock().unwrap();
                guard.update_graph(expr);
                Ok(())
            }
            None => bail!("Failed to parse"),
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
