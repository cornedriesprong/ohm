use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, FromSample, SizedSample, StreamConfig,
};
use notify::{event::ModifyKind, EventKind, RecursiveMode, Watcher};
use std::{
    fs,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

mod nodes;
mod utils;

mod container;
use container::*;

mod parser;
use crate::parser::Parser;

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

pub fn run<T>(device: &Device, config: &StreamConfig, filename: &str) -> Result<(), anyhow::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let container = Container::new();
    let container = Arc::new(Mutex::new(container));
    let container_clone = Arc::clone(&container);

    let stream = device.build_output_stream(
        config,
        move |data, _| {
            if let Ok(mut container) = container_clone.lock() {
                container.process(data);
            }
        },
        |err| eprintln!("an error occurred on stream: {}", err),
        None,
    )?;

    stream.play()?;

    let update_graph = |path| -> Result<(), anyhow::Error> {
        let src = fs::read_to_string(path)?;
        let parser = Parser::new(src, config.sample_rate.0);
        let mut arena = Arena::new();
        let root = parser.parse(&mut arena);

        if let Ok(mut container) = container.lock() {
            container.update_graph(arena, root);
        }
        Ok(())
    };

    update_graph(Path::new(filename))?;

    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(tx)?;

    let path = Path::new(filename);
    watcher.watch(path, RecursiveMode::NonRecursive)?;

    let mut last_update = Instant::now();
    let debounce_duration = Duration::from_millis(100);

    let mut check_duration_and_update = || {
        let now = Instant::now();
        if now.duration_since(last_update) >= debounce_duration {
            if let Err(e) = update_graph(path) {
                eprintln!("Error updating audio graph: {}", e);
            }
            last_update = now;
        }
    };

    for res in rx {
        match res {
            Ok(event) => match event.kind {
                EventKind::Modify(ModifyKind::Data(_)) => {
                    check_duration_and_update();
                }
                EventKind::Remove(_) | EventKind::Modify(ModifyKind::Name(_)) => {
                    // re-watch the file
                    watcher.unwatch(path).ok();
                    watcher.watch(path, RecursiveMode::NonRecursive).ok();

                    check_duration_and_update();
                }
                _ => {}
            },
            Err(e) => eprintln!("Watch error: {}", e),
        }
    }

    Ok(())
}
