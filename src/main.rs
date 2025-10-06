use anyhow::bail;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SizedSample,
};
use koto::prelude::*;
use notify::{event::ModifyKind, EventKind, RecursiveMode, Watcher};
use std::{
    fs,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

mod nodes;
mod utils;

mod env;
use crate::env::create_env;

mod op;
use crate::op::Op;

mod audio_graph;
use audio_graph::*;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let filename = if args.len() >= 2 {
        &args[1]
    } else {
        "patch.koto"
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
    let container_clone1 = Arc::clone(&container);
    let container_clone2 = Arc::clone(&container);

    let mut koto = Koto::default();
    create_env(&koto, container_clone1, config.sample_rate.0);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            container_clone2.lock().unwrap().process_interleaved(data);
        },
        err_fn,
        None,
    )?;

    stream.play()?;

    let mut update_audio_graph = |path: &Path| -> Result<(), anyhow::Error> {
        let src = fs::read_to_string(path)?;
        match koto.compile_and_run(CompileArgs {
            script: &src,
            script_path: Some(KString::from(path.to_str().unwrap())),
            compiler_settings: CompilerSettings {
                export_top_level_ids: true,
                ..Default::default()
            },
        })? {
            KValue::Object(obj) => match obj.cast::<Op>() {
                Ok(expr) => {
                    let graph = parse_to_graph(expr.to_owned());
                    let mut guard = container.lock().unwrap();
                    guard.update_graph(graph);
                    Ok(())
                }
                Err(e) => bail!("Failed to cast to Expr: {}", e),
            },
            KValue::Str(str) => {
                println!("{}", str);

                Ok(())
            }
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
