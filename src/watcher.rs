use anyhow::Result;
use notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebouncedEventKind};
use std::path::Path;
use std::sync::mpsc::channel;
use std::time::Duration;

use crate::config::Config;
use crate::store::Store;

/// Watch directories for changes and auto-index modified files
pub fn watch_directories(config: &Config) -> Result<()> {
    let (tx, rx) = channel();

    let debounce_duration = Duration::from_millis(config.watch.debounce_ms);
    let mut debouncer = new_debouncer(debounce_duration, tx)?;

    let watch_paths = config.watch_paths();

    for path in &watch_paths {
        println!("Watching: {}", path);
        debouncer
            .watcher()
            .watch(Path::new(path), RecursiveMode::Recursive)?;
    }

    println!("Excluding patterns: {:?}", config.watch.exclude);
    println!("Debounce: {}ms", config.watch.debounce_ms);
    println!();

    let store = Store::open()?;

    for result in rx {
        match result {
            Ok(events) => {
                for event in events {
                    if event.kind == DebouncedEventKind::Any {
                        let path_str = event.path.to_string_lossy().to_string();

                        // Only process markdown files
                        if !path_str.ends_with(".md") {
                            continue;
                        }

                        // Skip excluded patterns
                        if config.should_skip_watch(&path_str) {
                            continue;
                        }

                        // Check if file exists (might be a delete event)
                        if event.path.exists() {
                            println!("Changed: {}", path_str);
                            if let Err(e) = store.index_file(&path_str) {
                                eprintln!("Failed to index {}: {}", path_str, e);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Watch error: {:?}", e);
            }
        }
    }

    Ok(())
}
