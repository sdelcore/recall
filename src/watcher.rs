use anyhow::Result;
use notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebouncedEventKind};
use std::path::Path;
use std::sync::mpsc::channel;
use std::time::Duration;

use crate::config::Config;
use crate::store::Store;

/// Watch every collection's root_path and auto-index modified files into the
/// collection that owns the changed file.
pub fn watch_directories(config: &Config) -> Result<()> {
    let store = Store::open()?;
    let collections = store.list_collections()?;
    if collections.is_empty() {
        anyhow::bail!(
            "No collections to watch. Add one with `recall collection add <path> --name <name>`."
        );
    }

    let (tx, rx) = channel();
    let debounce_duration = Duration::from_millis(config.watch.debounce_ms);
    let mut debouncer = new_debouncer(debounce_duration, tx)?;

    for c in &collections {
        if c.root_path.is_empty() {
            continue;
        }
        println!("Watching collection {:?}: {}", c.name, c.root_path);
        debouncer
            .watcher()
            .watch(Path::new(&c.root_path), RecursiveMode::Recursive)?;
    }

    println!("Excluding patterns: {:?}", config.watch.exclude);
    println!("Debounce: {}ms", config.watch.debounce_ms);
    println!();

    for result in rx {
        match result {
            Ok(events) => {
                for event in events {
                    if event.kind != DebouncedEventKind::Any {
                        continue;
                    }
                    let path_str = event.path.to_string_lossy().to_string();
                    if !path_str.ends_with(".md") {
                        continue;
                    }
                    if config.should_skip_watch(&path_str) {
                        continue;
                    }
                    if !event.path.exists() {
                        continue;
                    }
                    let target = match store.collection_for_path(&path_str)? {
                        Some(c) => c,
                        None => {
                            eprintln!("No collection owns {}, skipping", path_str);
                            continue;
                        }
                    };
                    println!("Changed [{}]: {}", target.name, path_str);
                    if let Err(e) = store.index_file(target.id, &path_str) {
                        eprintln!("Failed to index {}: {}", path_str, e);
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
