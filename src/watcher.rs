use anyhow::Result;
use notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebouncedEventKind};
use std::path::Path;
use std::sync::mpsc::channel;
use std::time::Duration;

use crate::store::{self, Store};

/// How long the watcher waits for a path to stop changing before it indexes.
/// Obsidian and Syncthing both write a note several times in a burst; 1.5s is
/// long enough to collapse the burst and short enough that a save feels
/// indexed immediately.
const DEBOUNCE_MS: u64 = 1500;

/// Watch every collection's root_path and auto-index modified files into the
/// collection that owns the changed file.
pub fn watch_directories() -> Result<()> {
    let store = Store::open()?;
    let collections = store.list_collections()?;
    if collections.is_empty() {
        anyhow::bail!(
            "No collections to watch. Add one with `recall collection add <path> --name <name>`."
        );
    }

    let (tx, rx) = channel();
    let debounce_duration = Duration::from_millis(DEBOUNCE_MS);
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

    println!("Excluding patterns: {:?}", store::EXCLUDE_GLOBS);
    println!("Debounce: {}ms", DEBOUNCE_MS);
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
                    if store::is_excluded(&path_str) {
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
