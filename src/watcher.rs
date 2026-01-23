use anyhow::Result;
use notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebouncedEventKind};
use std::path::Path;
use std::sync::mpsc::channel;
use std::time::Duration;

use crate::store::Store;

/// Watch a directory for changes and auto-index modified files
pub fn watch_directory(path: &str, debounce_ms: u64) -> Result<()> {
    let (tx, rx) = channel();

    let debounce_duration = Duration::from_millis(debounce_ms);
    let mut debouncer = new_debouncer(debounce_duration, tx)?;

    debouncer
        .watcher()
        .watch(Path::new(path), RecursiveMode::Recursive)?;

    println!("Watching {} for changes...", path);

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
                        if should_skip(&path_str) {
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

fn should_skip(path: &str) -> bool {
    let exclude_patterns = [
        "Templates/",
        ".obsidian/",
        "attachments/",
        ".sync-conflict-",
    ];

    exclude_patterns.iter().any(|pattern| path.contains(pattern))
}
