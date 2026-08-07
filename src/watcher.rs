use anyhow::Result;
use notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebouncedEvent, DebouncedEventKind};
use std::path::Path;
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::embedder::{self, Embedder};
use crate::store::{self, Store};

/// How long the watcher waits for a path to stop changing before it indexes.
/// Obsidian and Syncthing both write a note several times in a burst; 1.5s is
/// long enough to collapse the burst and short enough that a save feels
/// indexed immediately.
const DEBOUNCE_MS: u64 = 1500;

/// How long indexed chunks may sit without a vector before a sweep embeds
/// them.
///
/// Indexing a note costs milliseconds; embedding its chunks costs a ~0.6s
/// model load plus ~110ms per chunk. Doing both inline would put the model on
/// the critical path of every save, and a sync that rewrites a hundred notes
/// would queue minutes of forward passes in front of the next one — so a note
/// saved during a bulk write would not be findable at all until the queue
/// drained. Keyword search is the half that decides whether a note is findable
/// at all; it stays first. Five minutes is long enough for a burst of writes
/// to cost one model load, and short enough that a note written now is
/// vector-searchable within the same sitting.
const EMBED_INTERVAL: Duration = Duration::from_secs(300);

/// How many chunks one sweep embeds before it returns to the event loop.
/// At ~9 chunks/sec that is about 15s, which is the longest a file change can
/// wait behind the embedder. A sweep that spends its whole budget on real work
/// schedules the next one immediately, so this bounds latency, not throughput:
/// a large backlog still drains at full speed, in slices.
const EMBED_BUDGET: usize = 128;

/// Watch every collection's root_path and auto-index modified files into the
/// collection that owns the changed file. Between events, embed the chunks
/// indexing produced — see [`Sweeper`].
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
    println!(
        "Embedding sweep: every {}s, up to {} chunks",
        EMBED_INTERVAL.as_secs(),
        EMBED_BUDGET
    );
    println!();

    let mut sweeper = Sweeper::default();
    let mut next_sweep = Instant::now() + EMBED_INTERVAL;

    loop {
        // The wait is whatever is left of the sweep interval, so an idle
        // watcher wakes up to embed and a busy one still handles events first.
        match rx.recv_timeout(next_sweep.saturating_duration_since(Instant::now())) {
            Ok(Ok(events)) => index_events(&store, &events)?,
            Ok(Err(e)) => eprintln!("Watch error: {:?}", e),
            Err(RecvTimeoutError::Timeout) => {}
            // The debouncer owns the sender, so this only happens on shutdown.
            Err(RecvTimeoutError::Disconnected) => break,
        }

        if Instant::now() >= next_sweep {
            next_sweep = if sweeper.sweep(&store) {
                // The budget ran out before the backlog did. Come straight
                // back, having given any queued events their turn.
                Instant::now()
            } else {
                Instant::now() + EMBED_INTERVAL
            };
        }
    }

    Ok(())
}

/// Index every changed `.md` file in one debounced batch.
///
/// Deletions are not handled here: an event whose path is gone is skipped, so
/// only `recall index` prunes a removed note.
fn index_events(store: &Store, events: &[DebouncedEvent]) -> Result<()> {
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
    Ok(())
}

/// The watcher's embedding half.
///
/// It embeds whatever the index is missing, not only what this process
/// indexed: the invariant worth holding is "the vectors keep up with the
/// chunks", and stating it that way also heals the backlog an earlier
/// `recall index` left behind. Without this the watcher indexed a note into
/// keyword search and left it there until someone ran `recall embed` by hand.
///
/// The model is held only while there is work. An idle vault should not cost
/// ~130MB of resident weights, and a reload is 0.6s against a sweep interval
/// measured in minutes.
#[derive(Default)]
struct Sweeper {
    embedder: Option<Embedder>,
    /// Set when the weights will not load. Indexing continues without them —
    /// keyword search matters more than vector search, and exiting would give
    /// neither — and the reason is printed once rather than every sweep.
    disabled: bool,
}

impl Sweeper {
    /// Embed up to [`EMBED_BUDGET`] pending chunks. Returns true when the
    /// budget ran out before the backlog did *and* the pass got work done —
    /// that is the caller's signal to come straight back, so a pass that
    /// embedded nothing must not send it there, or a model that fails on every
    /// batch turns the sweep into a busy loop.
    fn sweep(&mut self, store: &Store) -> bool {
        if self.disabled {
            return false;
        }

        // Counted first: an index that needs nothing must not load the model.
        let pending = match store.get_embedding_stats() {
            Ok((embedded, total)) => total - embedded,
            Err(e) => {
                eprintln!("Failed to count chunks without embeddings: {e}");
                return false;
            }
        };
        if pending == 0 {
            self.embedder = None;
            return false;
        }

        if self.embedder.is_none() {
            match Embedder::load() {
                Ok(embedder) => {
                    println!(
                        "Loaded {} from {}",
                        embedder::EMBEDDING_MODEL,
                        embedder.source()
                    );
                    self.embedder = Some(embedder);
                }
                Err(e) => {
                    eprintln!("Embedding disabled — the model failed to load: {e:#}");
                    eprintln!(
                        "Indexing continues; new notes stay keyword-only until this is fixed."
                    );
                    self.disabled = true;
                    return false;
                }
            }
        }
        let Some(embedder) = self.embedder.as_ref() else {
            return false;
        };

        match embedder::embed_pending(
            store,
            &|texts| embedder.embed_batch(texts),
            Some(EMBED_BUDGET),
            |_, _| {},
        ) {
            Ok(progress) => {
                println!(
                    "Embedded chunks: {} ({} pending)",
                    progress.embedded,
                    pending - progress.embedded as i64
                );
                progress.attempted == EMBED_BUDGET && progress.embedded > 0
            }
            Err(e) => {
                eprintln!("Embedding sweep failed: {e}");
                false
            }
        }
    }
}
