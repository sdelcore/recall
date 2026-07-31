use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

mod ast;
mod chunker;
mod embedder;
mod frontmatter;
mod intent;
mod lint;
mod mcp;
mod reranker;
mod search;
mod store;
mod watcher;

#[derive(Parser)]
#[command(name = "recall")]
#[command(about = "Semantic memory search with token-efficient retrieval")]
#[command(version)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Search memory for relevant information
    Search {
        /// The search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value_t = 5)]
        limit: usize,

        /// Output format: compact, json
        #[arg(short, long, default_value = "compact")]
        format: String,

        /// Only include results after this date (YYYY-MM-DD)
        #[arg(long)]
        after: Option<String>,

        /// Only include results before this date (YYYY-MM-DD)
        #[arg(long)]
        before: Option<String>,

        /// Restrict to a single collection by name
        #[arg(long)]
        collection: Option<String>,

        /// Rerank results using an LLM. Costs seconds; fails loudly if the
        /// model is unreachable rather than returning unreranked results.
        #[arg(long)]
        rerank: bool,

        /// Emit per-result diagnostic info (BM25 rank, vec rank, RRF score,
        /// reranker score). Forces JSON output.
        #[arg(long)]
        trace: bool,
    },

    /// Generate embeddings for indexed chunks that don't have one yet
    Embed,

    /// Index files into the memory database
    Index {
        /// Collection name. If omitted, indexes every collection at its
        /// registered root_path.
        #[arg(long)]
        collection: Option<String>,
    },

    /// Manage named collections (one root_path per collection)
    Collection {
        #[command(subcommand)]
        action: CollectionAction,
    },

    /// Database maintenance: VACUUM, FTS rebuild
    Maintenance {
        #[command(subcommand)]
        action: MaintenanceAction,
    },

    /// Check vault links: dangling wikilinks and orphaned notes
    #[command(
        long_about = "Check vault links: dangling wikilinks and orphaned notes.\n\n\
        A link resolves when ANY registered collection holds the target, so a \
        cross-project link is not a finding. Only an UNRESOLVED link — target found \
        nowhere — is reported.\n\n\
        An ORPHAN is a note with zero incoming AND zero outgoing wikilinks. Notes \
        that are daily notes or session logs are never reported as orphans.\n\n\
        Links inside fenced code blocks, inline code, and %%Obsidian comments%% are \
        ignored. Resolution matches note basenames and frontmatter `aliases:`.\n\n\
        Warn-only: lint reads the filesystem, changes nothing, and never runs as part \
        of indexing. Findings never change the exit code; only a usage error does. The \
        report goes to stdout; redirect it somewhere outside the vault if you want to \
        keep it."
    )]
    Lint {
        /// Restrict findings to a single collection by name
        #[arg(long)]
        collection: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show index status, statistics, and health
    Status {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Watch for file changes and auto-index
    Watch,

    /// Start MCP server (stdio transport) for Claude Code integration
    Serve,
}

#[derive(Subcommand)]
enum MaintenanceAction {
    /// VACUUM the database (reclaims space after deletes)
    Vacuum,
    /// Drop and rebuild the FTS5 index from chunks
    RebuildFts,
}

#[derive(Subcommand)]
enum CollectionAction {
    /// Register a new collection rooted at <path>
    Add {
        /// Root directory the collection indexes
        path: String,
        /// Name for the collection (must be unique)
        #[arg(long)]
        name: String,
    },
    /// Set or clear a collection's recency half-life (days)
    HalfLife {
        /// Collection name
        collection: String,
        /// Half-life in days. Omit to clear it.
        days: Option<f64>,
    },
    /// Set or clear the description returned alongside this collection's hits
    Describe {
        /// Collection name
        collection: String,
        /// Description text. Omit to clear it.
        description: Option<String>,
    },
    /// List all collections
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    /// Remove a collection (drops its files, chunks, and embeddings)
    Remove {
        /// Name of the collection to remove
        name: String,
    },
}

/// Expand a leading `~/` to the home directory. Paths reach recall from a
/// shell that may not have expanded them (a nix activation script, an MCP
/// argument), and a literal `~` directory is never what the caller meant.
fn expand_home(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    match cli.command {
        Commands::Search {
            query,
            limit,
            format,
            after,
            before,
            collection,
            rerank,
            trace,
        } => {
            let request = search::SearchRequest {
                query,
                limit,
                collection,
                after,
                before,
                rerank,
            };
            let format = if trace { "json" } else { format.as_str() };
            run_search(request, format, trace).await
        }
        Commands::Index { collection } => run_index(collection).await,
        Commands::Collection { action } => run_collection(action),
        Commands::Maintenance { action } => run_maintenance(action),
        Commands::Lint { collection, json } => run_lint(collection, json),
        Commands::Embed => run_embed(),
        Commands::Status { json } => run_status(json).await,
        Commands::Watch => run_watch(),
        Commands::Serve => mcp::serve_mcp().await,
    }
}

async fn run_search(request: search::SearchRequest, format: &str, trace: bool) -> Result<()> {
    let outcome = search::search(request).await?;
    let results: Vec<&store::SearchResult> = outcome.results.iter().map(|(r, _)| r).collect();

    match format {
        "json" => {
            let output = serde_json::json!({
                "query": outcome.query,
                "results": outcome.results.iter().map(|(r, t)| {
                    let mut obj = serde_json::json!({
                        "file": r.file_path,
                        "lines": format!("{}-{}", r.start_line, r.end_line),
                        "score": r.score,
                        "snippet": r.content,
                        "date": r.date,
                        "date_source": r.date_source,
                        "section": r.section,
                        "status": r.status,
                        "collection": {
                            "name": r.collection_name,
                            "description": r.collection_description,
                        },
                    });
                    if trace {
                        obj["trace"] = serde_json::json!({
                            "bm25_rank": t.bm25_rank,
                            "vec_rank": t.vec_rank,
                            "rrf_score": t.rrf_score,
                            "rerank_score": t.rerank_score,
                            "decay_factor": t.decay_factor,
                            "pre_decay_score": t.pre_decay_score,
                        });
                    }
                    obj
                }).collect::<Vec<_>>()
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        _ => {
            // Compact format (default)
            println!(
                "Found {} results for \"{}\":\n",
                results.len(),
                outcome.query
            );
            for (i, result) in results.iter().enumerate() {
                println!(
                    "[{}] {}:{}-{} (score: {:.2})",
                    i + 1,
                    result.file_path,
                    result.start_line,
                    result.end_line,
                    result.score
                );
                let snippet: String = result.content.chars().take(200).collect();
                let snippet = if result.content.len() > 200 {
                    format!("{}...", snippet.trim())
                } else {
                    snippet.trim().to_string()
                };
                println!("{}\n", snippet);
            }
        }
    }

    Ok(())
}

async fn run_index(collection: Option<String>) -> Result<()> {
    let store = store::Store::open()?;

    // Resolve target collections.
    let targets: Vec<store::Collection> = if let Some(name) = collection.as_deref() {
        let c = store.get_collection(name)?.ok_or_else(|| {
            anyhow::anyhow!(
                "Collection {:?} not found. Add one with `recall collection add <path> --name <name>`.",
                name
            )
        })?;
        vec![c]
    } else {
        let cs = store.list_collections()?;
        if cs.is_empty() {
            anyhow::bail!(
                "No collections registered. Add one with \
                 `recall collection add <path> --name <name>` and re-run."
            );
        }
        cs
    };

    for target in &targets {
        let index_path = target.root_path.clone();
        if index_path.is_empty() {
            anyhow::bail!("Collection {:?} has no root_path to index.", target.name);
        }
        println!("Indexing collection {:?}: {}", target.name, index_path);
        store.index(target.id, &index_path)?;
    }

    let stats = store.get_stats()?;
    println!(
        "Indexed {} files, {} chunks",
        stats.file_count, stats.chunk_count
    );
    Ok(())
}

fn run_collection(action: CollectionAction) -> Result<()> {
    let store = store::Store::open()?;
    match action {
        CollectionAction::Add { path, name } => {
            let abs = expand_home(&path);
            // A root that does not resolve is a typo, not a value to store.
            // `recall collection add ~/Obsidan` used to succeed; the root then
            // matched no file on disk, so indexing and the watcher both did
            // nothing forever and said nothing about it.
            let canon = std::path::Path::new(&abs)
                .canonicalize()
                .with_context(|| format!("Collection root does not exist: {}", abs))?
                .to_string_lossy()
                .to_string();
            let c = store.create_collection(&name, &canon)?;
            println!("Added collection {:?} → {}", c.name, c.root_path);
        }
        CollectionAction::HalfLife { collection, days } => {
            if let Some(d) = days {
                if d <= 0.0 {
                    anyhow::bail!("Half-life must be positive (got {})", d);
                }
            }
            if !store.set_collection_half_life(&collection, days)? {
                anyhow::bail!("Collection {:?} not found", collection);
            }
            match days {
                Some(d) => println!("Set half-life for {:?} to {} days", collection, d),
                None => println!("Cleared half-life for {:?}", collection),
            }
        }
        CollectionAction::Describe {
            collection,
            description,
        } => {
            if !store.set_collection_description(&collection, description.as_deref())? {
                anyhow::bail!("Collection {:?} not found", collection);
            }
            match description {
                Some(_) => println!("Set description for {:?}", collection),
                None => println!("Cleared description for {:?}", collection),
            }
        }
        CollectionAction::List { json } => {
            let cs = store.list_collections()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&cs)?);
            } else if cs.is_empty() {
                println!(
                    "No collections. Add one with `recall collection add <path> --name <name>`."
                );
            } else {
                for c in cs {
                    let half_life = match c.half_life_days {
                        Some(d) => format!("{}d", d),
                        None => "-".to_string(),
                    };
                    let description = c.description.as_deref().unwrap_or("-");
                    println!(
                        "{:<20} {:<8} {:<40} {}",
                        c.name, half_life, c.root_path, description
                    );
                }
            }
        }
        CollectionAction::Remove { name } => {
            if store.remove_collection(&name)? {
                println!("Removed collection {:?}", name);
            } else {
                anyhow::bail!("Collection {:?} not found", name);
            }
        }
    }
    Ok(())
}

fn run_maintenance(action: MaintenanceAction) -> Result<()> {
    let store = store::Store::open()?;
    match action {
        MaintenanceAction::Vacuum => {
            store.vacuum()?;
            println!("VACUUM complete.");
        }
        MaintenanceAction::RebuildFts => {
            store.rebuild_fts()?;
            println!("FTS5 index rebuilt.");
        }
    }
    Ok(())
}

/// Warn-only vault link check. Every collection is scanned so a link can be
/// resolved across collections; `collection` only narrows what is reported.
fn run_lint(collection: Option<String>, json: bool) -> Result<()> {
    let store = store::Store::open()?;
    let collections = store.list_collections()?;
    if collections.is_empty() {
        anyhow::bail!(
            "No collections registered. Add one with \
             `recall collection add <path> --name <name>` and re-run."
        );
    }
    if let Some(name) = collection.as_deref() {
        if !collections.iter().any(|c| c.name == name) {
            anyhow::bail!("Collection {:?} not found", name);
        }
    }

    let report = lint::lint(&collections, collection.as_deref())?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render_lint(&report));
    }
    Ok(())
}

fn render_lint(report: &lint::LintReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "Recall Lint");
    let _ = writeln!(s, "===========");
    let _ = writeln!(s, "Notes scanned:       {}", report.notes_scanned);
    let _ = writeln!(s, "Links:               {}", report.links_total);
    let _ = writeln!(s, "  resolved:          {}", report.resolved);
    let _ = writeln!(s, "  unresolved:        {}", report.unresolved.len());
    let _ = writeln!(s, "Orphans:             {}", report.orphans.len());

    if !report.unresolved.is_empty() {
        let _ = writeln!(s, "\nUnresolved links:");
        for link in &report.unresolved {
            let _ = writeln!(s, "  {}:{}  [[{}]]", link.file, link.line, link.target);
        }
    }
    if !report.orphans.is_empty() {
        let _ = writeln!(s, "\nOrphans (no incoming and no outgoing links):");
        for path in &report.orphans {
            let _ = writeln!(s, "  {}", path);
        }
    }
    s
}

/// Index status plus the read-only health checks. Both answer "is the index
/// usable?", so they are one command: a caller that reads the counts is
/// exactly the caller that needs to know the counts are trustworthy.
async fn run_status(json: bool) -> Result<()> {
    let store = store::Store::open()?;
    let stats = store.get_stats()?;
    let (embedded, _) = store.get_embedding_stats()?;
    let integrity = store.integrity_check()?;
    let orphans = store.orphan_counts()?;
    let collections = store.list_collections()?;
    let healthy =
        integrity == "ok" && orphans.chunks == 0 && orphans.files == 0 && orphans.embeddings == 0;

    if json {
        let output = serde_json::json!({
            "file_count": stats.file_count,
            "chunk_count": stats.chunk_count,
            "embedded_count": embedded,
            "last_indexed": stats.last_indexed,
            "database_path": store.path(),
            "collection_roots": collections.iter()
                .map(|c| c.root_path.as_str())
                .collect::<Vec<_>>(),
            "integrity": integrity,
            "orphans": {
                "chunks": orphans.chunks,
                "files": orphans.files,
                "embeddings": orphans.embeddings,
            },
            "healthy": healthy,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Recall Status");
        println!("=============");
        println!("Database: {}", store.path());
        println!();
        println!("Collections:");
        if collections.is_empty() {
            println!("  (none)");
        }
        for c in &collections {
            println!("  {:<20} {}", c.name, c.root_path);
        }
        println!();
        println!("Files indexed: {}", stats.file_count);
        println!("Chunks stored: {}", stats.chunk_count);
        println!(
            "Embeddings: {}/{} ({:.1}%)",
            embedded,
            stats.chunk_count,
            if stats.chunk_count > 0 {
                (embedded as f64 / stats.chunk_count as f64) * 100.0
            } else {
                0.0
            }
        );
        if let Some(last) = stats.last_indexed {
            println!("Last indexed: {}", last);
        }
        println!();
        println!("Integrity: {}", integrity);
        println!("Orphan chunks: {}", orphans.chunks);
        println!("Orphan files: {}", orphans.files);
        println!("Orphan embeddings: {}", orphans.embeddings);
        println!(
            "Health: {}",
            if healthy {
                "healthy"
            } else {
                "needs attention"
            }
        );
    }

    Ok(())
}

fn run_watch() -> Result<()> {
    println!("Recall File Watcher");
    println!("===================");
    watcher::watch_directories()
}

/// How many chunks go through the model in one forward pass. The batch is
/// where the throughput is; 32 keeps the padded activations small enough to
/// stay comfortable on a laptop.
const EMBED_BATCH: usize = 32;

fn run_embed() -> Result<()> {
    let store = store::Store::open()?;

    // Chunks that need embeddings. Embedding is always incremental: a chunk's
    // vector cannot go stale without the chunk itself being rewritten, and a
    // rewrite drops the old row.
    let chunks = store.get_chunks_without_embeddings()?;
    let total = chunks.len();

    if total == 0 {
        println!("All chunks already have embeddings.");
        return Ok(());
    }

    // Loading the model costs about 0.6s, so it happens once, here, and never
    // inside the loop.
    let embedder = embedder::Embedder::load()?;
    println!(
        "Generating embeddings for {} chunks using {} from {}...\n",
        total,
        embedder::EMBEDDING_MODEL,
        embedder.source()
    );

    let mut success_count = 0;
    let mut error_count = 0;

    for (done, batch) in chunks.chunks(EMBED_BATCH).enumerate() {
        let first = done * EMBED_BATCH + 1;
        print!("\r[{}/{}] Embedding...", first + batch.len() - 1, total);
        std::io::Write::flush(&mut std::io::stdout())?;

        let texts: Vec<&str> = batch.iter().map(|(_, content)| content.as_str()).collect();
        let embeddings = match embedder.embed_batch(&texts) {
            Ok(embeddings) => embeddings,
            Err(e) => {
                eprintln!(
                    "\nFailed to embed chunks {}..{}: {}",
                    first,
                    first + batch.len() - 1,
                    e
                );
                error_count += batch.len();
                continue;
            }
        };

        for ((chunk_id, _), embedding) in batch.iter().zip(embeddings) {
            if let Err(e) = store.store_embedding(*chunk_id, &embedding) {
                eprintln!("\nFailed to store embedding for chunk {}: {}", chunk_id, e);
                error_count += 1;
            } else {
                success_count += 1;
            }
        }
    }

    println!("\n\nEmbedding complete:");
    println!("  Success: {}", success_count);
    if error_count > 0 {
        println!("  Errors: {}", error_count);
    }

    let (embedded, total_chunks) = store.get_embedding_stats()?;
    println!(
        "  Total embedded: {}/{} chunks ({:.1}%)",
        embedded,
        total_chunks,
        (embedded as f64 / total_chunks as f64) * 100.0
    );

    Ok(())
}
