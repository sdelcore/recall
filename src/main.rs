use anyhow::Result;
use clap::{Parser, Subcommand};

mod embedder;
mod store;
mod watcher;

#[derive(Parser)]
#[command(name = "memory-search")]
#[command(about = "Semantic memory search with token-efficient retrieval")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search query (shorthand for `memory-search search <query>`)
    #[arg(trailing_var_arg = true)]
    query: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Search memory for relevant information
    Search {
        /// The search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Output format: compact, json, full
        #[arg(short, long, default_value = "compact")]
        format: String,

        /// Only include results after this date (YYYY-MM-DD)
        #[arg(long)]
        after: Option<String>,

        /// Filter by project name
        #[arg(long)]
        project: Option<String>,

        /// Filter by file pattern (glob)
        #[arg(long)]
        file: Option<String>,

        /// Use hybrid search (BM25 + vector)
        #[arg(long)]
        hybrid: bool,

        /// Vector weight for hybrid search (0.0-1.0)
        #[arg(long, default_value = "0.7")]
        vector_weight: f64,
    },

    /// Generate embeddings for indexed chunks
    Embed {
        /// Only embed chunks that don't have embeddings
        #[arg(long)]
        incremental: bool,

        /// Maximum chunks to embed (for testing)
        #[arg(long)]
        limit: Option<usize>,
    },

    /// Index files into the memory database
    Index {
        /// Path to index (defaults to ~/Obsidian)
        #[arg(short, long)]
        path: Option<String>,

        /// Only index changed files
        #[arg(long)]
        incremental: bool,

        /// Index a single file
        #[arg(long)]
        file: Option<String>,
    },

    /// Show index status and statistics
    Status {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Watch for file changes and auto-index
    Watch {
        /// Path to watch (defaults to ~/Obsidian)
        #[arg(short, long)]
        path: Option<String>,

        /// Debounce time in milliseconds
        #[arg(long, default_value = "1500")]
        debounce: u64,
    },

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Set a configuration value
    Set {
        /// Configuration key
        key: String,
        /// Configuration value
        value: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle direct query (no subcommand)
    if !cli.query.is_empty() {
        let query = cli.query.join(" ");
        return run_search(&query, 5, "compact", None, None, None, false, 0.7).await;
    }

    match cli.command {
        Some(Commands::Search {
            query,
            limit,
            format,
            after,
            project,
            file,
            hybrid,
            vector_weight,
        }) => {
            run_search(&query, limit, &format, after, project, file, hybrid, vector_weight).await
        }
        Some(Commands::Index {
            path,
            incremental,
            file,
        }) => {
            run_index(path, incremental, file).await
        }
        Some(Commands::Embed { incremental, limit }) => {
            run_embed(incremental, limit).await
        }
        Some(Commands::Status { json }) => {
            run_status(json).await
        }
        Some(Commands::Watch { path, debounce }) => {
            run_watch(path, debounce).await
        }
        Some(Commands::Config { action }) => {
            run_config(action).await
        }
        None => {
            // No command and no query - show help
            use clap::CommandFactory;
            Cli::command().print_help()?;
            Ok(())
        }
    }
}

async fn run_search(
    query: &str,
    limit: usize,
    format: &str,
    after: Option<String>,
    project: Option<String>,
    file: Option<String>,
    hybrid: bool,
    vector_weight: f64,
) -> Result<()> {
    let store = store::Store::open()?;

    // Build search options from filters
    let options = store::SearchOptions {
        after,
        project,
        file_pattern: file,
    };

    let results = if hybrid {
        // Hybrid search with embeddings
        let embedder = embedder::Embedder::new();

        // Check if embeddings are available
        let (embedded, _) = store.get_embedding_stats()?;
        if embedded == 0 {
            eprintln!("Warning: No embeddings found. Run 'memory-search embed' first.");
            eprintln!("Falling back to BM25 search.\n");
            store.search_fts_filtered(query, limit, &options)?
        } else {
            // Generate query embedding
            let query_embedding = embedder.embed(query).await?;
            let bm25_weight = 1.0 - vector_weight;

            store.search_hybrid(query, &query_embedding, limit, vector_weight, bm25_weight)?
        }
    } else {
        // BM25 only
        store.search_fts_filtered(query, limit, &options)?
    };

    match format {
        "json" => {
            let output = serde_json::json!({
                "query": query,
                "results": results.iter().map(|r| {
                    serde_json::json!({
                        "file": r.file_path,
                        "lines": format!("{}-{}", r.start_line, r.end_line),
                        "score": r.score,
                        "snippet": r.content,
                        "date": r.date,
                        "section": r.section,
                    })
                }).collect::<Vec<_>>()
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "full" => {
            println!("Found {} results for \"{}\":\n", results.len(), query);
            for (i, result) in results.iter().enumerate() {
                println!("[{}] {}:{}-{} (score: {:.2})",
                    i + 1, result.file_path, result.start_line, result.end_line, result.score);
                if let Some(section) = &result.section {
                    println!("Section: {}", section);
                }
                println!("{}\n", result.content);
            }
        }
        _ => {
            // Compact format (default)
            println!("Found {} results for \"{}\":\n", results.len(), query);
            for (i, result) in results.iter().enumerate() {
                println!("[{}] {}:{}-{} (score: {:.2})",
                    i + 1, result.file_path, result.start_line, result.end_line, result.score);
                // Truncate content for compact display
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

async fn run_index(path: Option<String>, incremental: bool, file: Option<String>) -> Result<()> {
    let store = store::Store::open()?;

    if let Some(file_path) = file {
        println!("Indexing single file: {}", file_path);
        store.index_file(&file_path)?;
        println!("Done.");
        return Ok(());
    }

    let index_path = path.unwrap_or_else(|| {
        dirs::home_dir()
            .map(|h| h.join("Obsidian").to_string_lossy().to_string())
            .unwrap_or_else(|| "~/Obsidian".to_string())
    });

    if incremental {
        println!("Incremental indexing: {}", index_path);
        store.index_incremental(&index_path)?;
    } else {
        println!("Full indexing: {}", index_path);
        store.index_full(&index_path)?;
    }

    let stats = store.get_stats()?;
    println!("Indexed {} files, {} chunks", stats.file_count, stats.chunk_count);

    Ok(())
}

async fn run_status(json: bool) -> Result<()> {
    let store = store::Store::open()?;
    let stats = store.get_stats()?;
    let (embedded, _) = store.get_embedding_stats()?;

    if json {
        let output = serde_json::json!({
            "file_count": stats.file_count,
            "chunk_count": stats.chunk_count,
            "embedded_count": embedded,
            "last_indexed": stats.last_indexed,
            "database_path": store.path(),
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Memory Search Status");
        println!("====================");
        println!("Database: {}", store.path());
        println!("Files indexed: {}", stats.file_count);
        println!("Chunks stored: {}", stats.chunk_count);
        println!("Embeddings: {}/{} ({:.1}%)",
            embedded, stats.chunk_count,
            if stats.chunk_count > 0 { (embedded as f64 / stats.chunk_count as f64) * 100.0 } else { 0.0 }
        );
        if let Some(last) = stats.last_indexed {
            println!("Last indexed: {}", last);
        }
    }

    Ok(())
}

async fn run_watch(path: Option<String>, debounce: u64) -> Result<()> {
    let watch_path = path.unwrap_or_else(|| {
        dirs::home_dir()
            .map(|h| h.join("Obsidian").to_string_lossy().to_string())
            .unwrap_or_else(|| "~/Obsidian".to_string())
    });

    println!("Watching {} for changes (debounce: {}ms)", watch_path, debounce);
    println!("Press Ctrl+C to stop\n");

    watcher::watch_directory(&watch_path, debounce)
}

async fn run_embed(incremental: bool, limit: Option<usize>) -> Result<()> {
    let store = store::Store::open()?;
    let embedder = embedder::Embedder::new();

    // Check Ollama connectivity
    println!("Checking Ollama connectivity...");
    if !embedder.health_check().await? {
        anyhow::bail!("Cannot connect to Ollama at http://nightman.tap:11434. Is it running?");
    }
    println!("Ollama is available.");

    // Get chunks that need embeddings
    let chunks = if incremental {
        store.get_chunks_without_embeddings()?
    } else {
        // For non-incremental, we'd need to get all chunks
        // For now, just do incremental (only missing)
        store.get_chunks_without_embeddings()?
    };

    let total = match limit {
        Some(l) => chunks.len().min(l),
        None => chunks.len(),
    };

    if total == 0 {
        println!("All chunks already have embeddings.");
        return Ok(());
    }

    println!("Generating embeddings for {} chunks...\n", total);

    let mut success_count = 0;
    let mut error_count = 0;

    for (i, (chunk_id, content)) in chunks.iter().take(total).enumerate() {
        // Progress indicator
        print!("\r[{}/{}] Embedding chunk {}...", i + 1, total, chunk_id);
        std::io::Write::flush(&mut std::io::stdout())?;

        match embedder.embed(content).await {
            Ok(embedding) => {
                if let Err(e) = store.store_embedding(*chunk_id, &embedding) {
                    eprintln!("\nFailed to store embedding for chunk {}: {}", chunk_id, e);
                    error_count += 1;
                } else {
                    success_count += 1;
                }
            }
            Err(e) => {
                eprintln!("\nFailed to generate embedding for chunk {}: {}", chunk_id, e);
                error_count += 1;
            }
        }
    }

    println!("\n\nEmbedding complete:");
    println!("  Success: {}", success_count);
    if error_count > 0 {
        println!("  Errors: {}", error_count);
    }

    let (embedded, total_chunks) = store.get_embedding_stats()?;
    println!("  Total embedded: {}/{} chunks ({:.1}%)",
        embedded, total_chunks,
        (embedded as f64 / total_chunks as f64) * 100.0
    );

    Ok(())
}

async fn run_config(action: ConfigAction) -> Result<()> {
    match action {
        ConfigAction::Show => {
            println!("Configuration:");
            println!("  index.paths: [\"~/Obsidian/\"]");
            println!("  embeddings.ollama_url: http://nightman.tap:11434");
            println!("  embeddings.model: nomic-embed-text");
            println!("  search.default_limit: 5");
            println!("  search.vector_weight: 0.7");
            println!("  search.bm25_weight: 0.3");
            println!("\nConfig file: ~/.config/memory-search/config.toml");
        }
        ConfigAction::Set { key, value } => {
            println!("Setting {} = {}", key, value);
            println!("Configuration management not yet implemented");
        }
    }

    Ok(())
}
