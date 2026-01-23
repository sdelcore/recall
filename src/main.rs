use anyhow::Result;
use clap::{Parser, Subcommand};

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
        return run_search(&query, 5, "compact", None, None, None).await;
    }

    match cli.command {
        Some(Commands::Search {
            query,
            limit,
            format,
            after,
            project,
            file,
        }) => {
            run_search(&query, limit, &format, after, project, file).await
        }
        Some(Commands::Index {
            path,
            incremental,
            file,
        }) => {
            run_index(path, incremental, file).await
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
) -> Result<()> {
    let store = store::Store::open()?;

    // Build search options from filters
    let options = store::SearchOptions {
        after,
        project,
        file_pattern: file,
    };

    // FTS5 search with filters (vector search comes later)
    let results = store.search_fts_filtered(query, limit, &options)?;

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

    if json {
        let output = serde_json::json!({
            "file_count": stats.file_count,
            "chunk_count": stats.chunk_count,
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
