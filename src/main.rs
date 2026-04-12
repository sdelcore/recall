use anyhow::Result;
use clap::{Parser, Subcommand};

mod config;
mod embedder;
mod mcp;
mod reranker;
mod store;
mod watcher;

use config::Config;

#[derive(Parser)]
#[command(name = "recall")]
#[command(about = "Semantic memory search with token-efficient retrieval")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search query (shorthand for `recall search <query>`)
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
        #[arg(short, long)]
        limit: Option<usize>,

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

        /// Rerank results using LLM (uses config provider, or --rerank-provider)
        #[arg(long)]
        rerank: bool,

        /// Override reranking provider (claude-code, anthropic, ollama)
        #[arg(long)]
        rerank_provider: Option<String>,
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
        /// Path to index (defaults to config paths)
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
    Watch,

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Start MCP server (stdio transport) for Claude Code integration
    Serve {
        /// Transport mode
        #[arg(long, default_value = "mcp")]
        mode: String,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Show config file path
    Path,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::load()?;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    // Handle direct query (no subcommand)
    if !cli.query.is_empty() {
        let query = cli.query.join(" ");
        return run_search(&config, &query, None, "compact", None, None, None, false, false, None).await;
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
            rerank,
            rerank_provider,
        }) => {
            run_search(&config, &query, limit, &format, after, project, file, hybrid, rerank, rerank_provider).await
        }
        Some(Commands::Index {
            path,
            incremental,
            file,
        }) => {
            run_index(&config, path, incremental, file).await
        }
        Some(Commands::Embed { incremental, limit }) => {
            run_embed(&config, incremental, limit).await
        }
        Some(Commands::Status { json }) => {
            run_status(json).await
        }
        Some(Commands::Watch) => {
            run_watch(&config)
        }
        Some(Commands::Config { action }) => {
            run_config(&config, action)
        }
        Some(Commands::Serve { mode }) => {
            if mode != "mcp" {
                anyhow::bail!("Unknown serve mode: {}. Only 'mcp' is supported.", mode);
            }
            mcp::serve_mcp(&config).await
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
    config: &Config,
    query: &str,
    limit: Option<usize>,
    format: &str,
    after: Option<String>,
    project: Option<String>,
    file: Option<String>,
    hybrid: bool,
    rerank: bool,
    rerank_provider: Option<String>,
) -> Result<()> {
    let store = store::Store::open()?;
    let limit = limit.unwrap_or(config.search.default_limit);
    let do_rerank = rerank || config.reranking.enabled;

    // When reranking, fetch more candidates so the reranker has material to work with
    let fetch_limit = if do_rerank {
        config.reranking.candidates.max(limit)
    } else {
        limit
    };

    // Build search options from filters
    let options = store::SearchOptions {
        after,
        project,
        file_pattern: file,
    };

    let mut results = if hybrid {
        // Hybrid search with embeddings
        let embedder = embedder::Embedder::new_with_config(config);

        // Check if embeddings are available
        let (embedded, _) = store.get_embedding_stats()?;
        if embedded == 0 {
            eprintln!("Warning: No embeddings found. Run 'recall embed' first.");
            eprintln!("Falling back to BM25 search.\n");
            store.search_fts_filtered(query, fetch_limit, &options)?
        } else {
            // Generate query embedding
            let query_embedding = embedder.embed(query).await?;

            store.search_hybrid(query, &query_embedding, fetch_limit, config.search.rrf_k)?
        }
    } else {
        // BM25 only
        store.search_fts_filtered(query, fetch_limit, &options)?
    };

    // LLM reranking
    if do_rerank && !results.is_empty() {
        let mut rerank_config = config.reranking.clone();
        // Override provider if specified on CLI
        if let Some(provider) = rerank_provider {
            rerank_config.provider = provider;
        }
        // Ensure top_k respects the user's requested limit
        rerank_config.top_k = limit;

        eprintln!(
            "Reranking {} candidates via {} (top {})...",
            results.len(),
            rerank_config.provider,
            rerank_config.top_k,
        );
        results = reranker::rerank(query, results, &rerank_config).await;
    } else if !do_rerank {
        results.truncate(limit);
    }

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

async fn run_index(config: &Config, path: Option<String>, incremental: bool, file: Option<String>) -> Result<()> {
    let store = store::Store::open()?;

    if let Some(file_path) = file {
        println!("Indexing single file: {}", file_path);
        store.index_file(&file_path)?;
        println!("Done.");
        return Ok(());
    }

    let index_paths = if let Some(p) = path {
        vec![config::expand_home(&p)]
    } else {
        config.index_paths()
    };

    for index_path in &index_paths {
        if incremental {
            println!("Incremental indexing: {}", index_path);
            store.index_incremental(index_path)?;
        } else {
            println!("Full indexing: {}", index_path);
            store.index_full(index_path)?;
        }
    }

    let stats = store.get_stats()?;
    println!("Indexed {} files, {} chunks", stats.file_count, stats.chunk_count);

    Ok(())
}

async fn run_status(json: bool) -> Result<()> {
    let store = store::Store::open()?;
    let config = Config::load()?;
    let stats = store.get_stats()?;
    let (embedded, _) = store.get_embedding_stats()?;

    if json {
        let output = serde_json::json!({
            "file_count": stats.file_count,
            "chunk_count": stats.chunk_count,
            "embedded_count": embedded,
            "last_indexed": stats.last_indexed,
            "database_path": store.path(),
            "config_path": Config::config_path(),
            "watch_paths": config.watch_paths(),
            "index_paths": config.index_paths(),
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("Recall Status");
        println!("=============");
        println!("Database: {}", store.path());
        println!("Config: {}", Config::config_path().display());
        println!();
        println!("Index paths: {:?}", config.index_paths());
        println!("Watch paths: {:?}", config.watch_paths());
        println!();
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

fn run_watch(config: &Config) -> Result<()> {
    println!("Recall File Watcher");
    println!("===================");
    watcher::watch_directories(config)
}

fn run_config(config: &Config, action: ConfigAction) -> Result<()> {
    match action {
        ConfigAction::Show => {
            println!("Configuration (from {:?}):", Config::config_path());
            println!();
            println!("[index]");
            println!("paths = {:?}", config.index.paths);
            println!("exclude = {:?}", config.index.exclude);
            println!();
            println!("[embeddings]");
            println!("ollama_url = \"{}\"", config.embeddings.ollama_url);
            println!("model = \"{}\"", config.embeddings.model);
            println!();
            println!("[search]");
            println!("default_limit = {}", config.search.default_limit);
            println!("rrf_k = {}", config.search.rrf_k);
            println!();
            println!("[watch]");
            println!("paths = {:?}", config.watch.paths);
            println!("exclude = {:?}", config.watch.exclude);
            println!("debounce_ms = {}", config.watch.debounce_ms);
        }
        ConfigAction::Path => {
            println!("{}", Config::config_path().display());
        }
    }

    Ok(())
}

async fn run_embed(config: &Config, incremental: bool, limit: Option<usize>) -> Result<()> {
    let store = store::Store::open()?;
    let embedder = embedder::Embedder::new_with_config(config);

    // Check Ollama connectivity
    println!("Checking Ollama connectivity at {}...", config.embeddings.ollama_url);
    if !embedder.health_check().await? {
        anyhow::bail!("Cannot connect to Ollama at {}. Is it running?", config.embeddings.ollama_url);
    }
    println!("Ollama is available.");

    // Ensure model is pulled
    embedder.ensure_model().await?;

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

    println!("Generating embeddings for {} chunks using {}...\n", total, config.embeddings.model);

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
