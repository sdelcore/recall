//! MCP (Model Context Protocol) server over stdio.
//!
//! Exposes recall's search, index, and status capabilities as MCP tools
//! so Claude Code (and other MCP clients) can use them.

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, error, info};

use crate::config::Config;
use crate::embedder::Embedder;
use crate::reranker;
use crate::store::{SearchOptions, Store};

/// Run the MCP server on stdio (JSON-RPC over newline-delimited JSON).
pub async fn serve_mcp(config: &Config) -> Result<()> {
    info!("Starting Recall MCP server (stdio)");

    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        debug!("MCP request: {}", &line[..line.len().min(200)]);

        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let err_resp = json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {"code": -32700, "message": format!("Parse error: {}", e)}
                });
                write_response(&mut stdout, &err_resp).await?;
                continue;
            }
        };

        let id = request.get("id").cloned();
        let method = request["method"].as_str().unwrap_or("");

        let response = match method {
            "initialize" => handle_initialize(&id),
            "tools/list" => handle_tools_list(&id),
            "tools/call" => handle_tools_call(&id, &request, config).await,
            "notifications/initialized" | "notifications/cancelled" => {
                // Notifications don't get responses
                continue;
            }
            _ => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": format!("Method not found: {}", method)}
            }),
        };

        write_response(&mut stdout, &response).await?;
    }

    info!("MCP server shutting down (stdin closed)");
    Ok(())
}

async fn write_response(stdout: &mut tokio::io::Stdout, response: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(response)?;
    stdout.write_all(&bytes).await?;
    stdout.write_all(b"\n").await?;
    stdout.flush().await?;
    Ok(())
}

fn handle_initialize(id: &Option<Value>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "recall",
                "version": env!("CARGO_PKG_VERSION")
            }
        }
    })
}

fn handle_tools_list(id: &Option<Value>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "tools": [
                {
                    "name": "recall_search",
                    "description": "Search the Obsidian vault and memory files using hybrid BM25 + vector search with optional LLM reranking. Returns ranked results with file paths, line numbers, and content snippets.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 5)",
                                "default": 5
                            },
                            "hybrid": {
                                "type": "boolean",
                                "description": "Use hybrid BM25 + vector search (default: true)",
                                "default": true
                            },
                            "rerank": {
                                "type": "boolean",
                                "description": "Rerank results using LLM for better relevance (default: false)",
                                "default": false
                            },
                            "after": {
                                "type": "string",
                                "description": "Only include results after this date (YYYY-MM-DD)"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "recall_index",
                    "description": "Trigger incremental re-indexing of the Obsidian vault. Only re-indexes files that have changed since last index.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Specific path to index (defaults to configured index paths)"
                            }
                        }
                    }
                },
                {
                    "name": "recall_status",
                    "description": "Get index health and statistics: file count, chunk count, embedding coverage, last indexed time.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }
    })
}

async fn handle_tools_call(id: &Option<Value>, request: &Value, config: &Config) -> Value {
    let tool_name = request["params"]["name"].as_str().unwrap_or("");
    let arguments = &request["params"]["arguments"];

    let result = match tool_name {
        "recall_search" => tool_search(arguments, config).await,
        "recall_index" => tool_index(arguments, config).await,
        "recall_status" => tool_status().await,
        _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_name)),
    };

    match result {
        Ok(content) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{"type": "text", "text": content}]
            }
        }),
        Err(e) => {
            error!("Tool {} failed: {}", tool_name, e);
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{"type": "text", "text": format!("Error: {}", e)}],
                    "isError": true
                }
            })
        }
    }
}

async fn tool_search(args: &Value, config: &Config) -> Result<String> {
    let query = args["query"]
        .as_str()
        .context("recall_search requires a 'query' string parameter")?;
    let limit = args["limit"].as_u64().unwrap_or(5) as usize;
    let hybrid = args["hybrid"].as_bool().unwrap_or(true);
    let do_rerank = args["rerank"].as_bool().unwrap_or(false);
    let after = args["after"].as_str().map(|s| s.to_string());

    let store = Store::open()?;

    let fetch_limit = if do_rerank {
        config.reranking.candidates.max(limit)
    } else {
        limit
    };

    let options = SearchOptions {
        after,
        project: None,
        file_pattern: None,
    };

    let mut results = if hybrid {
        let embedder = Embedder::new_with_config(config);
        let (embedded, _) = store.get_embedding_stats()?;
        if embedded == 0 {
            store.search_fts_filtered(query, fetch_limit, &options)?
        } else {
            let query_embedding = embedder.embed(query).await?;
            store.search_hybrid(query, &query_embedding, fetch_limit, config.search.rrf_k)?
        }
    } else {
        store.search_fts_filtered(query, fetch_limit, &options)?
    };

    if do_rerank && !results.is_empty() {
        let mut rerank_config = config.reranking.clone();
        rerank_config.top_k = limit;
        results = reranker::rerank(query, results, &rerank_config).await;
    } else {
        results.truncate(limit);
    }

    let output = json!({
        "query": query,
        "result_count": results.len(),
        "results": results.iter().map(|r| {
            json!({
                "file": r.file_path,
                "lines": format!("{}-{}", r.start_line, r.end_line),
                "score": format!("{:.2}", r.score),
                "section": r.section,
                "date": r.date,
                "content": r.content,
            })
        }).collect::<Vec<_>>()
    });

    serde_json::to_string_pretty(&output).context("Failed to serialize search results")
}

async fn tool_index(args: &Value, config: &Config) -> Result<String> {
    let store = Store::open()?;

    let paths = if let Some(path) = args["path"].as_str() {
        vec![crate::config::expand_home(path)]
    } else {
        config.index_paths()
    };

    for path in &paths {
        store.index_incremental(path)?;
    }

    let stats = store.get_stats()?;
    Ok(format!(
        "Indexed {} files, {} chunks across {} paths",
        stats.file_count,
        stats.chunk_count,
        paths.len()
    ))
}

async fn tool_status() -> Result<String> {
    let store = Store::open()?;
    let stats = store.get_stats()?;
    let (embedded, total) = store.get_embedding_stats()?;

    let output = json!({
        "files": stats.file_count,
        "chunks": stats.chunk_count,
        "embeddings": embedded,
        "embedding_coverage": if total > 0 {
            format!("{:.1}%", (embedded as f64 / total as f64) * 100.0)
        } else {
            "0%".to_string()
        },
        "last_indexed": stats.last_indexed,
        "database": store.path(),
    });

    serde_json::to_string_pretty(&output).context("Failed to serialize status")
}
