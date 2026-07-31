//! MCP (Model Context Protocol) server over stdio.
//!
//! Exposes recall's search, index, and status capabilities as MCP tools so
//! Claude Code (and other MCP clients) can use them.
//!
//! Two things here are deliberate and easy to undo by accident:
//!
//! 1. **The tool surface is narrower than the CLI's.** `trace` is a
//!    diagnostic, and retrieval strategy is not a decision an agent can make
//!    well — every parameter an agent cannot decide well is a parameter it
//!    will decide badly. The tool schema carries only what the *question*
//!    determines: text, date bounds, collection, effort.
//! 2. **Every payload goes out on both channels.** The Claude Code CLI
//!    forwards only `structuredContent` to the model; other MCP clients
//!    forward only `content[].text`. Emitting one and not the other silently
//!    blanks the tool for half the ecosystem, so [`ToolOutput`] always carries
//!    both renderings of the same data.

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, error, info};

use crate::search;
use crate::store::Store;

/// The retrieval contract, stated to the model verbatim in the server
/// instructions and repeated at the head of every result set. Search returns
/// what was *written down*, which is not the same thing as what is true now.
const RETRIEVAL_CONTRACT: &str = "Results are dated digests, not live state; \
when two results conflict, the newer date wins; volatile facts — versions, \
ports, hostnames, current status — must be verified live.";

/// A tool result rendered for both MCP channels. `text` is the human/plain
/// client rendering, `structured` the machine one; they describe the same
/// facts so neither client class is second-class.
struct ToolOutput {
    text: String,
    structured: Option<Value>,
}

impl ToolOutput {
    fn text_only(text: String) -> Self {
        Self {
            text,
            structured: None,
        }
    }
}

/// Run the MCP server on stdio (JSON-RPC over newline-delimited JSON).
pub async fn serve_mcp() -> Result<()> {
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
            "tools/call" => handle_tools_call(&id, &request).await,
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
            },
            "instructions": build_instructions(),
        }
    })
}

/// Build the server instructions from live index state.
///
/// MCP clients inject this into the model's system prompt at handshake time,
/// so it is the one place recall can tell the model what exists *before* the
/// first tool call — how much is indexed, which collections it can scope to,
/// and, crucially, which capabilities are missing. A BM25-only deployment
/// announcing its own degraded mode here saves the model from expecting
/// semantic recall it will not get.
fn build_instructions() -> String {
    let mut lines: Vec<String> = Vec::new();

    let state = Store::open().and_then(|store| {
        let stats = store.get_stats()?;
        let (embedded, total) = store.get_embedding_stats()?;
        let names: Vec<String> = store
            .list_collections()?
            .into_iter()
            .map(|c| c.name)
            .collect();
        Ok((stats, embedded, total, names))
    });

    match state {
        Ok((stats, embedded, total, names)) => {
            lines.push(format!(
                "recall is a local search index over your markdown notes: {} files, {} chunks.",
                stats.file_count, stats.chunk_count
            ));
            if let Some(last) = &stats.last_indexed {
                lines.push(format!("Last indexed: {last}."));
            }

            if !names.is_empty() {
                lines.push(String::new());
                lines.push(format!(
                    "Collections (scope with the `collection` parameter): {}",
                    names.join(", ")
                ));
            }

            // Capability gaps. Say them plainly — a model that knows the index
            // is lexical-only will phrase queries with the words it expects to
            // find instead of paraphrasing the concept.
            lines.push(String::new());
            if stats.chunk_count == 0 {
                lines.push("The index is empty. Run `recall index` before searching.".to_string());
            } else if embedded == 0 {
                lines.push(
                    "No vector embeddings; BM25-only. Run `recall embed` to enable hybrid search."
                        .to_string(),
                );
            } else if embedded < total {
                lines.push(format!(
                    "{} of {total} chunks have no embedding and are reachable by keyword search \
                     only. Run `recall embed` to complete coverage.",
                    total - embedded
                ));
            } else {
                lines.push(
                    "Hybrid search is available: BM25 and vector candidates are fused per query."
                        .to_string(),
                );
            }
        }
        Err(e) => {
            // The handshake must still succeed; a broken database is a search
            // failure, not a protocol failure.
            lines.push("recall is a local search index over your markdown notes.".to_string());
            lines.push(String::new());
            lines.push(format!("The index is currently unreadable ({e}). `recall_search` will fail until this is fixed; try `recall_status`."));
        }
    }

    lines.push(String::new());
    lines.push(format!("Retrieval contract: {RETRIEVAL_CONTRACT}"));

    lines.push(String::new());
    lines.push("Reading results:".to_string());
    lines.push(
        "  - `path` and `line` (absolute, 1-indexed) locate the chunk. For surrounding context, \
         call Read(path, offset=line-20, limit=80)."
            .to_string(),
    );
    lines.push(
        "  - `date_source` says where `date` came from: \"frontmatter\" (the author declared it), \
         \"filename\" (a dated note), \"mtime\" (the file was merely touched — weak evidence of \
         when the content was written)."
            .to_string(),
    );
    lines.push(
        "  - `status` is reported, never filtered on. A note marked \"draft\" or \"archived\" is \
         still returned; weigh it yourself."
            .to_string(),
    );
    lines.push(
        "  - Ranking already accounts for recency, so do not re-sort by date; use `after` / \
         `before` when the question itself is bounded in time."
            .to_string(),
    );

    lines.join("\n")
}

/// Long-form description for `recall_search`. The description is the only
/// prompt-engineering surface a tool has: the model reads it every turn and
/// nothing else explains when to reach for search, how to phrase the query,
/// or what to do with a hit. Worked examples cost tokens once and prevent
/// malformed calls forever.
const SEARCH_DESCRIPTION: &str = r#"Search the user's indexed markdown notes (Obsidian vault, memory files, project docs) and return ranked excerpts.

Reach for this whenever the answer might already be written down: past decisions and their reasons, meeting and session notes, project state, personal preferences, names and identifiers you were told once, or anything the user refers to as "my notes", "the vault", "what did I say about…".

## What comes back

Each result is one chunk of one file:

- `path` — absolute file path.
- `line` — absolute, 1-indexed line where the chunk starts. To read around a hit: Read(path, offset=line-20, limit=80).
- `date` + `date_source` — when the chunk is from, and how that was determined ("frontmatter" = author-declared, "filename" = dated note, "mtime" = filesystem timestamp only). `null` when unknown; a date is never invented.
- `status`, `collection`, `section` — metadata, reported and never used to exclude results.
- `score` — relevance after fusion, optional reranking, and recency decay. Comparable within one result set, not across searches.

Results are dated digests, not live state; when two results conflict, the newer date wins; volatile facts — versions, ports, hostnames, current status — must be verified live.

## Phrasing the query

Use the words that would appear in the note, plus the concept. Retrieval fuses keyword and semantic matching, so a short natural-language phrase beats a single generic keyword, and both beat a full sentence of filler.

- Good: `postgres connection pool timeout decision`
- Good: `why did we drop the discord gateway`
- Weak: `database` (matches everything)
- Weak: `Can you please tell me about the database work I did?` (filler words dilute the match)

## Examples

Recall a decision and its reasoning:
```json
{"query": "chose sqlite over postgres reasoning", "limit": 5}
```

Bound a question in time (dates are ISO, YYYY-MM-DD):
```json
{"query": "sprint planning notes", "after": "2026-01-01", "before": "2026-03-31"}
```

Scope to one collection and spend an LLM call on precision:
```json
{"query": "aria orchestrator prompt design", "collection": "vault", "rerank": true, "limit": 8}
```

## Notes

- `rerank` re-scores candidates with an LLM. It costs seconds; use it when precision matters more than latency, not by default. Nothing else turns it on.
- Retrieval fuses keyword and vector candidates automatically. There is no strategy knob and nothing to tune per call.
- Zero results usually means wrong vocabulary, not absent knowledge. Retry once with different terms before concluding the note does not exist."#;

fn search_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "result_count": {"type": "integer"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute path to the source file"},
                        "line": {"type": "integer", "description": "Absolute 1-indexed start line of the chunk"},
                        "date": {"type": ["string", "null"], "description": "ISO date, YYYY-MM-DD, or null if unknown"},
                        "date_source": {
                            "type": ["string", "null"],
                            "enum": ["frontmatter", "filename", "mtime", null],
                            "description": "How `date` was determined"
                        },
                        "status": {"type": ["string", "null"]},
                        "collection": {"type": ["string", "null"]},
                        "section": {"type": ["string", "null"]},
                        "score": {"type": "number"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "line", "date", "date_source", "status",
                                 "collection", "score", "content"]
                }
            }
        },
        "required": ["query", "result_count", "results"]
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
                    "description": SEARCH_DESCRIPTION,
                    "annotations": {"readOnlyHint": true, "openWorldHint": false},
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to look for. Keywords plus concept; see the examples in this tool's description."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 5)",
                                "default": 5
                            },
                            "rerank": {
                                "type": "boolean",
                                "description": "Re-score candidates with an LLM for precision. Costs seconds per call (default: false).",
                                "default": false
                            },
                            "after": {
                                "type": "string",
                                "description": "Only return chunks dated on or after this day. ISO date, YYYY-MM-DD."
                            },
                            "before": {
                                "type": "string",
                                "description": "Only return chunks dated on or before this day. ISO date, YYYY-MM-DD."
                            },
                            "collection": {
                                "type": "string",
                                "description": "Restrict to a single collection by name (default: all collections). Names are listed in the server instructions and by recall_status."
                            }
                        },
                        "required": ["query"]
                    },
                    "outputSchema": search_output_schema()
                },
                {
                    "name": "recall_index",
                    "description": "Reconcile the index with the filesystem: re-read changed files and forget deleted ones. With no args, does this for every collection at its root_path. Only needed when files changed and the watcher is not running.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "collection": {
                                "type": "string",
                                "description": "Index just this collection (by name)"
                            },
                            "path": {
                                "type": "string",
                                "description": "Override the collection's root_path (requires 'collection')"
                            }
                        }
                    }
                },
                {
                    "name": "recall_status",
                    "description": "Report index health: file count, chunk count, embedding coverage, last indexed time, and the registered collections with their root paths. Use it to check whether vector search is available and whether the index is stale.",
                    "annotations": {"readOnlyHint": true, "openWorldHint": false},
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }
    })
}

async fn handle_tools_call(id: &Option<Value>, request: &Value) -> Value {
    let tool_name = request["params"]["name"].as_str().unwrap_or("");
    let arguments = &request["params"]["arguments"];

    let result = match tool_name {
        "recall_search" => tool_search(arguments).await,
        "recall_index" => tool_index(arguments).await,
        "recall_status" => tool_status().await,
        _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_name)),
    };

    match result {
        Ok(output) => {
            let mut result_obj = json!({
                "content": [{"type": "text", "text": output.text}]
            });
            if let Some(structured) = output.structured {
                result_obj["structuredContent"] = structured;
            }
            json!({"jsonrpc": "2.0", "id": id, "result": result_obj})
        }
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

async fn tool_search(args: &Value) -> Result<ToolOutput> {
    let query = args["query"]
        .as_str()
        .context("recall_search requires a 'query' string parameter")?;
    let limit = args["limit"].as_u64().unwrap_or(5) as usize;
    let rerank = args["rerank"].as_bool().unwrap_or(false);
    let after = args["after"].as_str().map(|s| s.to_string());
    let before = args["before"].as_str().map(|s| s.to_string());
    let collection = args["collection"].as_str().map(|s| s.to_string());

    let outcome = search::search(search::SearchRequest {
        query: query.to_string(),
        limit,
        collection,
        after,
        before,
        rerank,
    })
    .await?;

    let results: Vec<Value> = outcome
        .results
        .iter()
        .map(|(r, _)| {
            json!({
                "path": r.file_path,
                "line": r.start_line,
                "date": r.date,
                "date_source": r.date_source,
                "status": r.status,
                "collection": r.collection_name,
                "section": r.section,
                "score": r.score,
                "content": r.content,
            })
        })
        .collect();

    let result_count = results.len();
    let structured = json!({
        "query": outcome.query,
        "result_count": result_count,
        "results": results,
    });

    // The text channel is the same payload, pretty-printed. It used to be a
    // hand-written prose renderer of the very same ten fields, which also
    // restated the retrieval contract and the `Read(path, offset=line-20)`
    // hint — both of which already reach the model through the server
    // instructions and this tool's description. Three copies of two sentences,
    // and a second formatter to keep in sync with the first.
    let text = if result_count == 0 {
        format!(
            "No results for \"{}\".\nTry different vocabulary — the words that would appear in \
             the note itself — before concluding it does not exist.",
            outcome.query
        )
    } else {
        serde_json::to_string_pretty(&structured)?
    };

    Ok(ToolOutput {
        text,
        structured: Some(structured),
    })
}

async fn tool_index(args: &Value) -> Result<ToolOutput> {
    let store = Store::open()?;
    let collection = args["collection"].as_str();
    let path_arg = args["path"].as_str();

    let targets = if let Some(name) = collection {
        let c = store
            .get_collection(name)?
            .ok_or_else(|| anyhow::anyhow!("Collection {:?} not found", name))?;
        vec![c]
    } else if path_arg.is_some() {
        anyhow::bail!("'path' requires 'collection' to specify which collection to index into.");
    } else {
        let cs = store.list_collections()?;
        if cs.is_empty() {
            anyhow::bail!(
                "No collections registered. Add one with `recall collection add` and re-run."
            );
        }
        cs
    };

    let mut count = 0usize;
    for target in &targets {
        let dir = match path_arg {
            Some(p) => crate::expand_home(p),
            None => target.root_path.clone(),
        };
        if dir.is_empty() {
            continue;
        }
        store.index(target.id, &dir)?;
        count += 1;
    }

    let stats = store.get_stats()?;
    Ok(ToolOutput::text_only(format!(
        "Indexed {} files, {} chunks across {} collection(s)",
        stats.file_count, stats.chunk_count, count
    )))
}

async fn tool_status() -> Result<ToolOutput> {
    let store = Store::open()?;
    let stats = store.get_stats()?;
    let (embedded, total) = store.get_embedding_stats()?;
    let collections = store.list_collections()?;

    let coverage = if total > 0 {
        (embedded as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    let structured = json!({
        "files": stats.file_count,
        "chunks": stats.chunk_count,
        "embeddings": embedded,
        "embedding_coverage": format!("{:.1}%", coverage),
        "vector_search_available": embedded > 0,
        "last_indexed": stats.last_indexed,
        "database": store.path(),
        "collections": collections.iter().map(|c| json!({
            "name": c.name,
            "root_path": c.root_path,
            "description": c.description,
            "half_life_days": c.half_life_days,
        })).collect::<Vec<_>>(),
    });

    let mut text = format!(
        "recall index: {} files, {} chunks, {} embeddings ({:.1}% coverage).\n\
         Vector search: {}.\nLast indexed: {}.\nDatabase: {}\n",
        stats.file_count,
        stats.chunk_count,
        embedded,
        coverage,
        if embedded > 0 {
            "available"
        } else {
            "unavailable (BM25-only; run `recall embed`)"
        },
        stats.last_indexed.as_deref().unwrap_or("never"),
        store.path(),
    );
    if collections.is_empty() {
        text.push_str("Collections: none registered.\n");
    } else {
        text.push_str("Collections:\n");
        for c in &collections {
            text.push_str(&format!("  - {}: {}\n", c.name, c.root_path));
        }
    }

    Ok(ToolOutput {
        text,
        structured: Some(structured),
    })
}
