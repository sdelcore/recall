# CLAUDE.md

Project-level guidance for the recall semantic memory search CLI.

## Build Commands

```bash
# Build (requires nix develop from aria for Rust toolchain)
cd recall && nix develop ~/aria/aria --command cargo build

# Run tests
nix develop ~/aria/aria --command cargo test

# Lint
nix develop ~/aria/aria --command cargo clippy

# Format
nix develop ~/aria/aria --command cargo fmt
```

## Project Structure

```
recall/
├── src/
│   ├── main.rs       # CLI entry point (clap), command dispatch
│   ├── store.rs      # SQLite + FTS5 + sqlite-vec database, chunking logic
│   ├── config.rs     # TOML configuration loading + reranking config
│   ├── embedder.rs   # Ollama HTTP client for embeddings
│   ├── reranker.rs   # LLM reranking (claude-code SDK, Anthropic API, Ollama)
│   ├── mcp.rs        # MCP stdio server (recall_search, recall_index, recall_status)
│   └── watcher.rs    # File system watcher (notify-rs)
├── Cargo.toml
└── README.md
```

## Architecture

- **Store** (`store.rs`): SQLite database with FTS5 for BM25 search and sqlite-vec for vector KNN search. Also contains `chunk_markdown()` chunking logic.
- **Config** (`config.rs`): TOML config from `~/.config/recall/config.toml`. Includes `[reranking]` section with per-provider settings.
- **Embedder** (`embedder.rs`): HTTP client for Ollama embedding API (nomic-embed-text, 768-dim)
- **Reranker** (`reranker.rs`): LLM-based reranking with 3 configurable providers:
  - `claude-code` (default): Uses `claude-agent-sdk` crate. No API key needed. Batches all candidates into one prompt.
  - `anthropic`: Direct Anthropic Messages API. Parallel calls, needs `ANTHROPIC_API_KEY`.
  - `ollama`: Local model fallback for offline use.
- **MCP** (`mcp.rs`): Model Context Protocol server over stdio. Exposes 3 tools: `recall_search`, `recall_index`, `recall_status`. Registered in ARIA's Claude Code MCP config.
- **Watcher** (`watcher.rs`): File system watcher with debouncing for auto-indexing

## Database

Tables: `files`, `chunks`, `fts_chunks` (FTS5), `vec_embeddings` (vec0), `config`

Location: `~/.local/share/recall/memory.sqlite`

## Key Patterns

- Hybrid search uses **Reciprocal Rank Fusion (RRF)** with configurable `rrf_k` parameter
- LLM reranking batches all candidates into one prompt for efficiency (1 LLM call per search)
- Chunking splits on `##` headers with ~240 char overlap at size boundaries
- FTS5 is kept in sync via triggers on the `chunks` table
- MCP server uses JSON-RPC over newline-delimited JSON on stdio
- All reranker error paths log diagnostics and fall back to RRF order (no silent failures)

## Search Pipeline

```
Query → [Query Expansion (optional)] → BM25 + Vector → RRF Fusion → [LLM Reranker] → Results
```

Use `--hybrid` for BM25+vector, `--rerank` for LLM reranking, or both.

## Configuration

Config at `~/.config/recall/config.toml`. Key sections:

```toml
[search]
default_limit = 5
rrf_k = 60

[reranking]
enabled = false              # or use --rerank flag
provider = "claude-code"     # "claude-code" | "anthropic" | "ollama"
candidates = 20
top_k = 5

[reranking.claude_code]
model = "haiku"
```

## MCP Server

Start: `recall serve --mode mcp`

Tools exposed:
- `recall_search(query, limit?, hybrid?, rerank?, after?)` — search the vault
- `recall_index(path?)` — trigger incremental re-indexing
- `recall_status()` — index health and stats

Registered in ARIA's Claude Code config at `~/.config/claude/settings.local.json`.
