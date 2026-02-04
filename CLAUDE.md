# CLAUDE.md

Project-level guidance for the recall semantic memory search CLI.

## Build Commands

```bash
# Enter nix development environment
cd aria/recall && nix develop

# Build
cargo build

# Run tests
cargo test

# Lint
cargo clippy

# Format
cargo fmt
```

## Project Structure

```
recall/
├── src/
│   ├── main.rs       # CLI entry point (clap), command dispatch
│   ├── store.rs      # SQLite + FTS5 + sqlite-vec database, chunking logic
│   ├── config.rs     # TOML configuration loading
│   ├── embedder.rs   # Ollama HTTP client for embeddings
│   └── watcher.rs    # File system watcher (notify-rs)
├── docs/
│   ├── ARCHITECTURE.md  # Technical architecture and schema
│   └── DEVELOPMENT.md   # Development guide
├── Cargo.toml
├── flake.nix
└── README.md
```

## Architecture

- **Store** (`store.rs`): SQLite database with FTS5 for BM25 search and sqlite-vec for vector KNN search. Also contains the `chunk_markdown()` chunking logic.
- **Config** (`config.rs`): TOML config from `~/.config/recall/config.toml`
- **Embedder** (`embedder.rs`): HTTP client for Ollama embedding API (nomic-embed-text, 768-dim)
- **Watcher** (`watcher.rs`): File system watcher with debouncing for auto-indexing

## Database

Tables: `files`, `chunks`, `fts_chunks` (FTS5), `vec_embeddings` (vec0), `config`

Location: `~/.local/share/recall/memory.sqlite`

## Key Patterns

- Hybrid search uses **Reciprocal Rank Fusion (RRF)** with configurable `rrf_k` parameter
- Chunking splits on `##` headers with ~240 char overlap at size boundaries
- FTS5 is kept in sync via triggers on the `chunks` table
