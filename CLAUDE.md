# CLAUDE.md

This file provides guidance to Claude Code when working with the memory-search project.

## Overview

`memory-search` is a semantic memory search CLI that indexes markdown files (primarily Obsidian vaults) and provides token-efficient retrieval for LLM consumption.

## Architecture

- **SQLite + FTS5** for BM25 keyword search
- **sqlite-vec** (future) for vector embeddings
- **Hybrid search** combining BM25 + vector with weighted merging
- **Token-efficient output** - returns small bundles with citations, not full documents

## Project Structure

```
src/
├── main.rs      # CLI entry point (clap)
├── store.rs     # SQLite wrapper + schema
├── chunker.rs   # Markdown-aware chunking
├── embedder.rs  # Ollama HTTP client
├── search.rs    # Hybrid BM25 + vector search
└── watcher.rs   # File watching with debounce
```

## Commands

```bash
# Development
nix develop              # Enter dev shell
cargo build              # Build
cargo run -- status      # Run status command
cargo run -- index       # Index files
cargo run -- "query"     # Search

# Testing
cargo test
cargo clippy
```

## Key Design Decisions

1. **Standalone project** - Not part of aria workspace, has its own flake.nix
2. **Local SQLite** - Database at `~/.local/share/memory-search/memory.sqlite`
3. **Remote embeddings** - HTTP to Ollama at nightman.tap:11434
4. **Token-efficient output** - Compact format with citations by default

## Configuration

Config file: `~/.config/memory-search/config.toml`

```toml
[index]
paths = ["~/Obsidian/"]
exclude = ["**/Templates/**", "**/.obsidian/**"]

[embeddings]
ollama_url = "http://nightman.tap:11434"
model = "nomic-embed-text"

[search]
default_limit = 5
vector_weight = 0.7
bm25_weight = 0.3
```

## References

- Design doc: `~/Obsidian/ARIA/Improvements/memory-search-design.md`
- Clawdbot memory: https://docs.clawd.bot/experiments/research/memory
- sqlite-vec: https://github.com/asg017/sqlite-vec
