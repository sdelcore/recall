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
# Paths to index for semantic search
paths = ["~/Obsidian"]
# Glob patterns to exclude from indexing
exclude = ["**/Templates/**", "**/.obsidian/**", "**/attachments/**"]

[embeddings]
# Ollama server URL for generating embeddings
ollama_url = "http://nightman.tap:11434"
# Embedding model name
model = "nomic-embed-text"

[search]
# Default number of results to return
default_limit = 5
# Vector weight for hybrid search (0.0-1.0)
vector_weight = 0.7
# BM25 weight for hybrid search (0.0-1.0)
bm25_weight = 0.3

[watch]
# Paths to watch for file changes (auto-index on change)
paths = ["~/Obsidian"]
# Patterns to exclude from watching (substring match)
exclude = ["Templates/", ".obsidian/", "attachments/", ".sync-conflict-"]
# Debounce time in milliseconds before indexing changed files
debounce_ms = 1500
```

Config commands:
```bash
memory-search config show   # Display current config
memory-search config path   # Show config file location
memory-search status        # Status including paths
```

## References

- Design doc: `~/Obsidian/ARIA/Improvements/memory-search-design.md`
- Clawdbot memory: https://docs.clawd.bot/experiments/research/memory
- sqlite-vec: https://github.com/asg017/sqlite-vec
