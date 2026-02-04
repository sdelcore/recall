# recall

A semantic memory search CLI that indexes markdown files (primarily Obsidian vaults) and provides token-efficient retrieval for LLM consumption.

## Features

- **SQLite + FTS5** - BM25 keyword search with full-text indexing
- **Vector Embeddings** - Optional semantic search via Ollama
- **Hybrid Search** - Combine BM25 + vector with Reciprocal Rank Fusion
- **File Watching** - Auto-index on file changes
- **Token-Efficient Output** - Compact results with citations
- **Configurable** - Paths, exclusions, and weights via TOML config

## Quick Start

```bash
# Enter development environment
nix develop

# Build
cargo build

# Index your vault
recall index

# Search
recall "what are the user's preferences"

# Check status
recall status
```

## CLI Commands

### Search

```bash
# Basic search (BM25)
recall "query"
recall "project deadlines" --limit 10

# Hybrid search (BM25 + vector)
recall "query" --hybrid

# Output formats
recall "query" --format compact   # Default
recall "query" --format json
recall "query" --format full

# Filters
recall "query" --after 2024-01-01
recall "query" --project "aria"
recall "query" --file "*.md"
```

### Indexing

```bash
recall index                  # Full index
recall index --incremental    # Changed files only
recall index --file path.md   # Single file
recall index --path ~/notes   # Custom path
```

### Embeddings

```bash
recall embed                  # Generate embeddings
recall embed --incremental    # Only missing
recall embed --limit 100      # Limit for testing
```

### File Watching

```bash
recall watch                  # Watch and auto-index
```

### Status & Config

```bash
recall status                 # Index statistics
recall status --json          # JSON output
recall config show            # Display config
recall config path            # Show config location
```

## Configuration

Config file: `~/.config/recall/config.toml`

```toml
[index]
# Paths to index
paths = ["~/Obsidian"]
# Patterns to exclude
exclude = ["**/Templates/**", "**/.obsidian/**", "**/attachments/**"]

[embeddings]
# Ollama server for embeddings
ollama_url = "http://localhost:11434"
model = "nomic-embed-text"

[search]
# Default results count
default_limit = 5
# RRF constant k for hybrid search (higher = more weight to lower-ranked results)
rrf_k = 60

[watch]
# Paths to watch for changes
paths = ["~/Obsidian"]
# Patterns to exclude from watching
exclude = ["Templates/", ".obsidian/", "attachments/", ".sync-conflict-"]
# Debounce time before indexing
debounce_ms = 1500
```

## Storage

| Location | Purpose |
|----------|---------|
| `~/.local/share/recall/memory.sqlite` | SQLite database |
| `~/.config/recall/config.toml` | Configuration |

## Running as Service

On AriaOS, recall runs as a systemd user service for auto-indexing:

```bash
# Status
systemctl --user status recall

# Logs
journalctl --user -u recall -f

# Restart
systemctl --user restart recall
```

## Integration with ARIA

recall provides semantic search capabilities for ARIA:

- **Context Retrieval** - Find relevant notes before answering questions
- **Memory Search** - Search MEMORY.md and past interactions
- **Cross-Document** - Find related information across vault

Example usage in prompts:
```
Before answering, search memory for relevant context:
- Use recall to find patterns about user preferences
- Search recent daily notes for in-progress work
```

## Architecture

```
                    ┌─────────────────┐
                    │   CLI (clap)    │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │   Index    │     │   Search   │     │   Watch    │
   │  (chunker) │     │(BM25+vec)  │     │  (notify)  │
   └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            ▼
                    ┌─────────────────┐
                    │     Store       │
                    │ (SQLite + FTS5) │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
             ┌──────────┐      ┌──────────┐
             │  Config  │      │ Embedder │
             │  (TOML)  │      │ (Ollama) │
             └──────────┘      └──────────┘
```

## Building

```bash
# With Nix
nix build

# With Cargo
cargo build --release
```
