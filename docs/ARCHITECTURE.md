# recall Architecture

Technical architecture documentation for the semantic search CLI.

## Overview

recall indexes markdown files and provides hybrid search combining:
- **BM25** keyword search via SQLite FTS5
- **Vector embeddings** via Ollama for semantic search

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           recall CLI                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                         Commands                                   │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │ │
│  │  │ search  │  │  index  │  │  embed  │  │  watch  │  │ status  │ │ │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘ │ │
│  └───────┼────────────┼───────────┼───────────┼───────────┼────────┘ │
│          │            │           │           │           │           │
│          └────────────┴───────────┼───────────┴───────────┘           │
│                                   ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                          Store                                   │  │
│  │                 (SQLite + FTS5 + sqlite-vec)                    │  │
│  │                                                                  │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │  │
│  │  │      files      │  │   fts_chunks    │  │ vec_embeddings  │  │  │
│  │  │  - file_path    │  │  (FTS5 virtual  │  │  (vec0 virtual  │  │  │
│  │  │  - mtime        │  │   table)        │  │   table,        │  │  │
│  │  │  - indexed_at   │  │                 │  │   float[768])   │  │  │
│  │  │  - chunk_count  │  │                 │  │                 │  │  │
│  │  └────────┬────────┘  └─────────────────┘  └─────────────────┘  │  │
│  │           │                                                      │  │
│  │  ┌────────┴────────┐  ┌─────────────────┐                       │  │
│  │  │     chunks      │  │     config      │                       │  │
│  │  │  - file_id (FK) │  │  - key          │                       │  │
│  │  │  - chunk_index  │  │  - value        │                       │  │
│  │  │  - date         │  │                 │                       │  │
│  │  │  - section      │  └─────────────────┘                       │  │
│  │  │  - project      │                                             │  │
│  │  │  - start_line   │                                             │  │
│  │  │  - end_line     │                                             │  │
│  │  │  - content      │                                             │  │
│  │  └─────────────────┘                                             │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                   │                                    │
│                    ┌──────────────┴──────────────┐                    │
│                    ▼                             ▼                    │
│  ┌─────────────────────────┐     ┌─────────────────────────┐         │
│  │        Config           │     │       Embedder          │         │
│  │   ~/.config/recall/     │     │   (Ollama client)       │         │
│  │   config.toml           │     │                         │         │
│  └─────────────────────────┘     └─────────────────────────┘         │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Database Schema

```sql
-- Metadata about indexed files
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    mtime INTEGER NOT NULL,
    indexed_at INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0
);

-- Text chunks with metadata
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    date TEXT,
    section TEXT,
    project TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);

-- FTS5 for BM25 search (content-sync'd with chunks table)
CREATE VIRTUAL TABLE fts_chunks USING fts5(
    content,
    content=chunks,
    content_rowid=id
);

-- Vector embeddings via sqlite-vec (768-dim float vectors)
CREATE VIRTUAL TABLE vec_embeddings USING vec0(
    embedding float[768]
);

-- Key-value configuration and state
CREATE TABLE config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

## Search Pipeline

### BM25 Search (Default)

```
Query → FTS5 MATCH → BM25 Ranking → Top K Results
```

### Hybrid Search

```
                    ┌─────────────────────────────┐
                    │          Query              │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
           ┌───────────────┐             ┌───────────────┐
           │  BM25 Search  │             │ Vector Search │
           │    (FTS5)     │             │ (sqlite-vec)  │
           └───────┬───────┘             └───────┬───────┘
                   │                             │
                   │  Ranked chunk ID lists      │
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │  Reciprocal Rank Fusion     │
                    │  score = Σ 1/(k + rank)     │
                    │  k = rrf_k (default 60)     │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      Top K Results          │
                    │   (sorted by RRF score)     │
                    └─────────────────────────────┘
```

## Chunking Strategy

Markdown files are split into chunks in `store.rs`:

```
Document
    │
    ▼
┌─────────────────────────────────────────┐
│           chunk_markdown()              │
│                                         │
│  1. Split on ## headers (sections)     │
│  2. Enforce max chunk size (~400 tok)  │
│  3. Overlap ~240 chars at size splits  │
│  4. Extract date from filename         │
│  5. Track section names                │
│                                         │
└────────────────┬────────────────────────┘
                 │
                 ▼
    ┌────────┬────────┬────────┐
    │ Chunk 1│ Chunk 2│ Chunk 3│ ...
    └────────┴────────┴────────┘
```

## File Watcher

The watch command monitors for file changes:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Watcher                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  notify-rs (inotify/kqueue/etc.)                               │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Event Filter    │  Skip: .obsidian/, Templates/,            │
│  │                 │        .sync-conflict-, etc.              │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   Debouncer     │  Wait 1500ms after last event            │
│  │                 │  (configurable debounce_ms)               │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  Index Update   │  Reindex changed file                    │
│  │                 │  Update embeddings if enabled             │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Embedder (Ollama)

```
┌─────────────────────────────────────────┐
│              Embedder                    │
├─────────────────────────────────────────┤
│                                         │
│  HTTP Client                            │
│       │                                 │
│       ▼                                 │
│  POST http://localhost:11434/api/embed  │
│  {                                      │
│    "model": "nomic-embed-text",        │
│    "input": "chunk text..."            │
│  }                                      │
│       │                                 │
│       ▼                                 │
│  Response: [0.123, -0.456, ...]        │
│  (768-dimensional vector)               │
│                                         │
└─────────────────────────────────────────┘
```

## Configuration

```toml
[index]
paths = ["~/Obsidian"]
exclude = ["**/Templates/**", "**/.obsidian/**"]

[embeddings]
ollama_url = "http://localhost:11434"
model = "nomic-embed-text"

[search]
default_limit = 5
rrf_k = 60

[watch]
paths = ["~/Obsidian"]
exclude = ["Templates/", ".obsidian/"]
debounce_ms = 1500
```
