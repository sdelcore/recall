# recall Development

Development guide for working on recall.

## Build Commands

```bash
# Enter nix development environment
cd recall && nix develop

# Build
cargo build

# Build release
cargo build --release

# Run tests
cargo test

# Format code
cargo fmt

# Lint
cargo clippy
```

## Project Structure

```
recall/
├── src/
│   ├── main.rs       # CLI entry point (clap)
│   ├── store.rs      # SQLite + FTS5 database
│   ├── config.rs     # TOML configuration
│   ├── embedder.rs   # Ollama HTTP client
│   ├── watcher.rs    # File system watcher
├── Cargo.toml
└── flake.nix
```

## Key Files

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI parsing, command dispatch |
| `src/store.rs` | All database operations (index, search, status) |
| `src/config.rs` | Config loading from TOML |
| `src/embedder.rs` | HTTP client for Ollama embeddings |
| `src/watcher.rs` | notify-rs based file watcher |

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `rusqlite` | SQLite database |
| `tokio` | Async runtime |
| `reqwest` | HTTP client (Ollama) |
| `notify` | File system events |
| `serde` | Serialization |
| `toml` | Config parsing |
| `clap` | CLI parsing |

## Database Location

```
~/.local/share/recall/memory.sqlite
```

To reset: delete this file and re-run `recall index`.

## Testing

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test store

# Run with output
cargo test -- --nocapture
```

### Manual Testing

```bash
# Index a test directory
recall index --path ./test-vault

# Search
recall "test query"

# Check status
recall status

# Test hybrid search (requires Ollama running)
recall "test query" --hybrid
```

## Debugging

### Check Index Status

```bash
recall status
recall status --json | jq
```

### View Database

```bash
sqlite3 ~/.local/share/recall/memory.sqlite

# List documents
SELECT file_path, chunk_count, indexed_at FROM files LIMIT 10;

# Check FTS index
SELECT * FROM fts_chunks WHERE fts_chunks MATCH 'query' LIMIT 5;

# Check embeddings
SELECT rowid FROM vec_embeddings LIMIT 10;
```

### Test Ollama Connection

```bash
curl http://localhost:11434/api/embed \
  -d '{"model": "nomic-embed-text", "input": "test"}'
```

### Watch Service Logs

On AriaOS:
```bash
journalctl --user -u recall -f
```

## Common Tasks

### Modify Search Ranking

Edit `src/store.rs` search functions:
- `search_bm25()` - FTS5 query building
- `search_hybrid()` - Score combination logic

### Add New Config Option

1. Add field to struct in `src/config.rs`
2. Add default in `Default` impl
3. Update TOML parsing
4. Use in relevant code
5. Update docs/ARCHITECTURE.md

### Modify Chunking Strategy

Edit `src/store.rs` (chunking logic lives here):
- `chunk_markdown()` - Main chunking function
- `MAX_CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS` - Size constants
- Section header detection and overlap at size boundaries

### Add New CLI Command

1. Add subcommand in `src/main.rs`
2. Implement handler
3. Wire up in main match
4. Update README.md

## Embedding Model Notes

Default model: `nomic-embed-text`
- 768 dimensions
- Efficient for short texts
- Good semantic understanding

To use a different model:
1. Pull model: `ollama pull <model>`
2. Update config: `model = "<model>"`
3. Re-run: `recall embed` (existing embeddings incompatible)

## Performance Tips

- Use `--incremental` for index/embed to avoid reprocessing
- Debounce_ms in config prevents rapid re-indexing
- FTS5 search is very fast; vector search adds latency
- Keep chunk sizes reasonable (~500 tokens)
