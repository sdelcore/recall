# recall

A semantic memory search CLI that indexes markdown files (primarily Obsidian vaults) and provides token-efficient retrieval for LLM consumption.

## Features

- **SQLite + FTS5** - BM25 keyword search with full-text indexing
- **Vector Embeddings** - Optional semantic search via Ollama
- **Hybrid Search** - Combine BM25 + vector with Reciprocal Rank Fusion
- **Recency Decay** - Optional per-collection half-life on the final score
- **Vault Linting** - Dangling wikilinks and orphaned notes, warn-only
- **File Watching** - Auto-index on file changes
- **Token-Efficient Output** - Compact results with citations
- **Configurable** - Paths, exclusions, and weights via TOML config

## Quick Start

```bash
# Enter development environment
nix develop

# Build
cargo build

# Register a collection, then index it
recall collection add ~/Obsidian --name vault
recall index

# Search
recall "what are the user's preferences"

# Check status
recall status
```

## CLI Commands

### Search

`recall "query"` is a shorthand for `recall search "query"` **with no flags**.
It collects every remaining word into the query, so `recall "q" --limit 10`
searches for the literal text `q --limit 10` and finds nothing. Use the
explicit `search` subcommand whenever you pass a flag.

```bash
# Basic search (BM25)
recall "query"
recall search "project deadlines" --limit 10

# Hybrid search (BM25 + vector). Falls back to BM25 if nothing is embedded.
recall search "query" --hybrid

# LLM reranking
recall search "query" --rerank
recall search "query" --rerank --rerank-provider ollama

# Output formats
recall search "query" --format compact   # Default
recall search "query" --format json
recall search "query" --format full

# Let the query's shape pick the parameters: a long, question-shaped query
# turns on hybrid + rerank; a query naming a year sets --after from it.
recall search "why did we move off sqlite for the queue" --auto

# Per-result diagnostics: BM25 rank, vector rank, RRF score, reranker score,
# decay factor, and pre-decay score. Forces JSON output.
recall search "query" --trace

# Filters
recall search "query" --after 2024-01-01    # Lower date bound, inclusive
recall search "query" --before 2024-12-31   # Upper date bound, inclusive
recall search "query" --collection vault
recall search "query" --project "aria"
recall search "query" --file "*.md"
```

Date bounds apply to `chunks.date` — whichever rung of the date cascade
(frontmatter → filename → mtime) supplied it. They are applied to the
candidate lists, so they behave identically in BM25 and hybrid mode.

### Indexing

Indexing writes into a collection, so register one first (see below).

```bash
recall index                              # Every collection, at its root_path
recall index --incremental                # Changed files only
recall index --collection vault           # One collection
recall index --collection vault --file path.md   # Single file
recall index --collection vault --path ~/notes   # Override the root_path
```

`--file` and `--path` require `--collection`; without it there is no way to
tell which collection the rows belong to, and recall errors rather than guess.

A schema or chunker change invalidates the index. There is no migration code:
recall drops the indexed rows and the next `recall index` (plus `recall embed`)
rebuilds them. Registered collections survive.

### Collections

A collection is a named root path. Indexing, search, and lint can all be
scoped to one.

```bash
recall collection add ~/Obsidian --name vault
recall collection add ~/work/notes --name work --half-life-days 30
recall collection list
recall collection half-life work 30   # Set or replace the recency half-life
recall collection half-life work      # Clear it; falls back to the config default
recall collection remove work         # Drops its files, chunks, and embeddings
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

### Linting

```bash
recall lint                        # Warn-only report on stdout (always exits 0)
recall lint --collection notes     # Report findings for one collection
recall lint --json                 # Machine-readable report
recall lint --out ~/lint.txt       # Write to a file outside every vault
recall lint --fail-on-unresolved   # Exit 1 when a link resolves nowhere
```

Every link lands in one of three states, and only the last is a finding:

- `resolved-local` — target is in the same collection.
- `resolved-foreign` — target is in another registered collection.
  Cross-project links are valid; they are listed, not flagged.
- `unresolved` — target found nowhere.

All collections are always scanned, so a link into another vault resolves;
`--collection` only narrows what gets reported. Targets match note basenames
and frontmatter `aliases:`. Links inside fenced code blocks, inline code, raw
HTML, frontmatter, and `%%Obsidian comments%%` are ignored — link extraction
walks the markdown AST rather than the raw text.

An **orphan** is a note with zero incoming *and* zero outgoing wikilinks.

Lint never runs as part of indexing and never writes to the vault: `--out`
refuses any path inside a collection root, because a report written into an
indexed vault gets indexed.

### Status & Config

```bash
recall status                 # Index statistics
recall status --json          # JSON output
recall config show            # Display config
recall config path            # Show config location

recall maintenance check      # PRAGMA integrity_check + orphan row counts
recall maintenance vacuum     # Reclaim space after deletes
recall maintenance rebuild-fts # Drop and rebuild the FTS5 index from chunks
```

## Recency Ranking

**Off by default.** When enabled, the final score of every dated result is
multiplied by a half-life curve:

```
score *= 0.5 + 0.5 * exp(-ln2 * age_days / half_life_days)
```

The factor never drops below 0.5, so age costs a result at most half its
score. Recency breaks ties between results the retriever already considers
comparable; it cannot promote a weak fresh note over a strong old one.

Enable it, then set a half-life per collection:

```toml
# ~/.config/recall/config.toml
[decay]
enabled = true
default_half_life_days = 90.0
```

```bash
recall collection half-life work 30    # This corpus goes stale in a month
recall collection half-life reference  # Clear: use default_half_life_days
```

**Half-life is a property of the collection, not the config file.** It lives
in the database because one config covers several corpora that go stale at
different rates — a daily-notes vault and a reference vault should not share
a number. `default_half_life_days` only applies to collections that have none
of their own.

Two cases are left alone:

- **Undated chunks.** No evidence of age is not evidence of staleness, so
  they keep their score untouched.
- **Temporal queries** — "notes from 2023", "when did we switch to Postgres".
  They ask for old material on purpose; demoting age would answer the
  opposite question. Intent classification detects these.

Dates are deliberately withheld from the reranker prompt, so this arithmetic
is the only thing in the pipeline that acts on recency. `recall search
--trace` reports `decay_factor` and `pre_decay_score` for every result.

## MCP Server

```bash
recall serve --mode mcp
```

JSON-RPC over stdio. Three tools:

| Tool | Parameters |
|---|---|
| `recall_search` | `query`, `limit?`, `rerank?`, `after?`, `before?`, `collection?` |
| `recall_index` | `collection?`, `path?` |
| `recall_status` | none |

The tool surface is narrower than the CLI's on purpose. `hybrid`,
`rerank_provider`, `project`, and `file_pattern` are server configuration, not
something an agent can judge — the schema carries only what the question
itself determines: text, date bounds, collection, and effort. Searches run
with intent routing on, so the temporal-query decay skip applies over MCP
exactly as it does with `recall search --auto`.

`initialize` returns instructions built from live index state: file and chunk
counts, collection names, and capability gaps (for example "No vector
embeddings; BM25-only").

Every result is emitted on both `content[].text` and `structuredContent`, and
`recall_search` declares an `outputSchema`. Each hit carries `path`, `line`
(absolute, 1-indexed), `date`, `date_source`, `status`, `memory_type`,
`collection`, `section`, `score`, and `content`, with explicit `null` for
unknowns.

Registered in ARIA's Claude Code config at `~/.config/claude/settings.local.json`.

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

[decay]
# Weight results toward recent notes. The final score is multiplied by
# 0.5 + 0.5 * exp(-ln2 * age_days / half_life_days), so age can cost a
# result at most half its score and never outranks a much stronger match.
# Undated chunks and temporal queries ("notes from 2023") are left alone.
enabled = false
# Half-life for collections with no half_life_days of their own.
# Set a per-collection value with `recall collection half-life <name> <days>`.
default_half_life_days = 90.0

[lint]
# Notes matching these globs are never reported as orphans. Daily notes and
# session logs are unlinked by design and would otherwise be the whole report.
# Matched case-insensitively against the absolute path; links in these notes
# still count.
orphan_exclude = [
  "**/daily/**",
  "**/journal/**",
  "**/sessions/**",
  "**/session-*.md",
  "**/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*.md",
]

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

## Testing

```bash
cargo test
```

The suite is hermetic — every test drives the CLI with `RECALL_DB_PATH` and
`RECALL_CONFIG_PATH` pointed at a temp dir, so it never reads your real
index or vault. Nothing needs Ollama; retrieval tests are BM25-only.

`tests/ranking.rs` guards ranking itself. It indexes a fixed 14-note fixture
vault and runs 24 queries with decay off and on, comparing the ranked
results against `tests/snapshots/ranking.json`. A change to retrieval,
fusion, or decay shows up as a diff of that file. When the change is
intended, regenerate the baseline and commit the diff:

```bash
RECALL_UPDATE_SNAPSHOTS=1 cargo test --test ranking
```

Sixteen of those queries also carry a known-correct top result. The hit rate
is 14/16 with decay off and 16/16 with it on — the two queries decay fixes
are the ones where a superseded note is denser in the query's keywords than
the note that replaced it.
