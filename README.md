# recall

A semantic memory search CLI that indexes markdown files (primarily Obsidian vaults) and provides token-efficient retrieval for LLM consumption.

## Features

- **SQLite + FTS5** - BM25 keyword search with full-text indexing
- **Vector Embeddings** - `BAAI/bge-small-en-v1.5`, 384 dimensions, computed in-process on the CPU. No service, no network at runtime
- **Hybrid Search** - Combine BM25 + vector with Reciprocal Rank Fusion. Works on any host; nothing to install or run alongside it
- **Recency Decay** - Per-collection half-life on the final score, always on
- **LLM Reranking** - Opt-in with `--rerank`; errors out rather than degrading
- **Vault Linting** - Dangling wikilinks and orphaned notes, warn-only
- **File Watching** - Auto-index on file changes
- **MCP Server** - `recall serve`, three tools over stdio
- **Token-Efficient Output** - Compact results with citations
- **No configuration** - Collections live in the database; there is no config file

## Quick Start

```bash
# Enter development environment
nix develop

# Build
cargo build

# Register a collection, index it, then embed it
recall collection add ~/Obsidian --name vault
recall index
recall embed

# Search
recall search "what are the user's preferences"

# Check status
recall status
```

`recall embed` needs nothing running: the model is loaded into the process and
runs on the CPU. Search works after `recall index` alone, but it is keyword-only
until `recall embed` has filled in the vectors.

## CLI Commands

### Search

Search is always hybrid (BM25 + vector), and always degrades to BM25 alone
when nothing is embedded yet. There is no retrieval-strategy flag: the fusion
is strictly better when vectors exist, and only the index knows whether they
do. A query naming a year sets `--after` from it automatically.

The vector half has no external dependency — the query is embedded in-process,
so hybrid search behaves the same on every host. BM25-only means one thing:
`recall embed` has not run on this index yet.

```bash
recall search "project deadlines" --limit 10

# LLM reranking. Errors out if the model is unreachable — it never
# quietly returns unreranked results. Nothing else turns it on.
recall search "query" --rerank

# Output formats
recall search "query" --format compact   # Default; snippets truncated
recall search "query" --format json      # Full text plus every field

# Per-result diagnostics: BM25 rank, vector rank, RRF score, reranker score,
# decay factor, and pre-decay score. Forces JSON output.
recall search "query" --trace

# Filters
recall search "query" --after 2024-01-01    # Lower date bound, inclusive
recall search "query" --before 2024-12-31   # Upper date bound, inclusive
recall search "query" --collection vault
```

Date bounds apply to `chunks.date` — whichever rung of the date cascade
(frontmatter → filename → mtime) supplied it. They are applied to the
candidate lists, so they behave identically in BM25 and hybrid mode.

`search` is a required subcommand. The bare `recall "query"` shorthand is gone:
it took the rest of the command line as part of the query, so `recall "q"
--limit 10` searched for the literal string `q --limit 10`.

### Indexing

Indexing writes into a collection, so register one first (see below).

```bash
recall index                    # Every collection, at its root_path
recall index --collection vault # One collection
```

`index` reconciles: it re-reads every file whose mtime moved and forgets every
indexed file that has left the disk. There is no `--incremental`, because the
two modes differed only in that the "full" one dropped the collection's rows
first — which made it the only one that noticed a deletion. The CLI defaulted
to full while the MCP tool ran incremental, so a deleted note stayed
searchable forever for whoever used the other front end.

A collection always indexes its own `root_path`. Single files are the
watcher's job (`recall watch`), which is what runs in production — but the
watcher never prunes. It re-indexes files that change and skips events whose
path is gone, so a deleted note leaves the index only on the next
`recall index`.

A schema, chunker, or embedding change invalidates the index — the model name
and the vector width are both in the fingerprint. There is no migration code:
recall drops the indexed rows and the next `recall index` **plus**
`recall embed` rebuilds them. Re-embedding is the slow half. Registered
collections survive.

### Collections

A collection is a named root path. Indexing, search, and lint can all be
scoped to one.

```bash
recall collection add ~/Obsidian --name vault
recall collection list
recall collection half-life work 30   # Set or replace the recency half-life
recall collection half-life work      # Clear it; falls back to 90 days
recall collection describe work "Client work notes"  # Context shown with hits
recall collection describe work       # Clear the description
recall collection remove work         # Drops its files, chunks, and embeddings
```

`half-life` and `describe` are the only writers of their columns, so each
validates in one place: a half-life must be positive, and a missing value
clears rather than guesses. `add` resolves the root and fails if it does not
exist — a typo is a typo, not a collection that silently indexes nothing.

### Embeddings

```bash
recall embed                  # Embed every chunk that has no vector yet
```

Embeddings run **in-process on the CPU** — `BAAI/bge-small-en-v1.5`, 384
dimensions, through candle. There is no service to start and no network call at
runtime. The model loads once per run (~0.6s) and then embeds roughly 9
chunks/sec, in batches of 32 — so a 10,000-chunk vault takes about 20 minutes,
once.

Weights come from `RECALL_MODEL_PATH` if it is set, otherwise from the Hugging
Face cache — which downloads them on first use. The Nix package pins them into
the store and sets the variable, so a packaged binary never reaches the network.

Embedding is always incremental. A vector cannot go stale without its chunk
being rewritten, and a rewrite drops the old row. Changing the model or its
dimension changes the index fingerprint, which drops every chunk and vector: a
full `recall index` plus a full `recall embed`.

### File Watching

```bash
recall watch                  # Watch, auto-index, and embed
```

The watcher indexes a changed note within `DEBOUNCE_MS` and embeds it on the
next sweep — every 300s, up to 128 chunks per sweep, more sweeps back to back
while a backlog remains. So a note is keyword-searchable in seconds and
vector-searchable within a few minutes, and neither `recall embed` nor a timer
has to be run by hand.

The two are separated on purpose. Indexing a note costs milliseconds and
embedding its chunks costs a model load plus ~110ms each, so embedding inline
would put a sync that rewrote a hundred notes in front of the next save.
Keyword search is what decides whether a note is findable at all; it goes
first, and the vectors catch up.

A sweep reads the pending count before anything else, so an idle vault never
loads the model, and the weights are dropped once the backlog is empty. If the
model cannot load at all, the watcher says so once and keeps indexing —
losing vector search is bad, losing both is worse. `recall status` reports the
coverage gap.

The watcher still never prunes: a deleted note leaves the index on the next
`recall index`.

### Linting

```bash
recall lint                        # Warn-only report on stdout
recall lint --collection notes     # Report findings for one collection
recall lint --json                 # Machine-readable report
```

A link is **resolved** when any registered collection holds the target, and
**unresolved** when none does. Only unresolved is a finding — a cross-project
link is deliberate, not a mistake.

All collections are always scanned, so a link into another vault resolves;
`--collection` only narrows what gets reported. Targets match note basenames
and frontmatter `aliases:`. Links inside fenced code blocks, inline code, raw
HTML, frontmatter, and `%%Obsidian comments%%` are ignored — link extraction
walks the markdown AST rather than the raw text.

An **orphan** is a note with zero incoming *and* zero outgoing wikilinks.

Lint never runs as part of indexing and never writes anything. Findings never
change the exit code — there is no `--fail-on-unresolved`, and lint warns
rather than gates. Only a usage error (no collections registered, or an
unknown `--collection`) exits nonzero.

The report goes to stdout; there is no `--out`, so redirect it yourself, and
redirect it somewhere **outside** every collection root if you want to keep
it, because a report written into an indexed vault gets indexed.

### Status & Maintenance

```bash
recall status                 # Collections, statistics, integrity, orphan counts
recall status --json          # JSON output

recall maintenance vacuum     # Reclaim space after deletes
recall maintenance rebuild-fts # Drop and rebuild the FTS5 index from chunks
```

`status` answers "is the index usable?" in one place: the counts and the
`PRAGMA integrity_check` / orphan-row checks that say whether to trust them.
There is no `maintenance check` — it opened the same store, ran the same
`get_stats()`, and printed the same counts. `maintenance` now holds only the
two commands that change the database.

## Recency Ranking

**Always on.** The final score of every dated result is multiplied by a
half-life curve:

```
score *= 0.5 + 0.5 * exp(-ln2 * age_days / half_life_days)
```

The factor never drops below 0.5, so age costs a result at most half its
score. Recency breaks ties between results the retriever already considers
comparable; it cannot promote a weak fresh note over a strong old one.

It was a config switch defaulting to off, which meant the ranking everyone
actually got was the worse one — the labeled query set below scores 14/16
without decay and 16/16 with it, and the floor guarantees it cannot make a
ranking worse. There was nothing left to hedge.

Set a half-life per collection:

```bash
recall collection half-life work 30    # This corpus goes stale in a month
recall collection half-life reference  # Clear: fall back to 90 days
```

**Half-life is a property of the collection.** It lives
in the database because one machine covers several corpora that go stale at
different rates — a daily-notes vault and a reference vault should not share
a number. A collection with none of its own falls back to the 90-day
`DEFAULT_HALF_LIFE_DAYS` const in `search.rs`.

Two cases are left alone:

- **Undated chunks.** No evidence of age is not evidence of staleness, so
  they keep their score untouched.
- **Temporal queries** — "notes from 2023", "when did we switch to Postgres".
  They ask for old material on purpose; demoting age would answer the
  opposite question. A year or a relative time word in the query detects
  these.

Dates are deliberately withheld from the reranker prompt, so this arithmetic
is the only thing in the pipeline that acts on recency. `recall search
--trace` reports `decay_factor` and `pre_decay_score` for every result.

## MCP Server

```bash
recall serve
```

JSON-RPC over stdio. `serve` takes no flags: `--mode` had one legal value,
checked against its own default, so it is gone. A caller still passing
`--mode mcp` fails at argument parsing.

Three tools:

| Tool | Parameters |
|---|---|
| `recall_search` | `query`, `limit?`, `rerank?`, `after?`, `before?`, `collection?` |
| `recall_index` | `collection?`, `path?` |
| `recall_status` | none |

The tool surface is narrower than the CLI's on purpose: `trace` is a
diagnostic, and retrieval strategy is not something an agent can judge. The
schema carries only what the question itself determines: text, date bounds,
collection, and effort. MCP and CLI searches run the identical pipeline, so
an identical query ranks identically in both.

`initialize` returns instructions built from live index state: file and chunk
counts, collection names, and capability gaps (for example "No vector
embeddings; BM25-only").

`recall_search` and `recall_status` emit their payload on both
`content[].text` and `structuredContent`, because the Claude Code CLI forwards
only the second and other clients only the first. `recall_index` and every
error are text-only. `recall_search` declares an `outputSchema`; for it the
text channel is the structured payload pretty-printed, where it used to be a
second, hand-written prose rendering of the same fields. Each hit carries
`path`, `line` (absolute, 1-indexed), `date`, `date_source`, `status`,
`collection`, `section`, `score`, and `content`, with explicit `null` for
unknowns.

`rerank: true` that cannot reach the model returns `isError: true`. It does
not fall back to unreranked results — a caller cannot tell the difference, and
over MCP the warning on stderr never reaches the model.

Registered in ARIA's Claude Code config at `~/.config/claude/settings.local.json`.

## Configuration

There is none. Recall has no config file, and no `RECALL_CONFIG_PATH`.

Everything that varied is either a decision the code owns or a value the
database owns:

| Was a config key | Now |
|---|---|
| `[index] paths`, `[watch] paths` | `collections.root_path` — `recall collection add` |
| `[index] exclude`, `[watch] exclude` | `EXCLUDE_GLOBS` in `store.rs`, one list for the indexer, the linter, and the watcher |
| `[search] default_limit` | `--limit`, default 5 |
| `[search] rrf_k` | `RRF_K` in `store.rs` — the constant from the RRF paper |
| `[embeddings] model` | `EMBEDDING_MODEL` / `EMBEDDING_DIM` in `embedder.rs` — both are in the index fingerprint, so changing either rebuilds the index |
| `[reranking] candidates` | `RERANK_CANDIDATES` in `search.rs` |
| `[reranking] enabled`, `provider`, `top_k` | `--rerank`, and one adapter |
| `[reranking.claude_code] model` | `RERANK_MODEL` in `reranker.rs` |
| `[watch] debounce_ms` | `DEBOUNCE_MS` in `watcher.rs` |
| `[decay] enabled`, `default_half_life_days` | always on; `recall collection half-life` |
| `[lint] orphan_exclude` | `ORPHAN_EXCLUDE` in `lint.rs` |

The file never existed on the machine this tool runs on, so every one of those
keys had only ever held its compiled-in default. A knob nobody turns is not
flexibility; it is a second place for the answer to live.

## Environment

| Variable | Purpose |
|---|---|
| `RECALL_DB_PATH` | Database location. Default `~/.local/share/recall/memory.sqlite` |
| `RECALL_MODEL_PATH` | Directory holding `config.json`, `tokenizer.json`, `model.safetensors`. Set it and the binary never touches the network; unset, the weights come from the hf-hub cache. The Nix package sets it for you |
| `RUST_LOG` | Tracing filter. Default `warn`, to stderr |

## Storage

| Location | Purpose |
|----------|---------|
| `~/.local/share/recall/memory.sqlite` | SQLite database |

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

## Architecture

```
        ┌─────────────┐          ┌─────────────┐
        │ CLI (clap)  │          │ MCP server  │
        └──────┬──────┘          └──────┬──────┘
               └───────────┬────────────┘
                           ▼
   ┌──────────┬────────────┬────────────┬──────────┐
   │  search  │   index    │   watch    │   lint   │
   │  BM25 +  │  comrak    │  notify    │  comrak  │
   │  vector  │  chunker   │            │   AST    │
   │  → RRF   │            │            │          │
   │ → rerank │            │            │          │
   │ → decay  │            │            │          │
   └─────┬────┴─────┬──────┴─────┬──────┴──────────┘
         │          │            │
         ▼          ▼            ▼
   ┌─────────────────────────┐       ┌──────────┐
   │          store          │◄──────┤ embedder │
   │  SQLite + FTS5 + vec0   │       │ (candle) │
   └─────────────────────────┘       └──────────┘
```

The CLI and the MCP server are two front ends over one pipeline. Both build the
same `SearchRequest`, so an identical query ranks identically in either.

`embedder` is a library, not a client: candle runs the BERT weights inside the
recall process, so the box has no arrow leaving the diagram. Once the weights
are on disk, every part of the system except `--rerank` (which calls an LLM)
works offline.

`lint` is the exception: it reads the filesystem rather than the index, and
asks the store only for the collection roots.

## Building

```bash
# With Nix
nix build

# With Cargo
cargo build --release
```

The Nix package pins the model weights into the store (`nix/model.nix`, at a
fixed Hugging Face revision) and wraps the binary so `RECALL_MODEL_PATH` points
at them. That adds ~128 MiB to the closure and buys a binary that embeds on a
host with no network. The wrapper uses `--set-default`, so exporting
`RECALL_MODEL_PATH` yourself still wins.

A Cargo build has no such default: it downloads the weights into the hf-hub
cache on first embed unless you set `RECALL_MODEL_PATH` yourself.

## Testing

```bash
cargo test
```

The suite is hermetic — every test drives the CLI with `RECALL_DB_PATH`
pointed at a temp dir, so it never reads your real
index or vault. Nothing downloads model weights; retrieval tests are BM25-only.

`tests/ranking.rs` guards ranking itself. It indexes a fixed 14-note fixture
vault and runs 24 queries, comparing the ranked results against
`tests/snapshots/ranking.json`. A change to retrieval, fusion, or decay shows
up as a diff of that file. When the change is intended, regenerate the
baseline and commit the diff:

```bash
RECALL_UPDATE_SNAPSHOTS=1 cargo test --test ranking
```

Sixteen of those queries also carry a known-correct top result. The hit rate
is 16/16 as ranked and 14/16 in pre-decay order — the two queries decay fixes
are the ones where a superseded note is denser in the query's keywords than
the note that replaced it. Because decay is unconditional there is no second
profile to run: `--trace` reports `pre_decay_score`, so sorting the same
result set on it reconstructs the ranking decay changed.
