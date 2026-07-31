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

# Accept a deliberate ranking change (rewrites tests/snapshots/ranking.json)
RECALL_UPDATE_SNAPSHOTS=1 nix develop ~/aria/aria --command cargo test --test ranking
```

## Project Structure

```
recall/
├── src/
│   ├── main.rs       # CLI entry point (clap), command dispatch, rendering
│   ├── search.rs     # The search pipeline; single entry point for CLI + MCP
│   ├── store.rs      # SQLite + FTS5 + sqlite-vec database, persistence
│   ├── ast.rs        # comrak AST chunker (structure only)
│   ├── chunker.rs    # File-level chunk metadata (date cascade, type, status)
│   ├── frontmatter.rs# Minimal YAML frontmatter scalar scanner
│   ├── intent.rs     # Heuristic query intent classifier (--auto routing)
│   ├── config.rs     # TOML configuration: search, reranking, decay, lint
│   ├── embedder.rs   # Ollama HTTP client for embeddings
│   ├── reranker.rs   # LLM reranking (claude-code SDK, Anthropic API, Ollama)
│   ├── lint.rs       # Vault link linter (dangling wikilinks, orphaned notes)
│   ├── mcp.rs        # MCP stdio server (recall_search, recall_index, recall_status)
│   └── watcher.rs    # File system watcher (notify-rs)
├── tests/
│   ├── common/       # RecallSandbox: hermetic RECALL_DB_PATH / RECALL_CONFIG_PATH
│   ├── ranking.rs    # Retrieval regression harness (snapshot + labeled queries)
│   ├── snapshots/    # Checked-in ranking baseline
│   ├── indexing.rs   # Date cascade, chunk metadata, search date bounds
│   ├── decay.rs      # Recency decay end-to-end
│   ├── collections.rs# Collection CRUD and half-life
│   ├── lint.rs       # Link states, orphans, --out guard
│   ├── mcp.rs        # MCP protocol surface over stdio
│   ├── maintenance.rs# check / vacuum / rebuild-fts
│   └── cli.rs        # Command dispatch and output formats
├── Cargo.toml
└── README.md
```

## Architecture

- **Search** (`search.rs`): the whole pipeline behind one function, `search()`. Intent classification, `--auto` routing, BM25 / vector / RRF fusion, optional reranking, recency decay. The CLI and the MCP server both call it, so the rules ("fetch `candidates` when reranking", "fall back to BM25 when no embeddings exist") exist once, not per call site. Renders nothing — callers format the `SearchOutcome`.
- **Store** (`store.rs`): SQLite database with FTS5 for BM25 search and sqlite-vec for vector KNN search. Persistence and query only; chunking moved out to `ast.rs` / `chunker.rs`.
- **AST chunker** (`ast.rs`): comrak-based splitter. Groups top-level blocks into chunks on heading boundaries (any level) and a soft 1600-char cap, but only ever cuts *between* blocks — a code block, list, or table is never torn. No overlap between chunks.
- **Chunker** (`chunker.rs`): wraps `ast.rs` with the file-level metadata stamped onto every chunk from a file — the date cascade, `memory_type`, and frontmatter `status`. Its heuristics are private and unit-tested in isolation.
- **Intent** (`intent.rs`): pure-heuristic classifier, no LLM call, so latency stays predictable. Buckets a query as `lookup` / `exploratory` / `temporal` / `structural`. It never changes behaviour by itself; `--auto` is what opts into the routing (exploratory ⇒ hybrid + rerank, temporal ⇒ `--after` from the extracted year).
- **Config** (`config.rs`): TOML config from `~/.config/recall/config.toml`. Includes `[reranking]` with per-provider settings, `[decay]` for recency ranking, and `[lint]`.
- **Frontmatter** (`frontmatter.rs`): hand-rolled scanner for the leading `---` block. Reads flat scalars only (`date`, `last_updated`, `created`, `updated`, `status`, `type`, `aliases`). No `serde_yaml` — it is unmaintained.
- **Embedder** (`embedder.rs`): HTTP client for Ollama embedding API (nomic-embed-text, 768-dim)
- **Reranker** (`reranker.rs`): LLM-based reranking with 3 configurable providers:
  - `claude-code` (default): Uses `claude-agent-sdk` crate. No API key needed. Batches all candidates into one prompt.
  - `anthropic`: Direct Anthropic Messages API. Parallel calls, needs `ANTHROPIC_API_KEY`.
  - `ollama`: Local model fallback for offline use.
- **MCP** (`mcp.rs`): Model Context Protocol server over stdio. Exposes 3 tools: `recall_search`, `recall_index`, `recall_status`, plus dynamic server instructions on `initialize`. Registered in ARIA's Claude Code MCP config.
- **Lint** (`lint.rs`): `recall lint` — reads the vault from disk (never the index) and reports dangling wikilinks and orphaned notes. See "Linting" below.
- **Watcher** (`watcher.rs`): File system watcher with debouncing for auto-indexing

## Database

Tables: `collections`, `files`, `chunks`, `fts_chunks` (FTS5), `vec_embeddings` (vec0), `config`

Location: `~/.local/share/recall/memory.sqlite`

No migration code by design. `config['index_fingerprint']` holds
`schema=<n>;chunker=<n>;embedding=<model>`, built from `SCHEMA_VERSION`,
`CHUNKER_VERSION`, and `config.embeddings.model`. On a mismatch, `Store::open`
drops `files` / `chunks` / `fts_chunks` / `vec_embeddings` and the next
`recall index` rebuilds them. Collections survive the rebuild.

**A schema or chunker change therefore costs a full reindex, plus a full
`recall embed`.** Bump `SCHEMA_VERSION` when the table shape changes and
`CHUNKER_VERSION` when the chunker's output would differ for unchanged input.
Never write `ALTER TABLE` or a `migrate_*` helper.

A chunk's `date` comes from a cascade — frontmatter `date:` /
`last_updated:`, then a `YYYY-MM-DD` filename, then the file's mtime — and
`date_source` records which rung won.

## Key Patterns

- Hybrid search uses **Reciprocal Rank Fusion (RRF)** with configurable `rrf_k` parameter
- LLM reranking batches all candidates into one prompt for efficiency (1 LLM call per search)
- Chunking splits on headings and a soft 1600-char cap, always at an AST block boundary. No overlap between chunks.
- FTS5 is kept in sync via triggers on the `chunks` table
- MCP server uses JSON-RPC over newline-delimited JSON on stdio
- All reranker error paths log diagnostics and fall back to RRF order (no silent failures)

## Search Pipeline

```
Query → Intent classify → [--auto routing] → BM25 [+ Vector → RRF Fusion]
      → [LLM Reranker] → truncate to limit → [Recency Decay] → Results
```

One implementation, in `search.rs::search_with_store`. Use `--hybrid` for
BM25+vector, `--rerank` for LLM reranking, `--auto` to let the classifier pick
both, or any combination. Intent is always classified — `--auto` only decides
whether it changes parameters — and it is reported by `--trace`.

Hybrid silently falls back to BM25 when the index has zero embeddings, so
`--hybrid` is safe to pass before `recall embed` has ever run.

`SearchOptions` (`after` / `before` / `project` / `file_pattern` /
`collection_id`) applies to **both** candidate lists before RRF fusion, not
just to the hydrated output — filtering after fusion would return fewer than
`limit`. `SearchOptions::is_filtered()` derives "is any filter set" from the
same place `append_filters` reads the fields, so adding a filter cannot leave
the vector path unpruned (that bug shipped once, for `before`).

Recency decay is **off by default** (`[decay] enabled = false`). It runs last,
in `search.rs::apply_decay`, on the already-reranked and truncated list:
`score *= 0.5 + 0.5 * exp(-ln2 * age_days / half_life_days)`. The floor of
0.5 is deliberate — recency separates comparable results and never beats
relevance. Half-life is **per collection**: the collection's own
`half_life_days` if it has one, else `config.decay.default_half_life_days`.
Skipped for undated chunks (no evidence of age is not evidence of staleness)
and for `Intent::Temporal` queries, which ask for old material on purpose.
Dates are kept **out** of the rerank prompt so the arithmetic is the only
recency authority. `--trace` reports `decay_factor` and `pre_decay_score`.

## Ranking Regression Harness

`tests/ranking.rs` is the baseline for every future ranking change. It
builds a 14-note fixture vault in a temp dir, indexes it, and runs 24 fixed
queries twice — decay off, then decay on. BM25-only, so CI needs no Ollama.

Two guards:

- **Differential snapshot.** Every query's ranked
  `relative/path.md:start_line` list is compared with
  `tests/snapshots/ranking.json`. A ranking change is a readable diff, not a
  surprise. Regenerate only on purpose:
  `RECALL_UPDATE_SNAPSHOTS=1 cargo test --test ranking`. Review that diff —
  it is the record of what the change did.
- **Labeled queries.** 16 queries with a known-correct top result, scored as
  a hit rate. Measured: **14/16 with decay off, 16/16 with decay on.** The
  two it fixes are supersession pairs (a keyword-dense superseded decision
  record against the short note that overturned it). That gap is the reason
  decay exists; the test asserts the miss list exactly, so a fixture drift
  that erases the gap fails loudly instead of quietly passing.

Fixture dates are written relative to today, so the corpus keeps its ages
forever and the snapshot cannot rot. Filenames are deliberately not
date-shaped for the same reason (`tests/indexing.rs` covers that rung of the
date cascade).

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

[decay]
enabled = false              # recency-aware ranking; OFF unless set
default_half_life_days = 90.0  # fallback for collections with no half-life

[lint]
# Globs never reported as orphans (still scanned for links).
orphan_exclude = [
  "**/daily/**", "**/journal/**", "**/sessions/**",
  "**/session-*.md", "**/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*.md",
]
```

Per-collection half-life lives in the DB, not the TOML — it is a property of
the corpus, and one config file covers several corpora with different rates of
change:

```bash
recall collection add ~/notes --name notes --half-life-days 30
recall collection half-life notes 30   # set or replace
recall collection half-life notes      # clear; falls back to default_half_life_days
```

## Linting

`recall lint [--collection <name>] [--json] [--out <path>] [--fail-on-unresolved]`

Reads the filesystem, not the index, so the answer never goes stale with the
index. It writes nothing and it **never runs as part of indexing** — with no
migration path, an index that aborts on a link typo is a self-inflicted outage.
Exit code is 0 unless `--fail-on-unresolved` is passed.

Three link states (only the last is a finding):

| State | Meaning |
|---|---|
| `resolved-local` | Target is in the same collection. |
| `resolved-foreign` | Target is in another registered collection. Cross-project links are valid; they are listed for visibility, not flagged. |
| `unresolved` | Target found nowhere. |

Every collection is always scanned; `--collection` only narrows what is
reported. Resolution matches note basenames **and** frontmatter `aliases:` —
an alias-blind resolver reports the most-linked notes in the vault as dangling.

Links come from the comrak AST, never a regex over raw text, so `[[Target]]`
inside a fenced block, inline code, raw HTML, or frontmatter is not a link.
`%%Obsidian comments%%` are stripped by a toggle over the AST text stream, so a
`%%` inside a code block cannot open a comment.

An **orphan** is a note with zero incoming *and* zero outgoing wikilinks. The
`[lint] orphan_exclude` globs (case-insensitive, matched against the absolute
path) keep daily notes and session logs out of that list — they are unlinked by
design and would otherwise be the whole report.

`--out` refuses any destination inside a collection root: a large report written
into an indexed vault gets indexed, and in one case froze Obsidian.

## MCP Server

Start: `recall serve --mode mcp`

Tools exposed:
- `recall_search(query, limit?, rerank?, after?, before?, collection?)` — search the vault
- `recall_index(collection?, path?)` — trigger incremental re-indexing
- `recall_status()` — index health, collections, and stats

The tool surface is deliberately narrower than the CLI's. `hybrid`,
`rerank_provider`, `project`, and `file_pattern` are config or implementation
detail; an agent cannot choose them well, so they are not exposed. Searches run
with `auto: true`, so intent routing (and the temporal-query decay skip) applies
over MCP exactly as it does with `recall search --auto`.

`initialize` returns dynamic `instructions` built from live index state: file
and chunk counts, collection names, capability gaps ("No vector embeddings;
BM25-only…"), and the retrieval contract — results are dated digests, not live
state; the newer date wins on conflict; volatile facts must be verified live.

Every tool result is emitted on **both** MCP channels: `content[].text` and
`structuredContent`. The Claude Code CLI forwards only the latter to the model,
other clients only the former, so emitting one alone blanks the tool for half
the ecosystem. `recall_search` also declares an `outputSchema`; each hit carries
`path`, `line` (absolute, 1-indexed), `date`, `date_source`, `status`,
`memory_type`, `collection`, `section`, `score`, `content`, with explicit
`null` for unknowns.

Registered in ARIA's Claude Code config at `~/.config/claude/settings.local.json`.
