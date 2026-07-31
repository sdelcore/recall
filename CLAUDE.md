# CLAUDE.md

Project-level guidance for the recall semantic memory search CLI.

## Build Commands

This repo has its own flake, and its dev shell carries the whole toolchain —
`cargo`, `rustfmt`, `clippy`. Use it rather than a sibling project's shell:
aria's shell has no rustfmt or clippy, so two of the four CI gates cannot run
in it.

```bash
nix develop --command cargo build
nix develop --command cargo test --locked --all-targets
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --all-targets --locked -- -D warnings

# Accept a deliberate ranking change (rewrites tests/snapshots/ranking.json)
RECALL_UPDATE_SNAPSHOTS=1 nix develop --command cargo test --test ranking
```

Those four commands are exactly what CI runs, on ubuntu and macos. CI installs
the toolchain with rustup instead of Nix; a separate workflow runs
`nix build .` so the flake cannot rot.

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
│   ├── intent.rs     # Temporal query detection (year extraction, decay skip)
│   ├── embedder.rs   # Ollama HTTP client for embeddings
│   ├── reranker.rs   # LLM reranking (claude-agent-sdk)
│   ├── lint.rs       # Vault link linter (dangling wikilinks, orphaned notes)
│   ├── mcp.rs        # MCP stdio server (recall_search, recall_index, recall_status)
│   └── watcher.rs    # File system watcher (notify-rs)
├── tests/
│   ├── common/       # RecallSandbox: hermetic RECALL_DB_PATH
│   ├── ranking.rs    # Retrieval regression harness (snapshot + labeled queries)
│   ├── snapshots/    # Checked-in ranking baseline
│   ├── indexing.rs   # Date cascade, chunk metadata, search date bounds
│   ├── decay.rs      # Recency decay end-to-end
│   ├── collections.rs# Collection CRUD and half-life
│   ├── lint.rs       # Link resolution, orphans
│   ├── mcp.rs        # MCP protocol surface over stdio
│   ├── maintenance.rs# vacuum / rebuild-fts
│   └── cli.rs        # Command dispatch and output formats
├── Cargo.toml
└── README.md
```

## Architecture

- **Search** (`search.rs`): the whole pipeline behind one function, `search()`. BM25 / vector / RRF fusion, optional reranking, recency decay. The CLI and the MCP server both call it, so the rules ("fetch `RERANK_CANDIDATES` when reranking", "fall back to BM25 when no embeddings exist") exist once, not per call site. Renders nothing — callers format the `SearchOutcome`.
- **Store** (`store.rs`): SQLite database with FTS5 for BM25 search and sqlite-vec for vector KNN search. Persistence and query only; chunking moved out to `ast.rs` / `chunker.rs`. Also owns `EXCLUDE_GLOBS` / `is_excluded()` — the one list of paths that are not notes, obeyed by the indexer, the linter, and the watcher alike.
- **AST chunker** (`ast.rs`): comrak-based splitter. Groups top-level blocks into chunks on heading boundaries (any level) and a soft 1600-char cap, but only ever cuts *between* blocks — a code block, list, or table is never torn. No overlap between chunks.
- **Chunker** (`chunker.rs`): wraps `ast.rs` with the file-level metadata stamped onto every chunk from a file — the date cascade and frontmatter `status`. Its heuristics are private and unit-tested in isolation.
- **Intent** (`intent.rs`): two string-inspection functions, no LLM call, so latency stays predictable. `year()` pulls a 2000-2099 year out of the query, which becomes an `after` bound; `is_temporal()` says whether the query asks about a point in time, which skips recency decay. It used to classify into four buckets, two of which routed nothing and one of which — `structural` — was checked first and actively harmful: `*.md from 2025` matched it and lost both the year bound and the decay skip. Reranking is never routed; it costs seconds, so only `--rerank` turns it on.
- **Frontmatter** (`frontmatter.rs`): hand-rolled scanner for the leading `---` block. Reads flat scalars only (`date`, `last_updated`, `status`, `type`, `aliases`). No `serde_yaml` — it is unmaintained.
- **Embedder** (`embedder.rs`): HTTP client for Ollama embedding API. `EMBEDDING_MODEL` (nomic-embed-text, 768-dim) is a const because it is half the index fingerprint; the server URL honors `RECALL_OLLAMA_URL`.
- **Reranker** (`reranker.rs`): LLM-based reranking through the `claude-agent-sdk` crate. No API key needed; all candidates go into one prompt. Reranking failures propagate as errors — degrading to RRF order is invisible to the caller, and over MCP the warning never reaches the model.
- **MCP** (`mcp.rs`): Model Context Protocol server over stdio. Exposes 3 tools: `recall_search`, `recall_index`, `recall_status`, plus dynamic server instructions on `initialize`. Registered in ARIA's Claude Code MCP config.
- **Lint** (`lint.rs`): `recall lint` — reads the vault from disk (never the index) and reports dangling wikilinks and orphaned notes. A link resolves against *every* registered collection, so a cross-project link is not a finding. See "Linting" below.
- **Watcher** (`watcher.rs`): notify-rs over every collection's `root_path`, debounced by `DEBOUNCE_MS`. It re-indexes a changed `.md` file into the collection that owns it. It does **not** notice deletions — an event whose path no longer exists is skipped, so only `recall index` (or the `recall_index` tool) prunes a removed note. That is the one job the watcher does not cover.

## Database

Location: `~/.local/share/recall/memory.sqlite`. `RECALL_DB_PATH` overrides it.

| Table | Columns |
|---|---|
| `collections` | `id`, `name` (unique), `root_path`, `description`, `half_life_days`, `created_at` |
| `files` | `id`, `collection_id`, `file_path`, `mtime`, `indexed_at`, `chunk_count`, `UNIQUE(collection_id, file_path)` |
| `chunks` | `id`, `file_id`, `collection_id`, `date`, `date_source`, `section`, `status`, `start_line`, `end_line`, `content` |
| `fts_chunks` | FTS5 external-content over `chunks.content`, synced by three triggers |
| `vec_embeddings` | vec0, `float[768]`, rowid = `chunks.id` |
| `config` | `key`, `value` — holds `index_fingerprint` and nothing else |

`chunks.collection_id` is denormalized from `files` so a collection-scoped
search filters without a join. Three columns are gone: `memory_type` (path
heuristics restating what `file_path` already said, never filtered or ranked
on), `project` (hardcoded `None` at every write, so it never held a value), and
`chunk_index` (written, never read).

Indexes are `idx_chunks_file_id`, `idx_chunks_collection_id`, and
`idx_files_collection_id` — the three that back a real predicate. Indexes on
`chunks.date` and `files.mtime` were dropped: both columns are only ever
residual filters on rows already fetched by `fts_chunks MATCH` or `id IN (...)`,
so the planner never chose them and they cost every insert.

No migration code by design, and the fingerprint is the *only* compatibility
mechanism — there is no second check that refuses to open an old database.
`config['index_fingerprint']` holds
`schema=<n>;chunker=<n>;embedding=<model>`, built from `SCHEMA_VERSION`
(currently 3, bumped when the three columns above were dropped),
`CHUNKER_VERSION` (currently 3), and `EMBEDDING_MODEL`. On a mismatch,
`Store::open` drops `files` / `chunks` / `fts_chunks` / `vec_embeddings` and
the next `recall index` rebuilds them. Collections survive the rebuild.

**A schema or chunker change therefore costs a full reindex plus a full
`recall embed`** — the vectors go with the chunks, and re-embedding a real
vault is the slow half. Bump `SCHEMA_VERSION` when the table shape changes and
`CHUNKER_VERSION` when the chunker's output would differ for unchanged input.
Never write `ALTER TABLE` or a `migrate_*` helper.

A chunk's `date` comes from a cascade — frontmatter `date:` /
`last_updated:`, then a `YYYY-MM-DD` filename, then the file's mtime — and
`date_source` records which rung won.

## Key Patterns

- Hybrid search uses **Reciprocal Rank Fusion (RRF)** with `RRF_K = 60`, the constant from the RRF paper
- LLM reranking batches all candidates into one prompt for efficiency (1 LLM call per search)
- Chunking splits on headings and a soft 1600-char cap, always at an AST block boundary. No overlap between chunks.
- FTS5 is kept in sync via triggers on the `chunks` table
- MCP server uses JSON-RPC over newline-delimited JSON on stdio
- Every reranker error path propagates. Degraded-but-plausible results are worse than an error the caller can see.
- The same rule holds for the store. No `COUNT(*)` is `unwrap_or(0)`, hydration
  propagates a DB error instead of skipping the hit, and a failed embedding
  delete fails the re-index. A corrupt database that reports itself as empty is
  indistinguishable from a fresh install, and every consumer draws the wrong
  conclusion from that. The one exception is the chunk count inside
  `check_index_fingerprint`, which runs before `init_schema` and so may
  legitimately find no `chunks` table; it is commented as such.

## Search Pipeline

```
Query → [year → after bound] → BM25 + Vector → RRF Fusion
      → [LLM Reranker] → truncate to limit → Recency Decay → Results
```

One implementation, in `search.rs::search_with_store`, and one retrieval
strategy: hybrid, always. There is no `--hybrid` flag because the fusion is
strictly better wherever vectors exist, and only the index knows whether they
do — with zero embeddings the pipeline degrades to BM25 on its own, so search
works before `recall embed` has ever run.

`--rerank` is the one retrieval knob the caller still holds, and it is the only
thing that turns reranking on: nothing routes to it, because it costs seconds
of LLM latency. Asking for it makes the pipeline over-fetch
`RERANK_CANDIDATES` before fusion. **A reranking failure is an error, not a
silent downgrade.** An unreachable model, or one that returns the wrong number
of scores, fails the search rather than returning RRF order — a caller who
asked for reranking and got unreranked results has no way to tell, and over
MCP the `warn!` never reaches the model.

`SearchOptions` (`after` / `before` / `collection_id`)
applies to **both** candidate lists before RRF fusion, not just to the
hydrated output — filtering after fusion would return fewer than `limit`. The
vector path over-fetches and prunes unconditionally; an "is any filter set"
shortcut has to restate `append_filters`'s field list by hand, and the stale
copy shipped a bug once (`before` was pruned nowhere).

Recency decay is **unconditional**. It runs last, in `search.rs::apply_decay`,
on the already-reranked and truncated list:
`score *= 0.5 + 0.5 * exp(-ln2 * age_days / half_life_days)`. The floor of
0.5 is deliberate — recency separates comparable results and never beats
relevance. It was a config switch defaulting to off, which meant the ranking
everyone actually got was the worse one: the labeled query set scores 14/16
without decay and 16/16 with it. Half-life is **per collection**: the
collection's own `half_life_days` if it has one, else the
`DEFAULT_HALF_LIFE_DAYS` const (90 days) in `search.rs`.

Two skips survive, because they are behaviour rather than configuration:
undated chunks (no evidence of age is not evidence of staleness) and temporal
queries, which ask for old material on purpose. Dates are kept **out** of the
rerank prompt so the arithmetic is the only recency authority. `--trace`
reports `decay_factor` and `pre_decay_score`.

## Ranking Regression Harness

`tests/ranking.rs` is the baseline for every future ranking change. It
builds a 14-note fixture vault in a temp dir, indexes it, and runs 24 fixed
queries. BM25-only, so CI needs no Ollama.

Decay is unconditional, so there is no second profile to run. The comparison
the old "decay off" profile provided comes from `--trace` instead:
`pre_decay_score` is the score the pipeline would have returned without decay,
so sorting on it reconstructs the pre-decay ranking from the *same* search.
That is a stronger baseline — it is measured from the shipped code path rather
than from a second one.

Two guards:

- **Differential snapshot.** Every query's ranked
  `relative/path.md:start_line` list is compared with
  `tests/snapshots/ranking.json`. A ranking change is a readable diff, not a
  surprise. Regenerate only on purpose:
  `RECALL_UPDATE_SNAPSHOTS=1 cargo test --test ranking`. Review that diff —
  it is the record of what the change did.
- **Labeled queries.** 16 queries with a known-correct top result. Measured:
  **16/16 as ranked, 14/16 in pre-decay order.** The two decay fixes are
  supersession pairs (a keyword-dense superseded decision record against the
  short note that overturned it). That gap is the reason decay exists; the
  test asserts the miss list exactly, so a fixture drift that erases the gap
  fails loudly instead of quietly passing.

Fixture dates are written relative to today, so the corpus keeps its ages
forever and the snapshot cannot rot. Filenames are deliberately not
date-shaped for the same reason (`tests/indexing.rs` covers that rung of the
date cascade).

## Configuration

There is none. Recall reads no config file, and `config.rs` is deleted. Every
value that was a key is now either a const next to the code that would be
re-measured if it changed, or a column the database owns:

| Const | Where | Value |
|---|---|---|
| `RRF_K` | `store.rs` | 60 — from the RRF paper |
| `EXCLUDE_GLOBS` | `store.rs` | Templates, `.obsidian`, attachments, sync conflicts |
| `EMBEDDING_MODEL` | `embedder.rs` | `nomic-embed-text` — half the index fingerprint |
| `RERANK_CANDIDATES` | `search.rs` | 20 — what fits in one rerank prompt |
| `DEFAULT_HALF_LIFE_DAYS` | `search.rs` | 90 |
| `RERANK_MODEL` | `reranker.rs` | `haiku` |
| `DEBOUNCE_MS` | `watcher.rs` | 1500 |
| `ORPHAN_EXCLUDE` | `lint.rs` | daily notes, journals, session logs |

Two environment variables change behaviour: `RECALL_DB_PATH` (also what makes
the test suite hermetic) and `RECALL_OLLAMA_URL` (a remote GPU box is the one
plausible second value). `RUST_LOG` filters tracing to stderr, default `warn`,
and `RECALL_UPDATE_SNAPSHOTS` is test-only. `--limit` defaults to 5 at the clap
layer, and the MCP `limit` argument defaults to 5 in `tool_search`.

The file never existed on the machine this runs on, so all 22 keys had only
ever held their compiled-in defaults — and `[index] paths` made `recall status`
report a path that had nothing to do with where the collections pointed. A knob
nobody turns is a second place for the answer to live.

Per-collection half-life lives in the DB — it is a property of the corpus, and
one machine covers several corpora with different rates of change:

```bash
recall collection half-life notes 30   # set or replace (must be positive)
recall collection half-life notes      # clear; falls back to DEFAULT_HALF_LIFE_DAYS
```

`collection half-life` is the column's only writer, which is why its `days > 0`
check is the only one: `recency_factor` divides by the half-life and carries no
guard of its own. `collection describe <name> [text]` has the same shape over
`collections.description`.

## Linting

`recall lint [--collection <name>] [--json]`

Reads the filesystem, not the index, so the answer never goes stale with the
index. It writes nothing and it **never runs as part of indexing** — with no
migration path, an index that aborts on a link typo is a self-inflicted outage.
Findings never change the exit code: lint warns, it does not gate, and there is
no `--fail-on-unresolved`. Only a usage error exits nonzero.

A link is **resolved** when any registered collection holds the target, and
**unresolved** when none does. Only unresolved is a finding. There used to be a
third state, `resolved-foreign`, with its own report section listing links that
are correct — the valuable part is resolving across collections, which needs no
state to record. Every collection is always scanned; `--collection` only
narrows what is reported. Resolution matches note basenames **and** frontmatter `aliases:` —
an alias-blind resolver reports the most-linked notes in the vault as dangling.

Links come from the comrak AST, never a regex over raw text, so `[[Target]]`
inside a fenced block, inline code, raw HTML, or frontmatter is not a link.
`%%Obsidian comments%%` are stripped by a toggle over the AST text stream, so a
`%%` inside a code block cannot open a comment.

An **orphan** is a note with zero incoming *and* zero outgoing wikilinks. The
`ORPHAN_EXCLUDE` globs in `lint.rs` (case-insensitive, matched against the
absolute path) keep daily notes and session logs out of that list — they are unlinked by
design and would otherwise be the whole report.

The report goes to stdout. Redirect it somewhere **outside** every collection
root if you keep it: a large report written into an indexed vault gets indexed,
and in one case froze Obsidian.

## MCP Server

Start: `recall serve`. It takes no flags — `--mode` had one legal value,
validated against its own default. Anything still passing `--mode mcp` fails at
argument parsing, so check the caller's nix config when the server will not
start.

Tools exposed:
- `recall_search(query, limit?, rerank?, after?, before?, collection?)` — search the vault
- `recall_index(collection?, path?)` — reconcile the index with the filesystem
- `recall_status()` — index health, collections, and stats

The tool surface is deliberately narrower than the CLI's: `trace` is a
diagnostic and retrieval strategy is not a choice an agent can make well, so
neither is exposed. Both front ends now build the same `SearchRequest` and run
the same pipeline, so an identical query ranks identically over MCP and on the
CLI.

`initialize` returns dynamic `instructions` built from live index state: file
and chunk counts, collection names, capability gaps ("No vector embeddings;
BM25-only…"), and the retrieval contract — results are dated digests, not live
state; the newer date wins on conflict; volatile facts must be verified live.

`recall_search` and `recall_status` emit their payload on **both** MCP
channels: `content[].text` and `structuredContent`. The Claude Code CLI
forwards only the latter to the model, other clients only the former, so
emitting one alone blanks the tool for half the ecosystem. `recall_index` and
every error are text-only — a one-line "Indexed N files" has no structure worth
a second rendering.

For `recall_search` the text channel is the structured payload pretty-printed.
It used to be a hand-written prose renderer of the same nine fields, which also
restated the retrieval contract and the `Read(path, offset=line-20)` hint that
already reach the model through `instructions` and the tool description — three
copies of two sentences, and a second formatter to keep in sync with the first.
`recall_search` also declares an `outputSchema`; each hit carries `path`, `line`
(absolute, 1-indexed), `date`, `date_source`, `status`, `collection`, `section`,
`score`, `content`, with explicit `null` for unknowns.

Registered in ARIA's Claude Code config at `~/.config/claude/settings.local.json`.
