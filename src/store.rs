use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{ffi::sqlite3_auto_extension, params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use zerocopy::AsBytes;

/// The one list of paths that are not notes.
///
/// It used to be three: globs hardcoded in `index_directory`, `[index] exclude`
/// read only by the linter, and `[watch] exclude` matched as substrings. Three
/// lists mean the linter can report an orphan the indexer never saw, and the
/// watcher can index a file `recall index` skips. Every walker over a
/// collection root goes through [`is_excluded`] instead.
pub const EXCLUDE_GLOBS: &[&str] = &[
    "**/Templates/**",
    "**/.obsidian/**",
    "**/attachments/**",
    "**/*.sync-conflict-*",
];

/// True when `path` matches [`EXCLUDE_GLOBS`]. Case-insensitive, because
/// `Templates/` and `templates/` are the same folder to the person who wrote
/// the pattern.
pub fn is_excluded(path: &str) -> bool {
    static COMPILED: OnceLock<Vec<glob::Pattern>> = OnceLock::new();
    let patterns = COMPILED.get_or_init(|| {
        EXCLUDE_GLOBS
            .iter()
            .map(|p| glob::Pattern::new(p).expect("EXCLUDE_GLOBS are compile-time constants"))
            .collect()
    });
    let opts = glob::MatchOptions {
        case_sensitive: false,
        ..glob::MatchOptions::new()
    };
    patterns.iter().any(|p| p.matches_with(path, opts))
}

/// Search result from the memory store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub content: String,
    pub score: f64,
    pub date: Option<String>,
    /// Which rung of the chunker's date cascade produced `date`:
    /// `"frontmatter"`, `"filename"`, or `"mtime"`. Lets a consumer weigh a
    /// declared date differently from a filesystem timestamp.
    pub date_source: Option<String>,
    pub section: Option<String>,
    /// Frontmatter `status:`. Reported, never filtered on.
    pub status: Option<String>,
    /// Name of the collection that owns this chunk (None when constructed
    /// outside of a DB query, e.g. in a reranker test).
    pub collection_name: Option<String>,
    /// Description of that collection, if any. Threaded through so callers
    /// can show source context (qmd-style "context tree") with each hit.
    pub collection_description: Option<String>,
}

/// Statistics about the memory store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStats {
    pub file_count: i64,
    pub chunk_count: i64,
    pub last_indexed: Option<String>,
}

/// Search options for filtering
#[derive(Default)]
pub struct SearchOptions {
    pub after: Option<String>,
    /// Upper bound on `chunks.date` (inclusive). Together with `after` this
    /// bounds a window; alone it answers "what did I know before X".
    pub before: Option<String>,
    /// Restrict results to a single collection (None = all collections)
    pub collection_id: Option<i64>,
}

/// A named collection: a root path that owns a set of indexed files.
/// `description` is the human-readable context string returned alongside
/// search hits so an LLM knows which source the chunk came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub id: i64,
    pub name: String,
    pub root_path: String,
    pub description: Option<String>,
    /// Recency half-life in days for this collection's chunks. Per-collection
    /// because corpora age at wildly different rates — a half-life tuned for a
    /// daily-notes vault pins an archive to the decay floor, and vice versa.
    pub half_life_days: Option<f64>,
    pub created_at: i64,
}

/// Per-result diagnostic info for `--trace`. Fields are populated by whichever
/// search path produced the result; ranks are 0-based.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchTrace {
    pub bm25_rank: Option<usize>,
    pub vec_rank: Option<usize>,
    pub rrf_score: f64,
    pub rerank_score: Option<f64>,
    /// Recency factor multiplied into the final score, and the score it was
    /// multiplied into. Both `None` when decay did not run (disabled, a
    /// temporal query, or an undated chunk) so `--trace` never implies a
    /// decay step that never happened.
    pub decay_factor: Option<f64>,
    pub pre_decay_score: Option<f64>,
}

/// SQLite-based memory store with sqlite-vec for vector search
pub struct Store {
    conn: Connection,
    db_path: PathBuf,
}

/// Reciprocal Rank Fusion's rank-smoothing constant. 60 is the value from the
/// RRF paper (Cormack et al. 2009) and the one every implementation ships.
/// Tuning it needs the labeled query set in `tests/ranking.rs`, not a config
/// file, so it lives with the code that would be re-measured.
const RRF_K: f64 = 60.0;

/// Bump when the `chunks` / `files` / `collections` DDL changes, or when the
/// embedding pipeline changes in a way `EMBEDDING_MODEL`/`EMBEDDING_DIM` don't
/// capture on their own (e.g. pooling strategy, query instruction prefix) —
/// same model name and width, but every stored vector is now incomparable
/// with a freshly embedded one.
const SCHEMA_VERSION: u32 = 5;
/// Bump when the chunker's output would differ for unchanged input
/// (block splitting, size cap, metadata extraction).
const CHUNKER_VERSION: u32 = 3;
/// `config` key holding the fingerprint of the code that built the index.
const FINGERPRINT_KEY: &str = "index_fingerprint";

/// Identity of the pipeline that produced the stored chunks and embeddings.
/// A mismatch means the index on disk was built by different code (or against
/// a different embedding model) and cannot be mixed with new rows, so it is
/// cold-rebuilt. Both halves of the embedding identity are here: the model,
/// because vectors from two models are not comparable even at equal width, and
/// the width, because `vec_embeddings` is declared with it and a stored row of
/// the wrong length cannot be read back at all.
fn index_fingerprint() -> String {
    let model = crate::embedder::EMBEDDING_MODEL;
    let dim = crate::embedder::EMBEDDING_DIM;
    format!("schema={SCHEMA_VERSION};chunker={CHUNKER_VERSION};embedding={model};dim={dim}")
}

/// Register sqlite-vec extension (must be called before opening any connection)
fn register_sqlite_vec() {
    use rusqlite::ffi::{sqlite3, sqlite3_api_routines};
    type AutoExtension = unsafe extern "C" fn(
        *mut sqlite3,
        *mut *mut std::os::raw::c_char,
        *const sqlite3_api_routines,
    ) -> std::os::raw::c_int;
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute::<*const (), AutoExtension>(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

impl Store {
    /// Open or create the memory store
    pub fn open() -> Result<Self> {
        Self::open_at(Self::default_path()?)
    }

    /// Open or create a store at an explicit path. `open` derives the path
    /// from the environment; tests need to vary it per-case without racing
    /// on a process-wide env var.
    pub(crate) fn open_at(db_path: PathBuf) -> Result<Self> {
        register_sqlite_vec();

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create data directory")?;
        }

        let conn = Connection::open(&db_path).context("Failed to open database")?;

        // Enable foreign keys for CASCADE to work
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;

        let store = Store { conn, db_path };
        store.check_index_fingerprint()?;
        store.init_schema()?;

        Ok(store)
    }

    /// Drop the indexed data when [`index_fingerprint`] no longer matches what
    /// built it. Collections (a user's own configuration) survive; files,
    /// chunks, FTS and embeddings are recreated empty by `init_schema`, so the
    /// next `recall index` re-indexes everything from scratch.
    ///
    /// This is the only compatibility mechanism. A second one used to run
    /// first, refuse to open a pre-collections database, and demand the user
    /// delete the file by hand — a contradictory policy that also left
    /// `fts_chunks` and `vec_embeddings` behind. A stale index is a rebuild,
    /// never an error and never a migration.
    fn check_index_fingerprint(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
        )?;
        let current = index_fingerprint();
        let stored: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM config WHERE key = ?1",
                params![FINGERPRINT_KEY],
                |row| row.get(0),
            )
            .optional()?;
        if stored.as_deref() == Some(current.as_str()) {
            return Ok(());
        }

        // The one COUNT(*) that may legitimately fail: this runs before
        // `init_schema`, so on a fresh database `chunks` does not exist yet.
        // The count only decides whether to print "discarding N chunks", so
        // treating a missing table as zero is the right answer, not a swallow.
        let chunk_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap_or(0);
        if chunk_count > 0 {
            eprintln!(
                "Index fingerprint changed ({} -> {}): discarding {} chunks. \
                 Run `recall index` (and `recall embed`) to rebuild.",
                stored.as_deref().unwrap_or("none"),
                current,
                chunk_count
            );
        }
        self.conn.execute_batch(
            "DROP TABLE IF EXISTS chunks; \
             DROP TABLE IF EXISTS files; \
             DROP TABLE IF EXISTS fts_chunks; \
             DROP TABLE IF EXISTS vec_embeddings;",
        )?;
        self.conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?1, ?2)",
            params![FINGERPRINT_KEY, current],
        )?;
        Ok(())
    }

    /// Get the default database path. Honors `RECALL_DB_PATH` for tests/sandboxing.
    fn default_path() -> Result<PathBuf> {
        if let Ok(p) = std::env::var("RECALL_DB_PATH") {
            return Ok(PathBuf::from(p));
        }
        let data_dir =
            dirs::data_local_dir().context("Could not determine local data directory")?;
        Ok(data_dir.join("recall").join("memory.sqlite"))
    }

    /// Get the database path
    pub fn path(&self) -> String {
        self.db_path.to_string_lossy().to_string()
    }

    /// Initialize the database schema
    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(r#"
            -- Named collections (root path + metadata). Owns files transitively.
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                root_path TEXT NOT NULL,
                description TEXT,
                half_life_days REAL,
                created_at INTEGER NOT NULL
            );

            -- Metadata about indexed files
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                collection_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                mtime INTEGER NOT NULL,
                indexed_at INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                UNIQUE(collection_id, file_path)
            );

            -- Text chunks with metadata. collection_id is denormalized from
            -- files for fast filtering at search time.
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                collection_id INTEGER NOT NULL,
                date TEXT,
                date_source TEXT,
                section TEXT,
                status TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
            );

            -- FTS5 for BM25 search
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                content,
                content=chunks,
                content_rowid=id
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, content) VALUES('delete', old.id, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO fts_chunks(fts_chunks, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
            END;

            -- Index configuration and state
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_collection_id ON chunks(collection_id);
            CREATE INDEX IF NOT EXISTS idx_files_collection_id ON files(collection_id);
        "#).context("Failed to initialize schema")?;

        // Create vec0 virtual table for vector embeddings (sqlite-vec)
        // vec0 tables use CREATE VIRTUAL TABLE which doesn't support IF NOT EXISTS
        // in the same way, so we check first
        let has_vec_table: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='vec_embeddings'",
            [],
            |row| row.get(0),
        ).unwrap_or(false);

        if !has_vec_table {
            // The width comes from the embedder so there is exactly one place
            // that knows it; the fingerprint carries it too, so a changed
            // width cold-rebuilds rather than colliding with this DDL.
            self.conn
                .execute_batch(&format!(
                    "CREATE VIRTUAL TABLE vec_embeddings USING vec0(embedding float[{}]);",
                    crate::embedder::EMBEDDING_DIM
                ))
                .context("Failed to create vec_embeddings table")?;
        }

        Ok(())
    }

    /// Get store statistics.
    ///
    /// Every count propagates its error. A swallowed `COUNT(*)` reports
    /// "0 files, 0 chunks" for a corrupt database, which is exactly what a
    /// fresh install reports — so `recall status` calls a broken index healthy
    /// and `run_retrieval` blames the embeddings for the empty result.
    pub fn get_stats(&self) -> Result<StoreStats> {
        let file_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?;

        let chunk_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;

        // Emit RFC3339 with explicit UTC offset (`...Z`) so consumers can render
        // in their own timezone instead of guessing whether the naive string is
        // local or UTC. SQLite's `datetime(..., 'unixepoch')` returns a naive
        // string; we format the unix timestamp directly via chrono instead.
        let last_indexed: Option<String> = self
            .conn
            .query_row("SELECT MAX(indexed_at) FROM files", [], |row| {
                row.get::<_, Option<i64>>(0)
            })?
            .and_then(|ts| chrono::DateTime::<chrono::Utc>::from_timestamp(ts, 0))
            .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true));

        Ok(StoreStats {
            file_count,
            chunk_count,
            last_indexed,
        })
    }

    /// Search using FTS5 (BM25) with filters. The query is sanitized first
    /// (see [`sanitize_fts_query`]) so callers can pass natural language
    /// without worrying about FTS5 operator syntax.
    pub fn search_fts_filtered(
        &self,
        query: &str,
        limit: usize,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        // Build dynamic query with filters
        let mut sql = String::from(
            r#"
            SELECT
                f.file_path,
                c.start_line,
                c.end_line,
                c.content,
                bm25(fts_chunks) as score,
                c.date,
                c.date_source,
                c.section,
                c.status,
                col.name,
                col.description
            FROM fts_chunks
            JOIN chunks c ON c.id = fts_chunks.rowid
            JOIN files f ON f.id = c.file_id
            JOIN collections col ON col.id = c.collection_id
            WHERE fts_chunks MATCH ?
        "#,
        );

        // Build parameter list dynamically
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> =
            vec![Box::new(sanitize_fts_query(query))];

        append_filters(&mut sql, options, &mut params_vec);

        sql.push_str(" ORDER BY score LIMIT ?");
        params_vec.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;

        // Convert params to references for rusqlite
        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        let results = stmt.query_map(params_refs.as_slice(), |row| {
            Ok(SearchResult {
                file_path: row.get(0)?,
                start_line: row.get(1)?,
                end_line: row.get(2)?,
                content: row.get(3)?,
                score: row.get::<_, f64>(4)?.abs(), // BM25 returns negative scores
                date: row.get(5)?,
                date_source: row.get(6)?,
                section: row.get(7)?,
                status: row.get(8)?,
                collection_name: row.get(9)?,
                collection_description: row.get(10)?,
            })
        })?;

        let mut search_results = Vec::new();
        for result in results {
            search_results.push(result?);
        }

        Ok(search_results)
    }

    /// FTS5 search returning chunk IDs in BM25 rank order (for hybrid search).
    /// Applies the same filters as [`Store::search_fts_filtered`] — the
    /// candidate list, not just the final hydration, must be filtered or the
    /// fused result set silently comes up short of `limit`.
    fn search_fts_chunk_ids(
        &self,
        query: &str,
        limit: usize,
        options: &SearchOptions,
    ) -> Result<Vec<i64>> {
        let mut sql = String::from(
            r#"
            SELECT c.id
            FROM fts_chunks
            JOIN chunks c ON c.id = fts_chunks.rowid
            JOIN files f ON f.id = c.file_id
            WHERE fts_chunks MATCH ?
        "#,
        );
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> =
            vec![Box::new(sanitize_fts_query(query))];
        append_filters(&mut sql, options, &mut params_vec);
        sql.push_str(" ORDER BY bm25(fts_chunks) LIMIT ?");
        params_vec.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(params_refs.as_slice(), |row| row.get::<_, i64>(0))?;
        let mut ids = Vec::new();
        for row in rows {
            ids.push(row?);
        }
        Ok(ids)
    }

    /// Store embedding for a chunk using sqlite-vec
    pub fn store_embedding(&self, chunk_id: i64, embedding: &[f32]) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO vec_embeddings(rowid, embedding) VALUES (?1, ?2)",
            params![chunk_id, embedding.as_bytes()],
        )?;
        Ok(())
    }

    /// Chunks that have no vector yet, lowest id first.
    ///
    /// `limit` bounds one call, for a caller that embeds the backlog a piece
    /// at a time. The order is part of the contract: without it a bounded call
    /// could be handed the same rows every time and never finish the backlog.
    /// SQLite reads a negative `LIMIT` as "no limit", so one statement covers
    /// both callers.
    pub fn get_chunks_without_embeddings(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare(
            r#"SELECT c.id, c.content
               FROM chunks c
               WHERE c.id NOT IN (SELECT rowid FROM vec_embeddings)
               ORDER BY c.id
               LIMIT ?1"#,
        )?;

        let bound = limit.map_or(-1, |n| n as i64);
        let results = stmt.query_map([bound], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut chunks = Vec::new();
        for result in results {
            chunks.push(result?);
        }

        Ok(chunks)
    }

    /// Get embedding statistics.
    ///
    /// `run_retrieval` degrades to BM25-only when the embedded count is zero,
    /// so a swallowed error here silently downgrades every search instead of
    /// reporting the broken database.
    pub fn get_embedding_stats(&self) -> Result<(i64, i64)> {
        let total_chunks: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;

        let embedded_chunks: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM vec_embeddings", [], |row| row.get(0))?;

        Ok((embedded_chunks, total_chunks))
    }

    /// Vector search using sqlite-vec KNN, returning chunk IDs in rank order.
    /// KNN cannot express the [`SearchOptions`] predicates, so we always ask
    /// for a larger neighbourhood and prune it afterwards — still in rank
    /// order — until `limit` survivors remain. Unconditionally: a "is anything
    /// filtered?" shortcut has to restate `append_filters`'s field list by
    /// hand, and a stale copy of that list silently drops a predicate.
    fn search_vector_chunk_ids(
        &self,
        query_embedding: &[f32],
        limit: usize,
        options: &SearchOptions,
    ) -> Result<Vec<i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT rowid FROM vec_embeddings WHERE embedding MATCH ?1 AND k = ?2")?;
        let rows = stmt.query_map(
            params![query_embedding.as_bytes(), (limit * 4) as i64],
            |row| row.get::<_, i64>(0),
        )?;
        let mut ids: Vec<i64> = Vec::new();
        for row in rows {
            ids.push(row?);
        }

        if ids.is_empty() {
            return Ok(ids);
        }

        let placeholders = vec!["?"; ids.len()].join(",");
        let mut sql = format!(
            "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id \
             WHERE c.id IN ({placeholders})"
        );
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = ids
            .iter()
            .map(|id| Box::new(*id) as Box<dyn rusqlite::ToSql>)
            .collect();
        append_filters(&mut sql, options, &mut params_vec);

        let mut stmt = self.conn.prepare(&sql)?;
        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let mut allowed: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for row in stmt.query_map(params_refs.as_slice(), |row| row.get::<_, i64>(0))? {
            allowed.insert(row?);
        }

        let mut kept = Vec::with_capacity(limit);
        for id in ids {
            if allowed.contains(&id) {
                kept.push(id);
                if kept.len() == limit {
                    break;
                }
            }
        }
        Ok(kept)
    }

    /// Hybrid search combining BM25 and vector search using Reciprocal Rank
    /// Fusion. Returns per-result trace info (BM25 rank, vector rank, fused
    /// RRF score) so callers can render `--trace` output.
    ///
    /// `options` is applied to *both* candidate lists before fusion. Filtering
    /// only the fused output would drop rows that were already counted against
    /// `limit`, returning fewer results than asked for.
    pub fn search_hybrid_traced(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
        options: &SearchOptions,
    ) -> Result<Vec<(SearchResult, SearchTrace)>> {
        let candidate_count = limit * 3;
        let k = RRF_K;

        let bm25_ranked = self.search_fts_chunk_ids(query, candidate_count, options)?;
        let vector_ranked =
            self.search_vector_chunk_ids(query_embedding, candidate_count, options)?;

        let bm25_idx: HashMap<i64, usize> = bm25_ranked
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let vec_idx: HashMap<i64, usize> = vector_ranked
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();

        // RRF: score(doc) = Σ 1/(k + rank + 1)
        let mut rrf_scores: HashMap<i64, f64> = HashMap::new();
        for (rank, chunk_id) in bm25_ranked.iter().enumerate() {
            *rrf_scores.entry(*chunk_id).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        }
        for (rank, chunk_id) in vector_ranked.iter().enumerate() {
            *rrf_scores.entry(*chunk_id).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        }

        let mut ranked: Vec<(i64, f64)> = rrf_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(limit);

        // Hydration skips a chunk that vanished between the ranking query and
        // this one (`Ok(None)`), and propagates everything else. Treating a
        // database error as "no such chunk" silently returns a shorter result
        // list, which is indistinguishable from the vault simply not holding
        // the answer.
        let mut out = Vec::new();
        for (chunk_id, rrf_score) in ranked {
            let Some(result) = self.get_chunk_by_id(chunk_id, rrf_score)? else {
                continue;
            };
            let trace = SearchTrace {
                bm25_rank: bm25_idx.get(&chunk_id).copied(),
                vec_rank: vec_idx.get(&chunk_id).copied(),
                rrf_score,
                rerank_score: None,
                ..Default::default()
            };
            out.push((result, trace));
        }
        Ok(out)
    }

    /// BM25-only search returning trace info. `vec_rank` is always `None`;
    /// `rrf_score` is the single-list reciprocal rank `1/(k + rank + 1)` so
    /// trace numbers are comparable across modes.
    pub fn search_fts_traced(
        &self,
        query: &str,
        limit: usize,
        options: &SearchOptions,
    ) -> Result<Vec<(SearchResult, SearchTrace)>> {
        let results = self.search_fts_filtered(query, limit, options)?;
        let k = RRF_K;
        Ok(results
            .into_iter()
            .enumerate()
            .map(|(rank, r)| {
                let trace = SearchTrace {
                    bm25_rank: Some(rank),
                    vec_rank: None,
                    rrf_score: 1.0 / (k + rank as f64 + 1.0),
                    rerank_score: None,
                    ..Default::default()
                };
                (r, trace)
            })
            .collect())
    }

    /// Get chunk by ID with score
    fn get_chunk_by_id(&self, chunk_id: i64, score: f64) -> Result<Option<SearchResult>> {
        let result = self.conn.query_row(
            r#"SELECT f.file_path, c.start_line, c.end_line, c.content,
                      c.date, c.date_source, c.section, c.status,
                      col.name, col.description
               FROM chunks c
               JOIN files f ON f.id = c.file_id
               JOIN collections col ON col.id = c.collection_id
               WHERE c.id = ?1"#,
            params![chunk_id],
            |row| {
                Ok(SearchResult {
                    file_path: row.get(0)?,
                    start_line: row.get(1)?,
                    end_line: row.get(2)?,
                    content: row.get(3)?,
                    score,
                    date: row.get(4)?,
                    date_source: row.get(5)?,
                    section: row.get(6)?,
                    status: row.get(7)?,
                    collection_name: row.get(8)?,
                    collection_description: row.get(9)?,
                })
            },
        );

        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Delete vec_embeddings for chunks belonging to a (collection, file) pair.
    /// Same physical file path can exist in multiple collections; we only
    /// drop embeddings for the one being re-indexed.
    ///
    /// One statement, and it propagates. `vec0` has no foreign keys, so a
    /// failure here leaves embeddings on chunk ids that `index_file` is about
    /// to delete and reuse — the caller then writes new chunks that inherit
    /// vectors describing the file's previous text. That is the exact
    /// corruption `orphan_counts().embeddings` exists to report, so producing
    /// it quietly is the worst possible outcome.
    fn delete_embeddings_for_file(&self, collection_id: i64, file_path: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM vec_embeddings WHERE rowid IN \
             (SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id \
              WHERE f.collection_id = ?1 AND f.file_path = ?2)",
            params![collection_id, file_path],
        )?;
        Ok(())
    }

    /// Index a single file into the given collection.
    pub fn index_file(&self, collection_id: i64, file_path: &str) -> Result<()> {
        let path = std::path::Path::new(file_path);
        if !path.exists() {
            anyhow::bail!("File does not exist: {}", file_path);
        }

        let metadata = std::fs::metadata(path)?;
        let mtime = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;

        let content = std::fs::read_to_string(path)?;
        let chunks = crate::chunker::chunk_file(&content, file_path, mtime);

        self.delete_embeddings_for_file(collection_id, file_path)?;

        self.conn.execute("BEGIN", [])?;

        self.conn.execute(
            "DELETE FROM files WHERE collection_id = ?1 AND file_path = ?2",
            params![collection_id, file_path],
        )?;

        self.conn.execute(
            "INSERT INTO files (collection_id, file_path, mtime, indexed_at, chunk_count) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                collection_id,
                file_path,
                mtime,
                Utc::now().timestamp(),
                chunks.len() as i64
            ],
        )?;

        let file_id = self.conn.last_insert_rowid();

        for chunk in &chunks {
            self.conn.execute(
                r#"INSERT INTO chunks (file_id, collection_id, date, date_source, section, start_line, end_line, content, status)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"#,
                params![
                    file_id,
                    collection_id,
                    chunk.date,
                    chunk.date_source,
                    chunk.section,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.content,
                    chunk.status,
                ],
            )?;
        }

        self.conn.execute("COMMIT", [])?;
        Ok(())
    }

    /// Reconcile a collection with the directory it indexes: re-chunk every
    /// markdown file whose mtime moved, then drop every indexed file that is
    /// no longer on disk.
    ///
    /// There is deliberately only one of these. There used to be a "full" and
    /// an "incremental" mode, and the only difference was that "full" dropped
    /// the collection's rows first — which made it the only mode that removed
    /// a deleted note's chunks. The CLI defaulted to full and the MCP tool and
    /// watcher always ran incremental, so a note deleted from the vault stayed
    /// searchable forever, silently, for whoever used the other front end.
    /// Reconciling on every run makes deletion work everywhere and costs one
    /// `Path::exists` per indexed file.
    pub fn index(&self, collection_id: i64, dir_path: &str) -> Result<()> {
        self.index_directory(collection_id, dir_path)?;
        self.prune_missing_files(collection_id)
    }

    /// Forget every file of this collection that has left the filesystem.
    /// Chunks follow by `ON DELETE CASCADE`; embeddings live in a vec0 virtual
    /// table with no foreign keys, so they are deleted explicitly first.
    fn prune_missing_files(&self, collection_id: i64) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare("SELECT file_path FROM files WHERE collection_id = ?1")?;
        let indexed: Vec<String> = stmt
            .query_map(params![collection_id], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(stmt);

        for file_path in indexed {
            if std::path::Path::new(&file_path).exists() {
                continue;
            }
            self.delete_embeddings_for_file(collection_id, &file_path)?;
            self.conn.execute(
                "DELETE FROM files WHERE collection_id = ?1 AND file_path = ?2",
                params![collection_id, file_path],
            )?;
        }
        Ok(())
    }

    /// Index all markdown files in a directory into a collection
    fn index_directory(&self, collection_id: i64, dir_path: &str) -> Result<()> {
        let pattern = format!("{}/**/*.md", dir_path);

        for entry in glob::glob(&pattern)? {
            let path = entry?;
            let path_str = path.to_string_lossy().to_string();

            if is_excluded(&path_str) {
                continue;
            }

            let metadata = std::fs::metadata(&path)?;
            let mtime = metadata
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs() as i64;

            let needs_index = self
                .conn
                .query_row(
                    "SELECT mtime FROM files WHERE collection_id = ?1 AND file_path = ?2",
                    params![collection_id, &path_str],
                    |row| row.get::<_, i64>(0),
                )
                .map(|stored_mtime| stored_mtime < mtime)
                .unwrap_or(true);

            if needs_index {
                if let Err(e) = self.index_file(collection_id, &path_str) {
                    eprintln!("Warning: Failed to index {}: {}", path_str, e);
                }
            }
        }

        Ok(())
    }

    // ── Collection CRUD ───────────────────────────────────────────────────

    /// A new collection starts with no half-life and no description;
    /// `recall collection half-life` and `recall collection describe` are the
    /// only writers of those two columns.
    pub fn create_collection(&self, name: &str, root_path: &str) -> Result<Collection> {
        let now = Utc::now().timestamp();
        self.conn.execute(
            "INSERT INTO collections (name, root_path, created_at) VALUES (?1, ?2, ?3)",
            params![name, root_path, now],
        )?;
        let id = self.conn.last_insert_rowid();
        Ok(Collection {
            id,
            name: name.to_string(),
            root_path: root_path.to_string(),
            description: None,
            half_life_days: None,
            created_at: now,
        })
    }

    pub fn list_collections(&self) -> Result<Vec<Collection>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, root_path, description, half_life_days, created_at \
             FROM collections ORDER BY name",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Collection {
                id: row.get(0)?,
                name: row.get(1)?,
                root_path: row.get(2)?,
                description: row.get(3)?,
                half_life_days: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn get_collection(&self, name: &str) -> Result<Option<Collection>> {
        let result = self.conn.query_row(
            "SELECT id, name, root_path, description, half_life_days, created_at \
             FROM collections WHERE name = ?1",
            params![name],
            |row| {
                Ok(Collection {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    root_path: row.get(2)?,
                    description: row.get(3)?,
                    half_life_days: row.get(4)?,
                    created_at: row.get(5)?,
                })
            },
        );
        match result {
            Ok(c) => Ok(Some(c)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Set or clear a collection's description. Returns false if no
    /// collection by that name exists.
    pub fn set_collection_description(
        &self,
        name: &str,
        description: Option<&str>,
    ) -> Result<bool> {
        let n = self.conn.execute(
            "UPDATE collections SET description = ?1 WHERE name = ?2",
            params![description, name],
        )?;
        Ok(n > 0)
    }

    /// Set or clear a collection's recency half-life (days). `None` clears it,
    /// so the collection falls back to `search::DEFAULT_HALF_LIFE_DAYS`.
    /// Returns false if no collection by that name exists.
    pub fn set_collection_half_life(
        &self,
        name: &str,
        half_life_days: Option<f64>,
    ) -> Result<bool> {
        let n = self.conn.execute(
            "UPDATE collections SET half_life_days = ?1 WHERE name = ?2",
            params![half_life_days, name],
        )?;
        Ok(n > 0)
    }

    /// Remove a collection by name. Cascades to files and chunks; also drops
    /// embeddings (vec0 has no FK CASCADE).
    pub fn remove_collection(&self, name: &str) -> Result<bool> {
        let cid = match self.get_collection(name)? {
            Some(c) => c.id,
            None => return Ok(false),
        };
        self.conn.execute(
            "DELETE FROM vec_embeddings WHERE rowid IN \
             (SELECT id FROM chunks WHERE collection_id = ?1)",
            params![cid],
        )?;
        self.conn
            .execute("DELETE FROM collections WHERE id = ?1", params![cid])?;
        Ok(true)
    }

    /// Best-effort collection lookup by file path: returns the collection
    /// whose `root_path` is a prefix of `file_path`. Used by the watcher.
    pub fn collection_for_path(&self, file_path: &str) -> Result<Option<Collection>> {
        let collections = self.list_collections()?;
        // Pick the longest matching root_path (most specific).
        let mut best: Option<Collection> = None;
        for c in collections {
            if !c.root_path.is_empty()
                && file_path.starts_with(&c.root_path)
                && best
                    .as_ref()
                    .map(|b| c.root_path.len() > b.root_path.len())
                    .unwrap_or(true)
            {
                best = Some(c);
            }
        }
        Ok(best)
    }

    // ── Maintenance ───────────────────────────────────────────────────────

    /// `PRAGMA integrity_check` — returns "ok" on a healthy DB.
    pub fn integrity_check(&self) -> Result<String> {
        let result: String = self
            .conn
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
        Ok(result)
    }

    /// Counts of rows that should not exist if FK constraints held: chunks
    /// pointing to a missing file, files pointing to a missing collection,
    /// and embeddings pointing to a missing chunk.
    pub fn orphan_counts(&self) -> Result<OrphanCounts> {
        let chunks: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE file_id NOT IN (SELECT id FROM files)",
            [],
            |row| row.get(0),
        )?;
        let files: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM files WHERE collection_id NOT IN (SELECT id FROM collections)",
            [],
            |row| row.get(0),
        )?;
        let embeddings: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM vec_embeddings WHERE rowid NOT IN (SELECT id FROM chunks)",
            [],
            |row| row.get(0),
        )?;
        Ok(OrphanCounts {
            chunks,
            files,
            embeddings,
        })
    }

    pub fn vacuum(&self) -> Result<()> {
        self.conn.execute_batch("VACUUM;")?;
        Ok(())
    }

    /// Drop and rebuild the FTS5 index from `chunks`.
    pub fn rebuild_fts(&self) -> Result<()> {
        self.conn
            .execute_batch("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild');")?;
        Ok(())
    }
}

/// Counts of rows that violate referential integrity. Always non-negative;
/// any non-zero value means the DB needs cleanup (or there's a code bug).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrphanCounts {
    pub chunks: i64,
    pub files: i64,
    pub embeddings: i64,
}

/// Append the [`SearchOptions`] predicates to a partially built `WHERE`
/// clause and push their bound values onto `params`. The caller's query must
/// already alias `chunks` as `c` and `files` as `f`, and must bind its own
/// parameters in the same textual order — placeholders are positional.
///
/// Shared by every retrieval path (BM25 rows, BM25 candidate IDs, vector
/// candidate IDs) so a filter cannot apply in one mode and silently vanish in
/// another.
fn append_filters(
    sql: &mut String,
    options: &SearchOptions,
    params: &mut Vec<Box<dyn rusqlite::ToSql>>,
) {
    if let Some(after) = &options.after {
        sql.push_str(" AND c.date >= ?");
        params.push(Box::new(after.clone()));
    }
    if let Some(before) = &options.before {
        sql.push_str(" AND c.date <= ?");
        params.push(Box::new(before.clone()));
    }
    if let Some(cid) = options.collection_id {
        sql.push_str(" AND c.collection_id = ?");
        params.push(Box::new(cid));
    }
}

/// Sanitize a free-text query into something FTS5 will accept as a MATCH
/// expression. Drops FTS5 operators (`?`, `:`, `"`, parentheses, etc.) so
/// natural-language queries from users / classifiers don't blow up with
/// `fts5: syntax error`. Tokens are rejoined with spaces — FTS5 treats
/// multiple unquoted tokens as an implicit AND-of-OR over its tokenizer.
///
/// The hyphen is dropped too, which is less obvious. FTS5 reads `a-b` as
/// the column filter `a` applied to `-b` and fails the whole query with
/// `no such column: b`, so any hyphenated term ("half-life",
/// "bge-small-en") aborted the search instead of matching. Splitting on
/// the hyphen also matches how unicode61 tokenized the document text in the
/// first place, so the two terms are exactly what is in the index.
fn sanitize_fts_query(query: &str) -> String {
    let cleaned: String = query
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A store plus an indexed two-file vault: one note dated 2020, one dated
    /// 2026, both matching the same query. Every chunk gets the same
    /// embedding, so the vector list is decided by the KNN tie-break, not by
    /// content — these tests are about filtering, not about ranking.
    fn indexed_store() -> (tempfile::TempDir, Store, Vec<f32>) {
        let dir = tempfile::tempdir().unwrap();
        let vault = dir.path().join("vault");
        std::fs::create_dir_all(&vault).unwrap();
        std::fs::write(vault.join("2020-01-02.md"), "# Old\n\nquokka zebra\n").unwrap();
        std::fs::write(vault.join("2026-01-02.md"), "# New\n\nquokka zebra\n").unwrap();

        let store = Store::open_at(dir.path().join("memory.sqlite")).unwrap();
        let root = vault.to_string_lossy().to_string();
        let collection = store.create_collection("t", &root).unwrap();
        store.index(collection.id, &root).unwrap();

        let embedding = vec![0.1f32; crate::embedder::EMBEDDING_DIM];
        for (chunk_id, _) in store.get_chunks_without_embeddings(None).unwrap() {
            store.store_embedding(chunk_id, &embedding).unwrap();
        }
        (dir, store, embedding)
    }

    fn hybrid_paths(store: &Store, embedding: &[f32], options: &SearchOptions) -> Vec<String> {
        store
            .search_hybrid_traced("quokka zebra", embedding, 10, options)
            .unwrap()
            .into_iter()
            .map(|(r, _)| r.file_path)
            .collect()
    }

    /// The fingerprint is the only compatibility mechanism, so this is the
    /// whole upgrade story: a database built by different code is rebuilt, not
    /// migrated and not refused. Collections are the user's own configuration
    /// and survive it.
    /// One list, so the indexer, the linter, and the watcher cannot disagree
    /// about what a note is. The watcher used to substring-match `"Templates/"`
    /// against the whole path, which is not what someone writing that pattern
    /// means.
    #[test]
    fn exclusion_is_one_case_insensitive_glob_list() {
        assert!(is_excluded("/v/Templates/daily.md"));
        assert!(is_excluded("/v/templates/nested/daily.md"));
        assert!(is_excluded("/v/.obsidian/workspace.json"));
        assert!(is_excluded("/v/attachments/img.png"));
        assert!(is_excluded("/v/notes/a.sync-conflict-20260101-ABCDEF.md"));

        assert!(!is_excluded("/v/notes/a.md"));
        assert!(!is_excluded("/v/NotTemplatesReally/a.md"));
    }

    #[test]
    fn a_stale_index_fingerprint_drops_the_index_and_keeps_collections() {
        let (dir, store, _) = indexed_store();
        assert!(store.get_stats().unwrap().chunk_count > 0);

        store
            .conn
            .execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?1, ?2)",
                params![FINGERPRINT_KEY, "schema=0;chunker=0;embedding=ancient"],
            )
            .unwrap();
        drop(store);

        let reopened = Store::open_at(dir.path().join("memory.sqlite")).unwrap();
        let stats = reopened.get_stats().unwrap();
        assert_eq!(stats.chunk_count, 0);
        assert_eq!(stats.file_count, 0);
        assert_eq!(reopened.get_embedding_stats().unwrap().0, 0);
        assert_eq!(reopened.list_collections().unwrap().len(), 1);
    }

    /// A broken database must not read as an empty one. Both stat calls used
    /// to `.unwrap_or(0)` their counts, so a corrupt index reported "0 files,
    /// 0 chunks, no embeddings" — byte for byte what a fresh install reports.
    /// `run_retrieval` then blamed the missing embeddings and quietly degraded
    /// every search to BM25.
    #[test]
    fn a_broken_database_errors_instead_of_reporting_an_empty_index() {
        let (_dir, store, _) = indexed_store();
        assert!(store.get_stats().unwrap().chunk_count > 0);

        store.conn.execute_batch("DROP TABLE chunks;").unwrap();

        assert!(store.get_stats().is_err());
        assert!(store.get_embedding_stats().is_err());
    }

    /// Hydration must propagate a database error rather than drop the hit.
    /// Dropping `collections.description` breaks only `get_chunk_by_id` — the
    /// ranking queries do not read it — so this reproduces the exact shape of
    /// the bug: ranked chunk ids that cannot be turned into results. The old
    /// `if let Ok(Some(..))` returned `Ok(vec![])` here, which the caller
    /// renders as "no matches in your vault".
    #[test]
    fn a_hydration_error_fails_the_search_instead_of_shortening_it() {
        let (_dir, store, embedding) = indexed_store();
        assert_eq!(
            hybrid_paths(&store, &embedding, &SearchOptions::default()).len(),
            2
        );

        store
            .conn
            .execute_batch("ALTER TABLE collections DROP COLUMN description;")
            .unwrap();

        assert!(store
            .search_hybrid_traced("quokka zebra", &embedding, 10, &SearchOptions::default())
            .is_err());
    }

    /// Re-indexing a file must take its embeddings with it, and must say so
    /// when it cannot. `index_file` deletes the file's chunks and inserts new
    /// ones, and SQLite hands the freed rowids straight back out — so a
    /// vector that survives the delete reattaches to a different chunk's text.
    /// `vec0` has no foreign keys, so nothing downstream catches that.
    ///
    /// Dropping `vec_embeddings` stands in for any failure of the delete: the
    /// swallowed `.ok()` let `index_file` report success and write the corrupt
    /// state anyway.
    #[test]
    fn a_failed_embedding_delete_fails_the_re_index() {
        let (dir, store, _) = indexed_store();
        let collection_id = store.list_collections().unwrap()[0].id;
        let note = dir.path().join("vault").join("2026-01-02.md");
        let note = note.to_string_lossy().to_string();

        std::fs::write(&note, "# New\n\nquokka zebra\n\n## Extra\n\nmore text\n").unwrap();
        store.index_file(collection_id, &note).unwrap();
        assert_eq!(store.orphan_counts().unwrap().embeddings, 0);

        store
            .conn
            .execute_batch("DROP TABLE vec_embeddings;")
            .unwrap();
        assert!(store.index_file(collection_id, &note).is_err());
    }

    #[test]
    fn hybrid_search_honors_the_after_filter() {
        let (_dir, store, embedding) = indexed_store();

        let all = hybrid_paths(&store, &embedding, &SearchOptions::default());
        assert!(all.iter().any(|p| p.ends_with("2020-01-02.md")));
        assert!(all.iter().any(|p| p.ends_with("2026-01-02.md")));

        let recent = hybrid_paths(
            &store,
            &embedding,
            &SearchOptions {
                after: Some("2025-01-01".to_string()),
                ..Default::default()
            },
        );
        assert!(!recent.is_empty());
        assert!(recent.iter().all(|p| p.ends_with("2026-01-02.md")));
    }

    /// `before` alone must bound the vector path too. It once did not: the
    /// prune ran behind an "is anything filtered?" check that had never been
    /// taught about `before`, so a `before`-only search returned unbounded
    /// KNN candidates. The check is gone; this pins the behaviour.
    #[test]
    fn hybrid_search_honors_a_lone_before_filter() {
        let (_dir, store, embedding) = indexed_store();

        let old = hybrid_paths(
            &store,
            &embedding,
            &SearchOptions {
                before: Some("2025-01-01".to_string()),
                ..Default::default()
            },
        );
        assert!(!old.is_empty());
        assert!(old.iter().all(|p| p.ends_with("2020-01-02.md")));
    }

    #[test]
    fn sanitize_splits_hyphenated_terms() {
        // FTS5 read `bge-small-en` as a column filter and aborted the
        // query with `no such column: small`.
        assert_eq!(
            sanitize_fts_query("bge-small-en"),
            "bge small en".to_string()
        );
        assert_eq!(sanitize_fts_query("half-life decay"), "half life decay");
    }

    #[test]
    fn sanitize_drops_fts5_operators_but_keeps_words() {
        assert_eq!(
            sanitize_fts_query("how does \"RRF\" work?"),
            "how does RRF work"
        );
        assert_eq!(sanitize_fts_query("section: Coffee"), "section Coffee");
        assert_eq!(sanitize_fts_query("snake_case stays"), "snake_case stays");
    }

    #[test]
    fn a_hyphenated_query_returns_results_instead_of_erroring() {
        let (_dir, store, _embedding) = indexed_store();
        let results = store
            .search_fts_filtered("quokka-zebra", 5, &SearchOptions::default())
            .expect("hyphenated query must not error");
        assert_eq!(results.len(), 2);
    }
}
