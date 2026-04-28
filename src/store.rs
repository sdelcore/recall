use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{ffi::sqlite3_auto_extension, params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use zerocopy::AsBytes;

/// Search result from the memory store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub content: String,
    pub score: f64,
    pub date: Option<String>,
    pub section: Option<String>,
    pub project: Option<String>,
    pub memory_type: Option<String>,
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
    pub project: Option<String>,
    pub file_pattern: Option<String>,
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
}

/// SQLite-based memory store with sqlite-vec for vector search
pub struct Store {
    conn: Connection,
    db_path: PathBuf,
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
        register_sqlite_vec();

        let db_path = Self::default_path()?;

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create data directory")?;
        }

        let conn = Connection::open(&db_path).context("Failed to open database")?;

        // Enable foreign keys for CASCADE to work
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;

        let store = Store { conn, db_path };
        store.check_schema_compat()?;
        store.init_schema()?;

        Ok(store)
    }

    /// Refuse to operate on a pre-collections DB so we never silently
    /// double-index or corrupt rows. No fallback by design.
    fn check_schema_compat(&self) -> Result<()> {
        let chunks_exists: bool = self
            .conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='chunks'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(false);
        if !chunks_exists {
            return Ok(());
        }
        let has_col = self
            .conn
            .prepare("SELECT collection_id FROM chunks LIMIT 0")
            .is_ok();
        if has_col {
            return Ok(());
        }
        let chunk_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap_or(0);
        if chunk_count > 0 {
            anyhow::bail!(
                "Schema upgrade required (collections support added): existing 'chunks' table \
                 has {} rows and lacks the collection_id column. Delete {} and re-index. \
                 No backwards-compat shim by design.",
                chunk_count,
                self.db_path.display()
            );
        }
        // Empty old schema — drop so init_schema can recreate with new columns.
        self.conn
            .execute_batch("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS files;")?;
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
                chunk_index INTEGER NOT NULL,
                date TEXT,
                section TEXT,
                project TEXT,
                memory_type TEXT,
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
            CREATE INDEX IF NOT EXISTS idx_chunks_date ON chunks(date);
            CREATE INDEX IF NOT EXISTS idx_chunks_memory_type ON chunks(memory_type);
            CREATE INDEX IF NOT EXISTS idx_files_collection_id ON files(collection_id);
            CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);
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
            self.conn
                .execute_batch(
                    "CREATE VIRTUAL TABLE vec_embeddings USING vec0(embedding float[768]);",
                )
                .context("Failed to create vec_embeddings table")?;
        }

        Ok(())
    }

    /// Get store statistics
    pub fn get_stats(&self) -> Result<StoreStats> {
        let file_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
            .unwrap_or(0);

        let chunk_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap_or(0);

        let last_indexed: Option<String> = self
            .conn
            .query_row(
                "SELECT datetime(MAX(indexed_at), 'unixepoch') FROM files",
                [],
                |row| row.get(0),
            )
            .ok();

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
                c.section,
                c.project,
                c.memory_type,
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

        // Add date filter
        if let Some(after) = &options.after {
            sql.push_str(" AND c.date >= ?");
            params_vec.push(Box::new(after.clone()));
        }

        // Add project filter (search in section name)
        if let Some(project) = &options.project {
            sql.push_str(" AND c.section LIKE ?");
            params_vec.push(Box::new(format!("%{}%", project)));
        }

        // Add file pattern filter
        if let Some(file_pattern) = &options.file_pattern {
            sql.push_str(" AND f.file_path LIKE ?");
            // Convert glob to SQL LIKE pattern
            let pattern = file_pattern.replace('*', "%").replace('?', "_");
            params_vec.push(Box::new(pattern));
        }

        // Add collection filter
        if let Some(cid) = options.collection_id {
            sql.push_str(" AND c.collection_id = ?");
            params_vec.push(Box::new(cid));
        }

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
                section: row.get(6)?,
                project: row.get(7)?,
                memory_type: row.get(8)?,
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

    /// FTS5 search returning chunk IDs in BM25 rank order (for hybrid search)
    fn search_fts_chunk_ids(
        &self,
        query: &str,
        limit: usize,
        collection_id: Option<i64>,
    ) -> Result<Vec<i64>> {
        let mut sql = String::from(
            r#"
            SELECT c.id
            FROM fts_chunks
            JOIN chunks c ON c.id = fts_chunks.rowid
            WHERE fts_chunks MATCH ?1
        "#,
        );
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> =
            vec![Box::new(sanitize_fts_query(query)), Box::new(limit as i64)];
        if let Some(cid) = collection_id {
            sql.push_str(" AND c.collection_id = ?3");
            params_vec.push(Box::new(cid));
        }
        sql.push_str(" ORDER BY bm25(fts_chunks) LIMIT ?2");

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

    /// Get all chunk IDs that don't have embeddings
    pub fn get_chunks_without_embeddings(&self) -> Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare(
            r#"SELECT c.id, c.content
               FROM chunks c
               WHERE c.id NOT IN (SELECT rowid FROM vec_embeddings)"#,
        )?;

        let results = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut chunks = Vec::new();
        for result in results {
            chunks.push(result?);
        }

        Ok(chunks)
    }

    /// Get embedding statistics
    pub fn get_embedding_stats(&self) -> Result<(i64, i64)> {
        let total_chunks: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap_or(0);

        let embedded_chunks: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM vec_embeddings", [], |row| row.get(0))
            .unwrap_or(0);

        Ok((embedded_chunks, total_chunks))
    }

    /// Vector search using sqlite-vec KNN, returning chunk IDs in rank order.
    /// When `collection_id` is set, runs KNN larger then filters down so the
    /// final list still contains `limit` items from the requested collection.
    fn search_vector_chunk_ids(
        &self,
        query_embedding: &[f32],
        limit: usize,
        collection_id: Option<i64>,
    ) -> Result<Vec<i64>> {
        let knn_k = if collection_id.is_some() {
            limit * 4
        } else {
            limit
        };
        let mut stmt = self
            .conn
            .prepare("SELECT rowid FROM vec_embeddings WHERE embedding MATCH ?1 AND k = ?2")?;
        let rows = stmt.query_map(params![query_embedding.as_bytes(), knn_k as i64], |row| {
            row.get::<_, i64>(0)
        })?;
        let mut ids: Vec<i64> = Vec::new();
        for row in rows {
            ids.push(row?);
        }

        if let Some(cid) = collection_id {
            // Filter to the requested collection without losing rank order.
            let mut filtered = Vec::with_capacity(limit);
            let placeholders = (0..ids.len())
                .map(|i| format!("?{}", i + 2))
                .collect::<Vec<_>>()
                .join(",");
            if ids.is_empty() {
                return Ok(filtered);
            }
            let sql = format!(
                "SELECT id FROM chunks WHERE collection_id = ?1 AND id IN ({})",
                placeholders
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(cid)];
            for id in &ids {
                params_vec.push(Box::new(*id));
            }
            let params_refs: Vec<&dyn rusqlite::ToSql> =
                params_vec.iter().map(|p| p.as_ref()).collect();
            let allowed: std::collections::HashSet<i64> = stmt
                .query_map(params_refs.as_slice(), |row| row.get::<_, i64>(0))?
                .filter_map(|r| r.ok())
                .collect();
            for id in ids {
                if allowed.contains(&id) {
                    filtered.push(id);
                    if filtered.len() == limit {
                        break;
                    }
                }
            }
            Ok(filtered)
        } else {
            Ok(ids)
        }
    }

    /// Hybrid search combining BM25 and vector search using Reciprocal Rank Fusion
    pub fn search_hybrid(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
        rrf_k: u32,
        collection_id: Option<i64>,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .search_hybrid_traced(query, query_embedding, limit, rrf_k, collection_id)?
            .into_iter()
            .map(|(r, _t)| r)
            .collect())
    }

    /// Same as [`search_hybrid`] but also returns per-result trace info for
    /// `--trace`: BM25 rank, vector rank, fused RRF score.
    pub fn search_hybrid_traced(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
        rrf_k: u32,
        collection_id: Option<i64>,
    ) -> Result<Vec<(SearchResult, SearchTrace)>> {
        let candidate_count = limit * 3;
        let k = rrf_k as f64;

        let bm25_ranked = self.search_fts_chunk_ids(query, candidate_count, collection_id)?;
        let vector_ranked =
            self.search_vector_chunk_ids(query_embedding, candidate_count, collection_id)?;

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

        let mut out = Vec::new();
        for (chunk_id, rrf_score) in ranked {
            if let Ok(Some(result)) = self.get_chunk_by_id(chunk_id, rrf_score) {
                let trace = SearchTrace {
                    bm25_rank: bm25_idx.get(&chunk_id).copied(),
                    vec_rank: vec_idx.get(&chunk_id).copied(),
                    rrf_score,
                    rerank_score: None,
                };
                out.push((result, trace));
            }
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
        rrf_k: u32,
        options: &SearchOptions,
    ) -> Result<Vec<(SearchResult, SearchTrace)>> {
        let results = self.search_fts_filtered(query, limit, options)?;
        let k = rrf_k as f64;
        Ok(results
            .into_iter()
            .enumerate()
            .map(|(rank, r)| {
                let trace = SearchTrace {
                    bm25_rank: Some(rank),
                    vec_rank: None,
                    rrf_score: 1.0 / (k + rank as f64 + 1.0),
                    rerank_score: None,
                };
                (r, trace)
            })
            .collect())
    }

    /// Get chunk by ID with score
    fn get_chunk_by_id(&self, chunk_id: i64, score: f64) -> Result<Option<SearchResult>> {
        let result = self.conn.query_row(
            r#"SELECT f.file_path, c.start_line, c.end_line, c.content,
                      c.date, c.section, c.project, c.memory_type,
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
                    section: row.get(5)?,
                    project: row.get(6)?,
                    memory_type: row.get(7)?,
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
    fn delete_embeddings_for_file(&self, collection_id: i64, file_path: &str) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id \
             WHERE f.collection_id = ?1 AND f.file_path = ?2",
        )?;
        let chunk_ids: Vec<i64> = stmt
            .query_map(params![collection_id, file_path], |row| {
                row.get::<_, i64>(0)
            })?
            .filter_map(|r| r.ok())
            .collect();

        for chunk_id in chunk_ids {
            self.conn
                .execute(
                    "DELETE FROM vec_embeddings WHERE rowid = ?1",
                    params![chunk_id],
                )
                .ok();
        }
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
        let chunks = chunk_markdown(&content, file_path);

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

        for (i, chunk) in chunks.iter().enumerate() {
            self.conn.execute(
                r#"INSERT INTO chunks (file_id, collection_id, chunk_index, date, section, project, start_line, end_line, content, memory_type)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"#,
                params![
                    file_id,
                    collection_id,
                    i as i64,
                    chunk.date,
                    chunk.section,
                    chunk.project,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.content,
                    chunk.memory_type,
                ],
            )?;
        }

        self.conn.execute("COMMIT", [])?;
        Ok(())
    }

    /// Full re-index of a directory into a collection (drops all of that
    /// collection's existing rows first, then walks the tree).
    pub fn index_full(&self, collection_id: i64, dir_path: &str) -> Result<()> {
        // Drop embeddings for chunks in this collection
        self.conn.execute(
            "DELETE FROM vec_embeddings WHERE rowid IN \
             (SELECT id FROM chunks WHERE collection_id = ?1)",
            params![collection_id],
        )?;
        self.conn.execute(
            "DELETE FROM chunks WHERE collection_id = ?1",
            params![collection_id],
        )?;
        self.conn.execute(
            "DELETE FROM files WHERE collection_id = ?1",
            params![collection_id],
        )?;
        self.index_directory(collection_id, dir_path)
    }

    /// Incremental index (only changed files) into the given collection.
    pub fn index_incremental(&self, collection_id: i64, dir_path: &str) -> Result<()> {
        self.index_directory(collection_id, dir_path)
    }

    /// Index all markdown files in a directory into a collection
    fn index_directory(&self, collection_id: i64, dir_path: &str) -> Result<()> {
        let pattern = format!("{}/**/*.md", dir_path);
        let exclude_patterns = [
            "**/Templates/**",
            "**/.obsidian/**",
            "**/attachments/**",
            "**/*.sync-conflict-*",
        ];

        for entry in glob::glob(&pattern)? {
            let path = entry?;
            let path_str = path.to_string_lossy().to_string();

            let should_skip = exclude_patterns.iter().any(|pattern| {
                glob::Pattern::new(pattern)
                    .map(|p| p.matches(&path_str))
                    .unwrap_or(false)
            });
            if should_skip {
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
            created_at: now,
        })
    }

    pub fn list_collections(&self) -> Result<Vec<Collection>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, root_path, description, created_at FROM collections ORDER BY name",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Collection {
                id: row.get(0)?,
                name: row.get(1)?,
                root_path: row.get(2)?,
                description: row.get(3)?,
                created_at: row.get(4)?,
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
            "SELECT id, name, root_path, description, created_at FROM collections WHERE name = ?1",
            params![name],
            |row| {
                Ok(Collection {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    root_path: row.get(2)?,
                    description: row.get(3)?,
                    created_at: row.get(4)?,
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

/// A chunk of text with metadata
#[derive(Debug, Clone)]
struct Chunk {
    content: String,
    start_line: i64,
    end_line: i64,
    date: Option<String>,
    section: Option<String>,
    project: Option<String>,
    memory_type: Option<String>,
}

/// Classify a file into a memory type based on its path.
/// Returns: "semantic", "procedural", "episodic", "skill", or None for general content.
/// Sanitize a free-text query into something FTS5 will accept as a MATCH
/// expression. Drops FTS5 operators (`?`, `:`, `"`, parentheses, etc.) so
/// natural-language queries from users / classifiers don't blow up with
/// `fts5: syntax error`. Tokens are rejoined with spaces — FTS5 treats
/// multiple unquoted tokens as an implicit AND-of-OR over its tokenizer.
fn sanitize_fts_query(query: &str) -> String {
    let cleaned: String = query
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn classify_memory_type(file_path: &str) -> Option<String> {
    let path_lower = file_path.to_lowercase();

    // Skills directory
    if path_lower.contains("/aria/skills/") {
        return Some("skill".to_string());
    }

    // ARIA core files
    if path_lower.ends_with("/memory.md") && path_lower.contains("/aria/") {
        return Some("semantic".to_string());
    }
    if path_lower.ends_with("/soul.md") || path_lower.ends_with("/user.md") {
        return Some("semantic".to_string());
    }
    if path_lower.ends_with("/issues.md") && path_lower.contains("/aria/") {
        return Some("procedural".to_string());
    }

    // Daily notes (both user and ARIA)
    if path_lower.contains("/daily notes/") || path_lower.contains("/periodic/daily/") {
        return Some("episodic".to_string());
    }

    // Messages
    if path_lower.contains("/aria/messages/") {
        return Some("episodic".to_string());
    }

    // Contacts
    if path_lower.contains("/aria/contacts/") {
        return Some("semantic".to_string());
    }

    None
}

/// Maximum chunk size in characters (~400 tokens). Soft cap — chunks split
/// only at AST block boundaries, never mid-block (code, list, table).
const MAX_CHUNK_CHARS: usize = 1600;

/// Chunk markdown content along block boundaries via the AST chunker, then
/// stamp file-level metadata (date from filename, memory type from path).
fn chunk_markdown(content: &str, file_path: &str) -> Vec<Chunk> {
    let memory_type = classify_memory_type(file_path);
    let date = std::path::Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| s.len() == 10 && s.chars().nth(4) == Some('-') && s.chars().nth(7) == Some('-'))
        .map(|s| s.to_string());

    crate::ast::chunk_markdown_ast(content, MAX_CHUNK_CHARS)
        .into_iter()
        .map(|raw| Chunk {
            content: raw.content,
            start_line: raw.start_line,
            end_line: raw.end_line,
            section: raw.section,
            date: date.clone(),
            project: None,
            memory_type: memory_type.clone(),
        })
        .collect()
}
