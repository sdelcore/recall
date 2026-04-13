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
}

/// SQLite-based memory store with sqlite-vec for vector search
pub struct Store {
    conn: Connection,
    db_path: PathBuf,
}

/// Register sqlite-vec extension (must be called before opening any connection)
fn register_sqlite_vec() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

impl Store {
    /// Open or create the memory store
    pub fn open() -> Result<Self> {
        register_sqlite_vec();

        let db_path = Self::default_path()?;

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create data directory")?;
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open database")?;

        // Enable foreign keys for CASCADE to work
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;

        let store = Store { conn, db_path };
        store.init_schema()?;
        store.migrate_embeddings()?;
        store.migrate_memory_type()?;

        Ok(store)
    }

    /// Get the default database path
    fn default_path() -> Result<PathBuf> {
        let data_dir = dirs::data_local_dir()
            .context("Could not determine local data directory")?;
        Ok(data_dir.join("recall").join("memory.sqlite"))
    }

    /// Get the database path
    pub fn path(&self) -> String {
        self.db_path.to_string_lossy().to_string()
    }

    /// Initialize the database schema
    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(r#"
            -- Metadata about indexed files
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                mtime INTEGER NOT NULL,
                indexed_at INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0
            );

            -- Text chunks with metadata
            CREATE TABLE IF NOT EXISTS chunks (
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
            CREATE INDEX IF NOT EXISTS idx_chunks_date ON chunks(date);
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
            self.conn.execute_batch(
                "CREATE VIRTUAL TABLE vec_embeddings USING vec0(embedding float[768]);"
            ).context("Failed to create vec_embeddings table")?;
        }

        Ok(())
    }

    /// Migrate from old BLOB-based embeddings table to vec0
    fn migrate_embeddings(&self) -> Result<()> {
        let has_old_table: bool = self.conn.query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='embeddings'",
            [],
            |row| row.get(0),
        ).unwrap_or(false);

        if !has_old_table {
            return Ok(());
        }

        let old_count: i64 = self.conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))
            .unwrap_or(0);

        if old_count == 0 {
            self.conn.execute_batch("DROP TABLE IF EXISTS embeddings;")?;
            return Ok(());
        }

        eprintln!("Migrating {} embeddings from BLOB to sqlite-vec format...", old_count);

        let mut stmt = self.conn.prepare(
            "SELECT chunk_id, embedding FROM embeddings"
        )?;

        let rows: Vec<(i64, Vec<u8>)> = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?.filter_map(|r| r.ok()).collect();

        for (chunk_id, bytes) in &rows {
            // Check if this chunk still exists (may have been orphaned)
            let chunk_exists: bool = self.conn.query_row(
                "SELECT COUNT(*) > 0 FROM chunks WHERE id = ?1",
                params![chunk_id],
                |row| row.get(0),
            ).unwrap_or(false);

            if !chunk_exists {
                continue;
            }

            // The old format is already le bytes of f32, which is what vec0 expects
            self.conn.execute(
                "INSERT OR REPLACE INTO vec_embeddings(rowid, embedding) VALUES (?1, ?2)",
                params![chunk_id, bytes],
            )?;
        }

        self.conn.execute_batch("DROP TABLE embeddings;")?;
        eprintln!("Migration complete.");

        Ok(())
    }

    /// Add memory_type column if not present
    fn migrate_memory_type(&self) -> Result<()> {
        let has_col: bool = self.conn
            .prepare("SELECT memory_type FROM chunks LIMIT 0")
            .is_ok();
        if !has_col {
            self.conn.execute_batch(
                "ALTER TABLE chunks ADD COLUMN memory_type TEXT;"
            )?;
            self.conn.execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_chunks_memory_type ON chunks(memory_type);"
            )?;
        }
        Ok(())
    }

    /// Get store statistics
    pub fn get_stats(&self) -> Result<StoreStats> {
        let file_count: i64 = self.conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
            .unwrap_or(0);

        let chunk_count: i64 = self.conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap_or(0);

        let last_indexed: Option<String> = self.conn
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

    /// Search using FTS5 (BM25) with filters
    pub fn search_fts_filtered(
        &self,
        query: &str,
        limit: usize,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        // Build dynamic query with filters
        let mut sql = String::from(r#"
            SELECT
                f.file_path,
                c.start_line,
                c.end_line,
                c.content,
                bm25(fts_chunks) as score,
                c.date,
                c.section,
                c.project,
                c.memory_type
            FROM fts_chunks
            JOIN chunks c ON c.id = fts_chunks.rowid
            JOIN files f ON f.id = c.file_id
            WHERE fts_chunks MATCH ?
        "#);

        // Build parameter list dynamically
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(query.to_string())];

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

        sql.push_str(" ORDER BY score LIMIT ?");
        params_vec.push(Box::new(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;

        // Convert params to references for rusqlite
        let params_refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

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
            })
        })?;

        let mut search_results = Vec::new();
        for result in results {
            search_results.push(result?);
        }

        Ok(search_results)
    }

    /// FTS5 search returning chunk IDs in BM25 rank order (for hybrid search)
    fn search_fts_chunk_ids(&self, query: &str, limit: usize) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(r#"
            SELECT c.id
            FROM fts_chunks
            JOIN chunks c ON c.id = fts_chunks.rowid
            WHERE fts_chunks MATCH ?1
            ORDER BY bm25(fts_chunks)
            LIMIT ?2
        "#)?;

        let rows = stmt.query_map(params![query, limit as i64], |row| {
            row.get::<_, i64>(0)
        })?;

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
        let total_chunks: i64 = self.conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap_or(0);

        let embedded_chunks: i64 = self.conn
            .query_row("SELECT COUNT(*) FROM vec_embeddings", [], |row| row.get(0))
            .unwrap_or(0);

        Ok((embedded_chunks, total_chunks))
    }

    /// Vector search using sqlite-vec KNN, returning chunk IDs in rank order
    fn search_vector_chunk_ids(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT rowid FROM vec_embeddings WHERE embedding MATCH ?1 AND k = ?2"
        )?;

        let rows = stmt.query_map(
            params![query_embedding.as_bytes(), limit as i64],
            |row| row.get::<_, i64>(0),
        )?;

        let mut ids = Vec::new();
        for row in rows {
            ids.push(row?);
        }
        Ok(ids)
    }

    /// Hybrid search combining BM25 and vector search using Reciprocal Rank Fusion
    pub fn search_hybrid(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
        rrf_k: u32,
    ) -> Result<Vec<SearchResult>> {
        // Get more candidates than needed for merging
        let candidate_count = limit * 3;
        let k = rrf_k as f64;

        // Get ranked lists from both search methods
        let bm25_ranked = self.search_fts_chunk_ids(query, candidate_count)?;
        let vector_ranked = self.search_vector_chunk_ids(query_embedding, candidate_count)?;

        // Reciprocal Rank Fusion: score(doc) = Σ 1/(k + rank)
        let mut rrf_scores: HashMap<i64, f64> = HashMap::new();

        for (rank, chunk_id) in bm25_ranked.iter().enumerate() {
            *rrf_scores.entry(*chunk_id).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        }

        for (rank, chunk_id) in vector_ranked.iter().enumerate() {
            *rrf_scores.entry(*chunk_id).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        }

        // Sort by RRF score descending
        let mut ranked: Vec<(i64, f64)> = rrf_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(limit);

        // Fetch full results
        let mut results = Vec::new();
        for (chunk_id, score) in ranked {
            if let Ok(Some(result)) = self.get_chunk_by_id(chunk_id, score) {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Get chunk by ID with score
    fn get_chunk_by_id(&self, chunk_id: i64, score: f64) -> Result<Option<SearchResult>> {
        let result = self.conn.query_row(
            r#"SELECT f.file_path, c.start_line, c.end_line, c.content, c.date, c.section, c.project, c.memory_type
               FROM chunks c
               JOIN files f ON f.id = c.file_id
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
                })
            },
        );

        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Delete vec_embeddings for chunks belonging to a file
    fn delete_embeddings_for_file(&self, file_path: &str) -> Result<()> {
        // Get chunk IDs for this file
        let mut stmt = self.conn.prepare(
            "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.file_path = ?1"
        )?;
        let chunk_ids: Vec<i64> = stmt.query_map(params![file_path], |row| {
            row.get::<_, i64>(0)
        })?.filter_map(|r| r.ok()).collect();

        for chunk_id in chunk_ids {
            self.conn.execute(
                "DELETE FROM vec_embeddings WHERE rowid = ?1",
                params![chunk_id],
            ).ok(); // Ignore errors for missing rows
        }
        Ok(())
    }

    /// Index a single file
    pub fn index_file(&self, file_path: &str) -> Result<()> {
        let path = std::path::Path::new(file_path);
        if !path.exists() {
            anyhow::bail!("File does not exist: {}", file_path);
        }

        let metadata = std::fs::metadata(path)?;
        let mtime = metadata.modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;

        let content = std::fs::read_to_string(path)?;
        let chunks = chunk_markdown(&content, file_path);

        // Clean up vec_embeddings before deleting chunks (no CASCADE for virtual tables)
        self.delete_embeddings_for_file(file_path)?;

        // Begin transaction
        self.conn.execute("BEGIN", [])?;

        // Delete existing file entry if exists (CASCADE deletes chunks and FTS)
        self.conn.execute("DELETE FROM files WHERE file_path = ?1", params![file_path])?;

        // Insert file record
        self.conn.execute(
            "INSERT INTO files (file_path, mtime, indexed_at, chunk_count) VALUES (?1, ?2, ?3, ?4)",
            params![file_path, mtime, Utc::now().timestamp(), chunks.len() as i64],
        )?;

        let file_id = self.conn.last_insert_rowid();

        // Insert chunks
        for (i, chunk) in chunks.iter().enumerate() {
            self.conn.execute(
                r#"INSERT INTO chunks (file_id, chunk_index, date, section, project, start_line, end_line, content, memory_type)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"#,
                params![
                    file_id,
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

    /// Full index of a directory
    pub fn index_full(&self, dir_path: &str) -> Result<()> {
        // Clear vec_embeddings first (no CASCADE)
        self.conn.execute("DELETE FROM vec_embeddings", [])?;
        // Clear chunks and files
        self.conn.execute("DELETE FROM chunks", [])?;
        self.conn.execute("DELETE FROM files", [])?;

        self.index_directory(dir_path)
    }

    /// Incremental index (only changed files)
    pub fn index_incremental(&self, dir_path: &str) -> Result<()> {
        self.index_directory(dir_path)
    }

    /// Index all markdown files in a directory
    fn index_directory(&self, dir_path: &str) -> Result<()> {
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

            // Skip excluded patterns
            let should_skip = exclude_patterns.iter().any(|pattern| {
                glob::Pattern::new(pattern)
                    .map(|p| p.matches(&path_str))
                    .unwrap_or(false)
            });

            if should_skip {
                continue;
            }

            // Check if file needs re-indexing
            let metadata = std::fs::metadata(&path)?;
            let mtime = metadata.modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs() as i64;

            let needs_index = self.conn
                .query_row(
                    "SELECT mtime FROM files WHERE file_path = ?1",
                    params![&path_str],
                    |row| row.get::<_, i64>(0),
                )
                .map(|stored_mtime| stored_mtime < mtime)
                .unwrap_or(true);

            if needs_index {
                if let Err(e) = self.index_file(&path_str) {
                    eprintln!("Warning: Failed to index {}: {}", path_str, e);
                }
            }
        }

        Ok(())
    }
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

/// Number of overlap characters when splitting at size boundary (~15% of 1600)
const CHUNK_OVERLAP_CHARS: usize = 240;
/// Maximum chunk size in characters (~400 tokens)
const MAX_CHUNK_CHARS: usize = 1600;

/// Chunk markdown content into sections with overlap at size boundaries
fn chunk_markdown(content: &str, file_path: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return chunks;
    }

    let memory_type = classify_memory_type(file_path);

    // Extract date from filename (YYYY-MM-DD.md pattern)
    let date = std::path::Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| s.len() == 10 && s.chars().nth(4) == Some('-') && s.chars().nth(7) == Some('-'))
        .map(|s| s.to_string());

    let mut current_section: Option<String> = None;
    let mut current_chunk_start = 0;
    let mut current_chunk_lines: Vec<&str> = Vec::new();
    // Lines to prepend as overlap from previous size-split chunk
    let mut overlap_lines: Vec<&str> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        // Check for section headers
        if line.starts_with("## ") {
            // Save previous chunk if non-empty (no overlap at header boundaries)
            if !current_chunk_lines.is_empty() {
                let content = current_chunk_lines.join("\n");
                if !content.trim().is_empty() {
                    chunks.push(Chunk {
                        content,
                        start_line: (current_chunk_start + 1) as i64,
                        end_line: i as i64,
                        date: date.clone(),
                        section: current_section.clone(),
                        project: None,
                        memory_type: memory_type.clone(),
                    });
                }
            }

            // Start new section — clear overlap since header is a semantic boundary
            current_section = Some(line[3..].trim().to_string());
            current_chunk_start = i;
            current_chunk_lines = vec![*line];
            overlap_lines.clear();
        } else {
            current_chunk_lines.push(*line);

            // If chunk is getting too long, split it
            let chunk_text = current_chunk_lines.join("\n");
            if chunk_text.len() > MAX_CHUNK_CHARS {
                let split_lines = &current_chunk_lines[..current_chunk_lines.len() - 1];
                let content = split_lines.join("\n");
                if !content.trim().is_empty() {
                    chunks.push(Chunk {
                        content,
                        start_line: (current_chunk_start + 1) as i64,
                        end_line: i as i64,
                        date: date.clone(),
                        section: current_section.clone(),
                        project: None,
                        memory_type: memory_type.clone(),
                    });
                }

                // Compute overlap: take lines from the end of the emitted chunk
                // that total ~CHUNK_OVERLAP_CHARS characters
                overlap_lines.clear();
                let mut overlap_len = 0;
                for &ol in split_lines.iter().rev() {
                    overlap_len += ol.len() + 1; // +1 for newline
                    overlap_lines.push(ol);
                    if overlap_len >= CHUNK_OVERLAP_CHARS {
                        break;
                    }
                }
                overlap_lines.reverse();

                // Start new chunk with overlap + current line
                current_chunk_start = i.saturating_sub(overlap_lines.len());
                current_chunk_lines = overlap_lines.clone();
                current_chunk_lines.push(*line);
                overlap_lines.clear();
            }
        }
    }

    // Save final chunk
    if !current_chunk_lines.is_empty() {
        let content = current_chunk_lines.join("\n");
        if !content.trim().is_empty() {
            chunks.push(Chunk {
                content,
                start_line: (current_chunk_start + 1) as i64,
                end_line: lines.len() as i64,
                date: date.clone(),
                section: current_section,
                project: None,
                memory_type: memory_type.clone(),
            });
        }
    }

    chunks
}
