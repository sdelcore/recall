use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

/// SQLite-based memory store
pub struct Store {
    conn: Connection,
    db_path: PathBuf,
}

impl Store {
    /// Open or create the memory store
    pub fn open() -> Result<Self> {
        let db_path = Self::default_path()?;

        // Ensure parent directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create data directory")?;
        }

        let conn = Connection::open(&db_path)
            .context("Failed to open database")?;

        let store = Store { conn, db_path };
        store.init_schema()?;

        Ok(store)
    }

    /// Get the default database path
    fn default_path() -> Result<PathBuf> {
        let data_dir = dirs::data_local_dir()
            .context("Could not determine local data directory")?;
        Ok(data_dir.join("memory-search").join("memory.sqlite"))
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

            -- Embeddings storage (stored as JSON array for simplicity)
            -- Note: For production, consider sqlite-vec extension for better performance
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
        "#).context("Failed to initialize schema")?;

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
                c.project
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
            })
        })?;

        let mut search_results = Vec::new();
        for result in results {
            search_results.push(result?);
        }

        Ok(search_results)
    }

    /// Store embedding for a chunk
    pub fn store_embedding(&self, chunk_id: i64, embedding: &[f32]) -> Result<()> {
        // Serialize embedding as bytes (4 bytes per f32)
        let bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (chunk_id, embedding) VALUES (?1, ?2)",
            params![chunk_id, bytes],
        )?;

        Ok(())
    }

    /// Get embedding for a chunk
    #[allow(dead_code)]
    pub fn get_embedding(&self, chunk_id: i64) -> Result<Option<Vec<f32>>> {
        let result: Result<Vec<u8>, _> = self.conn.query_row(
            "SELECT embedding FROM embeddings WHERE chunk_id = ?1",
            params![chunk_id],
            |row| row.get(0),
        );

        match result {
            Ok(bytes) => {
                // Deserialize bytes to f32 array
                let embedding: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Ok(Some(embedding))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get all chunk IDs that don't have embeddings
    pub fn get_chunks_without_embeddings(&self) -> Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare(
            r#"SELECT c.id, c.content
               FROM chunks c
               LEFT JOIN embeddings e ON c.id = e.chunk_id
               WHERE e.chunk_id IS NULL"#,
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
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))
            .unwrap_or(0);

        Ok((embedded_chunks, total_chunks))
    }

    /// Vector search using cosine similarity
    pub fn search_vector(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(i64, f64)>> {
        // Get all embeddings and compute similarity
        let mut stmt = self.conn.prepare(
            "SELECT chunk_id, embedding FROM embeddings"
        )?;

        let results = stmt.query_map([], |row| {
            let chunk_id: i64 = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            Ok((chunk_id, bytes))
        })?;

        let mut scored: Vec<(i64, f64)> = Vec::new();

        for result in results {
            let (chunk_id, bytes) = result?;

            // Deserialize embedding
            let embedding: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            // Compute cosine similarity
            let similarity = crate::embedder::cosine_similarity(query_embedding, &embedding);
            scored.push((chunk_id, similarity as f64));
        }

        // Sort by similarity (descending) and take top N
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(scored)
    }

    /// Hybrid search combining BM25 and vector search
    pub fn search_hybrid(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
        vector_weight: f64,
        bm25_weight: f64,
    ) -> Result<Vec<SearchResult>> {
        // Get more candidates than needed for merging
        let candidate_count = limit * 3;

        // BM25 search
        let bm25_results = self.search_fts_filtered(query, candidate_count, &SearchOptions::default())?;

        // Vector search
        let vector_results = self.search_vector(query_embedding, candidate_count)?;

        // Normalize scores and merge
        let mut combined_scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();

        // Normalize and add BM25 scores
        if !bm25_results.is_empty() {
            let max_bm25 = bm25_results.iter().map(|r| r.score).fold(0.0_f64, f64::max);
            if max_bm25 > 0.0 {
                for result in bm25_results.iter() {
                    // Get chunk ID from position (we need to look it up)
                    if let Ok(chunk_id) = self.get_chunk_id_by_content(&result.content) {
                        let normalized = result.score / max_bm25;
                        *combined_scores.entry(chunk_id).or_insert(0.0) += normalized * bm25_weight;
                    }
                }
            }
        }

        // Normalize and add vector scores
        if !vector_results.is_empty() {
            let max_vector = vector_results.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);
            if max_vector > 0.0 {
                for (chunk_id, score) in &vector_results {
                    let normalized = score / max_vector;
                    *combined_scores.entry(*chunk_id).or_insert(0.0) += normalized * vector_weight;
                }
            }
        }

        // Sort by combined score
        let mut ranked: Vec<(i64, f64)> = combined_scores.into_iter().collect();
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

    /// Get chunk ID by content (helper for hybrid search)
    fn get_chunk_id_by_content(&self, content: &str) -> Result<i64> {
        let chunk_id: i64 = self.conn.query_row(
            "SELECT id FROM chunks WHERE content = ?1 LIMIT 1",
            params![content],
            |row| row.get(0),
        )?;
        Ok(chunk_id)
    }

    /// Get chunk by ID with score
    fn get_chunk_by_id(&self, chunk_id: i64, score: f64) -> Result<Option<SearchResult>> {
        let result = self.conn.query_row(
            r#"SELECT f.file_path, c.start_line, c.end_line, c.content, c.date, c.section, c.project
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
                })
            },
        );

        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
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

        // Begin transaction
        self.conn.execute("BEGIN", [])?;

        // Delete existing file entry if exists
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
                r#"INSERT INTO chunks (file_id, chunk_index, date, section, project, start_line, end_line, content)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"#,
                params![
                    file_id,
                    i as i64,
                    chunk.date,
                    chunk.section,
                    chunk.project,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.content,
                ],
            )?;
        }

        self.conn.execute("COMMIT", [])?;

        Ok(())
    }

    /// Full index of a directory
    pub fn index_full(&self, dir_path: &str) -> Result<()> {
        // Clear existing data
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
}

/// Chunk markdown content into sections
fn chunk_markdown(content: &str, file_path: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return chunks;
    }

    // Extract date from filename (YYYY-MM-DD.md pattern)
    let date = std::path::Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| s.len() == 10 && s.chars().nth(4) == Some('-') && s.chars().nth(7) == Some('-'))
        .map(|s| s.to_string());

    let mut current_section: Option<String> = None;
    let mut current_chunk_start = 0;
    let mut current_chunk_lines: Vec<&str> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        // Check for section headers
        if line.starts_with("## ") {
            // Save previous chunk if non-empty
            if !current_chunk_lines.is_empty() {
                let content = current_chunk_lines.join("\n");
                if !content.trim().is_empty() {
                    chunks.push(Chunk {
                        content,
                        start_line: (current_chunk_start + 1) as i64,
                        end_line: i as i64,
                        date: date.clone(),
                        section: current_section.clone(),
                        project: None, // TODO: Extract project tags
                    });
                }
            }

            // Start new section
            current_section = Some(line[3..].trim().to_string());
            current_chunk_start = i;
            current_chunk_lines = vec![*line];
        } else {
            current_chunk_lines.push(*line);

            // If chunk is getting too long, split it
            // Using a simple character count approximation for tokens
            let chunk_text = current_chunk_lines.join("\n");
            if chunk_text.len() > 1600 {
                // ~400 tokens
                let content = current_chunk_lines[..current_chunk_lines.len() - 1].join("\n");
                if !content.trim().is_empty() {
                    chunks.push(Chunk {
                        content,
                        start_line: (current_chunk_start + 1) as i64,
                        end_line: i as i64,
                        date: date.clone(),
                        section: current_section.clone(),
                        project: None,
                    });
                }
                current_chunk_start = i;
                current_chunk_lines = vec![*line];
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
            });
        }
    }

    chunks
}
