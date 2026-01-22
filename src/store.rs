use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
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

    /// Search using FTS5 (BM25)
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let mut stmt = self.conn.prepare(r#"
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
            WHERE fts_chunks MATCH ?1
            ORDER BY score
            LIMIT ?2
        "#)?;

        let results = stmt.query_map(params![query, limit as i64], |row| {
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
