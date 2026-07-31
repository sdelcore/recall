//! Turn a markdown file into the chunk records that get persisted.
//!
//! Wraps the structural AST chunker (`crate::ast`) with file-level metadata
//! enrichment: the chunk's date, a memory type, and the note's `status`. The
//! persistence layer calls [`chunk_file`] and writes the resulting
//! [`Chunk`]s; the heuristics here are kept private and unit-tested in
//! isolation.
//!
//! The date matters more than the other fields because recency ranking is
//! built on it, so it is resolved through an explicit cascade — frontmatter,
//! then a `YYYY-MM-DD` filename, then the file's mtime — and the rung that
//! won is recorded in `date_source`. Without that provenance a low-confidence
//! mtime date is indistinguishable from an author-declared one.

use crate::frontmatter::Frontmatter;

/// Maximum chunk size in characters (~400 tokens). Soft cap — chunks split
/// only at AST block boundaries, never mid-block (code, list, table).
const MAX_CHUNK_CHARS: usize = 1600;

/// A chunk of text ready to persist. Everything but `content` / line numbers
/// / `section` is file-level metadata stamped onto every chunk from a file.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub start_line: i64,
    pub end_line: i64,
    pub date: Option<String>,
    /// Which rung of the date cascade produced `date`:
    /// `"frontmatter"`, `"filename"`, or `"mtime"`.
    pub date_source: Option<String>,
    pub section: Option<String>,
    pub project: Option<String>,
    pub memory_type: Option<String>,
    /// Frontmatter `status:`, carried through verbatim. Recorded and
    /// returned, never used to filter or exclude results.
    pub status: Option<String>,
}

/// Chunk a markdown file's content into persistable records, stamped with
/// file-level metadata. Single entry point for the indexer. `mtime` is the
/// file's modification time in unix seconds — the last rung of the date
/// cascade.
pub fn chunk_file(content: &str, file_path: &str, mtime: i64) -> Vec<Chunk> {
    let frontmatter = crate::frontmatter::parse(content);
    let (date, date_source) = resolve_date(&frontmatter, file_path, mtime);
    let memory_type = resolve_memory_type(&frontmatter, file_path);
    let status = frontmatter.status;

    crate::ast::chunk_markdown_ast(content, MAX_CHUNK_CHARS)
        .into_iter()
        .map(|raw| Chunk {
            content: raw.content,
            start_line: raw.start_line,
            end_line: raw.end_line,
            section: raw.section,
            date: date.clone(),
            date_source: date_source.clone(),
            project: None,
            memory_type: memory_type.clone(),
            status: status.clone(),
        })
        .collect()
}

/// Resolve a file's date and record which rung of the cascade won:
/// frontmatter `date:` / `last_updated:` → `YYYY-MM-DD` filename → mtime.
/// The mtime rung always answers, so the result is `None` only when the
/// timestamp itself is out of range.
fn resolve_date(
    frontmatter: &Frontmatter,
    file_path: &str,
    mtime: i64,
) -> (Option<String>, Option<String>) {
    if let Some(date) = frontmatter
        .date
        .clone()
        .or_else(|| frontmatter.last_updated.clone())
    {
        return (Some(date), Some("frontmatter".to_string()));
    }
    if let Some(date) = extract_date_from_filename(file_path) {
        return (Some(date), Some("filename".to_string()));
    }
    match chrono::DateTime::from_timestamp(mtime, 0) {
        Some(dt) => (
            Some(dt.format("%Y-%m-%d").to_string()),
            Some("mtime".to_string()),
        ),
        None => (None, None),
    }
}

/// Frontmatter `type:` wins over the path heuristics; an unrecognized type
/// falls through to them rather than blanking the classification.
fn resolve_memory_type(frontmatter: &Frontmatter, file_path: &str) -> Option<String> {
    frontmatter
        .doc_type
        .as_deref()
        .and_then(memory_type_from_frontmatter)
        .or_else(|| classify_memory_type(file_path))
}

/// Map a note's declared `type:` onto a memory type.
fn memory_type_from_frontmatter(doc_type: &str) -> Option<String> {
    match doc_type.trim().to_ascii_lowercase().as_str() {
        "session" | "scratchpad" => Some("episodic".to_string()),
        "project" => Some("semantic".to_string()),
        _ => None,
    }
}

/// Pull a date out of filenames matching `YYYY-MM-DD.*` (daily notes).
/// Other filenames return None — date is opportunistic, not required.
fn extract_date_from_filename(file_path: &str) -> Option<String> {
    std::path::Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| s.len() == 10 && s.chars().nth(4) == Some('-') && s.chars().nth(7) == Some('-'))
        .map(|s| s.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_date_pulls_iso_dated_filenames() {
        assert_eq!(
            extract_date_from_filename("/vault/Daily Notes/2026-04-29.md"),
            Some("2026-04-29".to_string())
        );
    }

    #[test]
    fn extract_date_returns_none_for_non_dated_filenames() {
        assert_eq!(extract_date_from_filename("/vault/notes/projects.md"), None);
        assert_eq!(extract_date_from_filename("/vault/2026-04.md"), None); // too short
        assert_eq!(extract_date_from_filename("/vault/20260429.md"), None); // no dashes
    }

    #[test]
    fn extract_date_ignores_dashes_in_wrong_positions() {
        // Length 10 but dashes not at 4 and 7 → reject.
        assert_eq!(extract_date_from_filename("/vault/abc-def-gh.md"), None);
    }

    #[test]
    fn classify_skill_directory() {
        assert_eq!(
            classify_memory_type("/home/u/Obsidian/aria/skills/coding.md"),
            Some("skill".to_string())
        );
    }

    #[test]
    fn classify_aria_core_files() {
        assert_eq!(
            classify_memory_type("/Obsidian/aria/memory.md"),
            Some("semantic".to_string())
        );
        assert_eq!(
            classify_memory_type("/anywhere/soul.md"),
            Some("semantic".to_string())
        );
        assert_eq!(
            classify_memory_type("/anywhere/user.md"),
            Some("semantic".to_string())
        );
        assert_eq!(
            classify_memory_type("/Obsidian/aria/issues.md"),
            Some("procedural".to_string())
        );
    }

    #[test]
    fn classify_daily_notes() {
        assert_eq!(
            classify_memory_type("/vault/Daily Notes/2026-04-29.md"),
            Some("episodic".to_string())
        );
        assert_eq!(
            classify_memory_type("/vault/periodic/daily/2026-04-29.md"),
            Some("episodic".to_string())
        );
    }

    #[test]
    fn classify_aria_messages_and_contacts() {
        assert_eq!(
            classify_memory_type("/vault/aria/messages/2026-04-29.md"),
            Some("episodic".to_string())
        );
        assert_eq!(
            classify_memory_type("/vault/aria/contacts/alice.md"),
            Some("semantic".to_string())
        );
    }

    #[test]
    fn classify_unmatched_paths_return_none() {
        assert_eq!(classify_memory_type("/vault/projects/foo.md"), None);
        assert_eq!(classify_memory_type("/vault/random.md"), None);
    }

    #[test]
    fn classify_is_case_insensitive() {
        assert_eq!(
            classify_memory_type("/Vault/ARIA/Skills/Coding.md"),
            Some("skill".to_string())
        );
    }

    /// 2026-04-29T00:00:00Z — a fixed mtime so tests never depend on the clock.
    const MTIME: i64 = 1_777_420_800;

    fn fm(yaml: &str) -> Frontmatter {
        crate::frontmatter::parse(&format!("---\n{yaml}\n---\n"))
    }

    #[test]
    fn date_cascade_prefers_frontmatter() {
        let (date, source) = resolve_date(
            &fm("date: 2020-01-02"),
            "/vault/Daily Notes/2026-04-29.md",
            MTIME,
        );
        assert_eq!(date.as_deref(), Some("2020-01-02"));
        assert_eq!(source.as_deref(), Some("frontmatter"));
    }

    #[test]
    fn date_cascade_accepts_last_updated_as_frontmatter_date() {
        let (date, source) = resolve_date(
            &fm("last_updated: 2026-07-31T00:06:49Z"),
            "/vault/notes/foo.md",
            MTIME,
        );
        assert_eq!(date.as_deref(), Some("2026-07-31"));
        assert_eq!(source.as_deref(), Some("frontmatter"));
    }

    #[test]
    fn date_cascade_falls_back_to_filename() {
        let (date, source) = resolve_date(
            &Frontmatter::default(),
            "/vault/Daily Notes/2026-04-29.md",
            MTIME,
        );
        assert_eq!(date.as_deref(), Some("2026-04-29"));
        assert_eq!(source.as_deref(), Some("filename"));
    }

    #[test]
    fn date_cascade_falls_back_to_mtime() {
        let (date, source) = resolve_date(&Frontmatter::default(), "/vault/notes/foo.md", MTIME);
        assert_eq!(date.as_deref(), Some("2026-04-29"));
        assert_eq!(source.as_deref(), Some("mtime"));
    }

    #[test]
    fn memory_type_maps_frontmatter_types() {
        assert_eq!(
            memory_type_from_frontmatter("session"),
            Some("episodic".to_string())
        );
        assert_eq!(
            memory_type_from_frontmatter("Scratchpad"),
            Some("episodic".to_string())
        );
        assert_eq!(
            memory_type_from_frontmatter("project"),
            Some("semantic".to_string())
        );
        assert_eq!(memory_type_from_frontmatter("unknown"), None);
    }

    #[test]
    fn memory_type_frontmatter_overrides_path_heuristic() {
        assert_eq!(
            resolve_memory_type(&fm("type: project"), "/vault/Daily Notes/2026-04-29.md"),
            Some("semantic".to_string())
        );
    }

    #[test]
    fn memory_type_falls_back_to_path_heuristic() {
        // No frontmatter at all, and an unrecognized type, both fall through.
        assert_eq!(
            resolve_memory_type(&Frontmatter::default(), "/vault/Daily Notes/2026-04-29.md"),
            Some("episodic".to_string())
        );
        assert_eq!(
            resolve_memory_type(&fm("type: meeting"), "/vault/Daily Notes/2026-04-29.md"),
            Some("episodic".to_string())
        );
        assert_eq!(
            resolve_memory_type(&fm("type: meeting"), "/vault/projects/foo.md"),
            None
        );
    }

    #[test]
    fn chunk_file_stamps_date_and_memory_type_on_every_chunk() {
        let content = "# Heading One\n\nSome content.\n\n## Heading Two\n\nMore content.\n";
        let chunks = chunk_file(content, "/vault/Daily Notes/2026-04-29.md", MTIME);

        assert!(!chunks.is_empty(), "should produce at least one chunk");
        for c in &chunks {
            assert_eq!(c.date.as_deref(), Some("2026-04-29"));
            assert_eq!(c.date_source.as_deref(), Some("filename"));
            assert_eq!(c.memory_type.as_deref(), Some("episodic"));
            assert_eq!(c.project, None);
            assert_eq!(c.status, None);
        }
    }

    #[test]
    fn chunk_file_stamps_frontmatter_metadata_on_every_chunk() {
        let content = "---\ndate: 2026-01-02\nstatus: active\ntype: project\n---\n\n\
                       # Heading One\n\nSome content.\n\n## Heading Two\n\nMore.\n";
        let chunks = chunk_file(content, "/vault/Daily Notes/2026-04-29.md", MTIME);

        assert!(chunks.len() > 1, "should produce several chunks");
        for c in &chunks {
            assert_eq!(c.date.as_deref(), Some("2026-01-02"));
            assert_eq!(c.date_source.as_deref(), Some("frontmatter"));
            assert_eq!(c.memory_type.as_deref(), Some("semantic"));
            assert_eq!(c.status.as_deref(), Some("active"));
        }
    }

    #[test]
    fn chunk_file_dates_undated_files_from_mtime() {
        let content = "# Heading\n\nContent.\n";
        let chunks = chunk_file(content, "/vault/projects/foo.md", MTIME);

        assert!(!chunks.is_empty());
        for c in &chunks {
            assert_eq!(c.date.as_deref(), Some("2026-04-29"));
            assert_eq!(c.date_source.as_deref(), Some("mtime"));
            assert_eq!(c.memory_type, None);
        }
    }

    #[test]
    fn chunk_file_returns_empty_for_empty_content() {
        let chunks = chunk_file("", "/vault/notes/empty.md", MTIME);
        assert!(chunks.is_empty());
    }
}
