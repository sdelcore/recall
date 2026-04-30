//! Turn a markdown file into the chunk records that get persisted.
//!
//! Wraps the structural AST chunker (`crate::ast`) with file-level metadata
//! enrichment: a date extracted from `YYYY-MM-DD` filenames, and a memory
//! type classified from path patterns. The persistence layer calls
//! [`chunk_file`] and writes the resulting [`Chunk`]s; the heuristics here
//! are kept private and unit-tested in isolation.

/// Maximum chunk size in characters (~400 tokens). Soft cap — chunks split
/// only at AST block boundaries, never mid-block (code, list, table).
const MAX_CHUNK_CHARS: usize = 1600;

/// A chunk of text ready to persist. Date / project / memory_type are
/// file-level metadata stamped onto every chunk produced from a file.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub start_line: i64,
    pub end_line: i64,
    pub date: Option<String>,
    pub section: Option<String>,
    pub project: Option<String>,
    pub memory_type: Option<String>,
}

/// Chunk a markdown file's content into persistable records, stamped with
/// file-level metadata. Single entry point for the indexer.
pub fn chunk_file(content: &str, file_path: &str) -> Vec<Chunk> {
    let memory_type = classify_memory_type(file_path);
    let date = extract_date_from_filename(file_path);

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

    #[test]
    fn chunk_file_stamps_date_and_memory_type_on_every_chunk() {
        let content = "# Heading One\n\nSome content.\n\n## Heading Two\n\nMore content.\n";
        let chunks = chunk_file(content, "/vault/Daily Notes/2026-04-29.md");

        assert!(!chunks.is_empty(), "should produce at least one chunk");
        for c in &chunks {
            assert_eq!(c.date.as_deref(), Some("2026-04-29"));
            assert_eq!(c.memory_type.as_deref(), Some("episodic"));
            assert_eq!(c.project, None);
        }
    }

    #[test]
    fn chunk_file_handles_unmatched_paths_with_no_metadata() {
        let content = "# Heading\n\nContent.\n";
        let chunks = chunk_file(content, "/vault/projects/foo.md");

        assert!(!chunks.is_empty());
        for c in &chunks {
            assert_eq!(c.date, None);
            assert_eq!(c.memory_type, None);
        }
    }

    #[test]
    fn chunk_file_returns_empty_for_empty_content() {
        let chunks = chunk_file("", "/vault/notes/empty.md");
        assert!(chunks.is_empty());
    }
}
