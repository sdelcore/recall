//! Minimal YAML frontmatter scanner for markdown notes.
//!
//! Recall only needs a handful of flat scalars out of a note's frontmatter —
//! the dates that drive recency ranking, plus `status` and `type` which are
//! returned as metadata. A full YAML parser would be a large dependency (and
//! `serde_yaml` is unmaintained) for a job that is a dozen lines of scanning,
//! so this module hand-rolls it.
//!
//! Deliberately narrow: it reads a *leading* `---` block, top-level scalar
//! keys only, and silently ignores anything nested or unrecognized. Malformed
//! input yields empty fields rather than an error — frontmatter is
//! opportunistic metadata, never a reason to fail an index run.

/// The frontmatter fields recall cares about. Everything else in the block is
/// ignored. Date-shaped fields are normalized to `YYYY-MM-DD` (a full ISO
/// datetime is truncated); a value that is not date-shaped becomes `None`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Frontmatter {
    pub date: Option<String>,
    pub last_updated: Option<String>,
    pub created: Option<String>,
    pub updated: Option<String>,
    pub status: Option<String>,
    /// The `type:` key. Named `doc_type` because `type` is a Rust keyword.
    pub doc_type: Option<String>,
    pub aliases: Vec<String>,
}

/// Scan a markdown document's leading frontmatter block. Returns an empty
/// [`Frontmatter`] when there is no block, the block is unterminated, or none
/// of the recognized keys are present.
pub fn parse(content: &str) -> Frontmatter {
    let mut fm = Frontmatter::default();
    let Some(block) = extract_block(content) else {
        return fm;
    };

    // Tracks the most recent top-level key so a following `- item` block list
    // can be attached to it. Only `aliases` consumes list items.
    let mut current_key: Option<String> = None;

    for line in block.lines() {
        let body = line.trim();
        if body.is_empty() || body.starts_with('#') {
            continue;
        }

        if let Some(item) = list_item(body) {
            if current_key.as_deref() == Some("aliases") {
                let value = unquote(item);
                if !value.is_empty() {
                    fm.aliases.push(value);
                }
            }
            continue;
        }

        // Indented non-list lines belong to a nested structure we don't read.
        if line.starts_with(' ') || line.starts_with('\t') {
            continue;
        }

        let Some((key, value)) = body.split_once(':') else {
            current_key = None;
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim();
        current_key = Some(key.clone());
        if value.is_empty() {
            continue;
        }

        match key.as_str() {
            "date" => fm.date = normalize_date(value),
            "last_updated" => fm.last_updated = normalize_date(value),
            "created" => fm.created = normalize_date(value),
            "updated" => fm.updated = normalize_date(value),
            "status" => fm.status = non_empty(unquote(value)),
            "type" => fm.doc_type = non_empty(unquote(value)),
            "aliases" => fm.aliases = parse_inline_list(value),
            _ => {}
        }
    }

    fm
}

/// Return the text between a leading `---` line and the next `---` line.
/// `None` when the document does not open with `---` or the block never
/// closes (an unterminated block is body text, not frontmatter).
fn extract_block(content: &str) -> Option<&str> {
    let mut lines = content.split_inclusive('\n');
    let first = lines.next()?;
    if first.trim_end() != "---" {
        return None;
    }
    let start = first.len();
    let mut end = start;
    for line in lines {
        if line.trim_end() == "---" {
            return Some(&content[start..end]);
        }
        end += line.len();
    }
    None
}

/// `- value` → `Some("value")`, for both `- x` and a bare `-`.
fn list_item(body: &str) -> Option<&str> {
    let rest = body.strip_prefix('-')?;
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        Some(rest.trim())
    } else {
        None
    }
}

/// `[a, b]` → `["a", "b"]`. A bare scalar is treated as a one-item list.
fn parse_inline_list(value: &str) -> Vec<String> {
    let inner = match value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
        Some(inner) => inner,
        None => return non_empty(unquote(value)).into_iter().collect(),
    };
    inner
        .split(',')
        .filter_map(|item| non_empty(unquote(item.trim())))
        .collect()
}

/// Truncate a date-shaped value to `YYYY-MM-DD`. Anything that does not start
/// with that shape is rejected outright so a garbled value cannot poison the
/// recency ranking.
fn normalize_date(value: &str) -> Option<String> {
    let value = unquote(value);
    let b = value.as_bytes();
    let digits = |range: &[u8]| range.iter().all(u8::is_ascii_digit);
    let shaped = b.len() >= 10
        && digits(&b[0..4])
        && b[4] == b'-'
        && digits(&b[5..7])
        && b[7] == b'-'
        && digits(&b[8..10]);
    shaped.then(|| value[..10].to_string())
}

/// Strip one layer of matching surrounding quotes.
fn unquote(value: &str) -> String {
    let v = value.trim();
    for q in ['"', '\''] {
        if v.len() >= 2 && v.starts_with(q) && v.ends_with(q) {
            return v[1..v.len() - 1].to_string();
        }
    }
    v.to_string()
}

fn non_empty(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_block() {
        let md = "---\ndate: 2026-04-29\nstatus: active\ntype: session\n---\n\n# Body\n";
        let fm = parse(md);
        assert_eq!(fm.date.as_deref(), Some("2026-04-29"));
        assert_eq!(fm.status.as_deref(), Some("active"));
        assert_eq!(fm.doc_type.as_deref(), Some("session"));
    }

    #[test]
    fn missing_block_yields_empty() {
        assert_eq!(parse("# Just a heading\n\nBody.\n"), Frontmatter::default());
        // A `---` that is not the first line is a horizontal rule, not frontmatter.
        assert_eq!(
            parse("# Heading\n\n---\ndate: 2026-04-29\n---\n"),
            Frontmatter::default()
        );
    }

    #[test]
    fn unterminated_block_yields_empty() {
        assert_eq!(
            parse("---\ndate: 2026-04-29\n\nBody.\n"),
            Frontmatter::default()
        );
    }

    #[test]
    fn malformed_lines_are_skipped_without_panic() {
        let md = "---\nnot a pair\n: leading colon\ndate:\nstatus: ok\n---\n";
        let fm = parse(md);
        assert_eq!(fm.date, None);
        assert_eq!(fm.status.as_deref(), Some("ok"));
    }

    #[test]
    fn empty_and_delimiter_only_input_is_safe() {
        assert_eq!(parse(""), Frontmatter::default());
        assert_eq!(parse("---\n---\n"), Frontmatter::default());
        assert_eq!(parse("---"), Frontmatter::default());
    }

    #[test]
    fn strips_surrounding_quotes() {
        let fm = parse("---\ndate: \"2026-04-29\"\nstatus: 'in progress'\n---\n");
        assert_eq!(fm.date.as_deref(), Some("2026-04-29"));
        assert_eq!(fm.status.as_deref(), Some("in progress"));
    }

    #[test]
    fn truncates_iso_datetime_to_date() {
        let fm = parse("---\nlast_updated: 2026-07-31T00:06:49Z\n---\n");
        assert_eq!(fm.last_updated.as_deref(), Some("2026-07-31"));
    }

    #[test]
    fn rejects_non_date_shaped_values() {
        let fm = parse("---\ndate: yesterday\ncreated: 2026-4-29\nupdated: 2026-04-29\n---\n");
        assert_eq!(fm.date, None);
        assert_eq!(fm.created, None);
        assert_eq!(fm.updated.as_deref(), Some("2026-04-29"));
    }

    #[test]
    fn reads_inline_alias_list() {
        let fm = parse("---\naliases: [Alpha, \"Beta Note\", ]\n---\n");
        assert_eq!(fm.aliases, vec!["Alpha", "Beta Note"]);
    }

    #[test]
    fn reads_block_alias_list() {
        let fm = parse("---\naliases:\n  - Alpha\n  - \"Beta Note\"\nstatus: done\n---\n");
        assert_eq!(fm.aliases, vec!["Alpha", "Beta Note"]);
        assert_eq!(fm.status.as_deref(), Some("done"));
    }

    #[test]
    fn block_list_items_do_not_leak_across_keys() {
        let fm = parse("---\naliases:\n  - Alpha\ntags:\n  - not-an-alias\n---\n");
        assert_eq!(fm.aliases, vec!["Alpha"]);
    }

    #[test]
    fn nested_structures_are_ignored() {
        let md = "---\nmeta:\n  date: 2020-01-01\n  status: nested\nstatus: top\n---\n";
        let fm = parse(md);
        assert_eq!(fm.date, None);
        assert_eq!(fm.status.as_deref(), Some("top"));
    }

    #[test]
    fn keys_are_case_insensitive_and_comments_skipped() {
        let fm = parse("---\n# a comment\nStatus: Draft\nDATE: 2026-01-02\n---\n");
        assert_eq!(fm.status.as_deref(), Some("Draft"));
        assert_eq!(fm.date.as_deref(), Some("2026-01-02"));
    }

    #[test]
    fn handles_crlf_line_endings() {
        let fm = parse("---\r\ndate: 2026-04-29\r\n---\r\n");
        assert_eq!(fm.date.as_deref(), Some("2026-04-29"));
    }
}
