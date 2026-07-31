//! Vault link linter: dangling wikilinks and orphaned notes.
//!
//! Recall indexes several Obsidian vaults as separate collections, and a link
//! from one vault to another is deliberate, not a mistake. Off-the-shelf vault
//! linters assume a single vault, so they report every cross-project link as
//! broken. This module therefore resolves a link against *all* registered
//! collections and reports three states — `resolved-local`, `resolved-foreign`,
//! and `unresolved`. Only `unresolved` is a finding.
//!
//! Two other things it does differently from the usual regex-over-raw-text
//! linter:
//!
//! - Links are extracted from the comrak AST, not the raw bytes. Code blocks,
//!   inline code, and `%%Obsidian comments%%` never contribute links, because
//!   a `[[Target]]` written inside a fenced block is documentation, not a link.
//! - Resolution matches note basenames *and* frontmatter `aliases:`. A resolver
//!   that ignores aliases reports the vault's most-linked notes as dangling.
//!
//! The linter reads the filesystem, not the index, so its answer does not go
//! stale when the index does. It never mutates anything and never runs as part
//! of indexing: an index that aborts on a link typo is a self-inflicted outage.

use anyhow::Result;
use comrak::nodes::{AstNode, NodeValue};
use comrak::{parse_document, Arena, ComrakOptions};
use glob::MatchOptions;
use serde::Serialize;
use std::collections::HashMap;

use crate::config::Config;
use crate::frontmatter;
use crate::store::Collection;

/// How a wikilink resolved against the set of registered collections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LinkState {
    /// Target found in the same collection as the note that links to it.
    ResolvedLocal,
    /// Target found in a different collection. Valid, reported for visibility.
    ResolvedForeign,
    /// Target found nowhere. The only finding.
    Unresolved,
}

/// One wikilink occurrence.
#[derive(Debug, Clone, Serialize)]
pub struct Link {
    pub collection: String,
    pub file: String,
    pub line: u32,
    /// The link target after stripping `|alias` and `#heading`.
    pub target: String,
    pub state: LinkState,
    /// Collection holding the target, when `state` is `resolved-foreign`.
    pub resolved_in: Option<String>,
}

/// The result of a lint run. Counts cover every scanned note; the vectors hold
/// only what a human needs to act on.
#[derive(Debug, Default, Serialize)]
pub struct LintReport {
    pub notes_scanned: usize,
    pub links_total: usize,
    pub resolved_local: usize,
    pub foreign: Vec<Link>,
    pub unresolved: Vec<Link>,
    /// Notes with zero incoming and zero outgoing wikilinks, minus the
    /// configured `[lint] orphan_exclude` globs.
    pub orphans: Vec<String>,
}

/// A scanned note plus the links it contains.
struct Note {
    collection: String,
    path: String,
    /// Lowercased basename and aliases — every name this note answers to.
    keys: Vec<String>,
    links: Vec<(String, u32)>,
    incoming: usize,
}

/// Lint every collection, reporting findings for `only` (or for all when it is
/// `None`). Notes outside `only` are still scanned, because a link is only
/// unresolved when *no* collection holds the target and an orphan is only an
/// orphan when *no* collection links to it.
pub fn lint(config: &Config, collections: &[Collection], only: Option<&str>) -> Result<LintReport> {
    let mut notes = Vec::new();
    for collection in collections {
        if collection.root_path.is_empty() {
            continue;
        }
        scan_collection(config, collection, &mut notes)?;
    }

    // name -> the notes answering to it. A name is rarely ambiguous, but two
    // vaults holding a `README.md` each is normal, so this is a multimap.
    let mut by_key: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, note) in notes.iter().enumerate() {
        for key in &note.keys {
            by_key.entry(key.as_str()).or_default().push(idx);
        }
    }

    let mut report = LintReport::default();
    // Resolve first, then apply: `incoming` is a mutation of `notes`, which is
    // borrowed while resolving.
    let mut resolutions: Vec<(Option<usize>, Link)> = Vec::new();

    for note in &notes {
        for (target, line) in &note.links {
            let candidates = by_key
                .get(target.to_lowercase().as_str())
                .map(Vec::as_slice)
                .unwrap_or_default();
            let local = candidates
                .iter()
                .copied()
                .find(|&i| notes[i].collection == note.collection);
            let (state, hit) = match (local, candidates.first().copied()) {
                (Some(i), _) => (LinkState::ResolvedLocal, Some(i)),
                (None, Some(i)) => (LinkState::ResolvedForeign, Some(i)),
                (None, None) => (LinkState::Unresolved, None),
            };
            resolutions.push((
                hit,
                Link {
                    collection: note.collection.clone(),
                    file: note.path.clone(),
                    line: *line,
                    target: target.clone(),
                    state,
                    resolved_in: match state {
                        LinkState::ResolvedForeign => hit.map(|i| notes[i].collection.clone()),
                        _ => None,
                    },
                },
            ));
        }
    }

    for (hit, link) in resolutions {
        if let Some(hit) = hit {
            notes[hit].incoming += 1;
        }
        if !selected(&link.collection, only) {
            continue;
        }
        report.links_total += 1;
        match link.state {
            LinkState::ResolvedLocal => report.resolved_local += 1,
            LinkState::ResolvedForeign => report.foreign.push(link),
            LinkState::Unresolved => report.unresolved.push(link),
        }
    }

    let orphan_exclude = compile_globs(&config.lint.orphan_exclude);
    for note in &notes {
        if !selected(&note.collection, only) {
            continue;
        }
        report.notes_scanned += 1;
        let orphan = note.links.is_empty() && note.incoming == 0;
        if orphan && !matches_any(&orphan_exclude, &note.path) {
            report.orphans.push(note.path.clone());
        }
    }

    report.orphans.sort();
    Ok(report)
}

fn selected(collection: &str, only: Option<&str>) -> bool {
    match only {
        Some(name) => name == collection,
        None => true,
    }
}

/// Walk one collection's root for markdown files, honoring `[index] exclude`
/// so the linter sees exactly the notes the indexer sees.
fn scan_collection(config: &Config, collection: &Collection, out: &mut Vec<Note>) -> Result<()> {
    let exclude = compile_globs(&config.index.exclude);
    let pattern = format!("{}/**/*.md", collection.root_path);
    for entry in glob::glob(&pattern)? {
        let path = entry?;
        let path_str = path.to_string_lossy().to_string();
        if matches_any(&exclude, &path_str) || path_str.contains(".sync-conflict-") {
            continue;
        }
        // A note we cannot read is a warning, never a failed run.
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Warning: skipping {}: {}", path.display(), e);
                continue;
            }
        };
        let fm = frontmatter::parse(&content);

        let mut keys = Vec::new();
        if let Some(stem) = path.file_stem() {
            keys.push(stem.to_string_lossy().to_lowercase());
        }
        keys.extend(fm.aliases.iter().map(|a| a.to_lowercase()));

        out.push(Note {
            collection: collection.name.clone(),
            path: path_str,
            keys,
            links: extract_wikilinks(&content),
            incoming: 0,
        });
    }
    Ok(())
}

fn compile_globs(patterns: &[String]) -> Vec<glob::Pattern> {
    patterns
        .iter()
        .filter_map(|p| glob::Pattern::new(p).ok())
        .collect()
}

/// Glob matching is case-insensitive: `Daily/` and `daily/` are the same
/// folder to a user writing an exclusion pattern.
fn matches_any(patterns: &[glob::Pattern], path: &str) -> bool {
    let opts = MatchOptions {
        case_sensitive: false,
        ..MatchOptions::new()
    };
    patterns.iter().any(|p| p.matches_with(path, opts))
}

/// Extract `[[Target]]` targets and their 1-based source lines.
///
/// Wikilinks are not markdown, so comrak leaves them as literal text. We use
/// the AST anyway for what it *excludes*: text inside code blocks, inline code,
/// raw HTML, and frontmatter never reaches the scanner.
fn extract_wikilinks(content: &str) -> Vec<(String, u32)> {
    let arena = Arena::new();
    let mut opts = ComrakOptions::default();
    opts.extension.front_matter_delimiter = Some("---".to_string());
    let root = parse_document(&arena, content, &opts);

    let mut text = String::new();
    let mut lines: Vec<u32> = Vec::new();
    collect_text(root, &mut text, &mut lines);
    scan(&text, &lines)
}

fn collect_text<'a>(node: &'a AstNode<'a>, text: &mut String, lines: &mut Vec<u32>) {
    let data = node.data.borrow();
    let line = data.sourcepos.start.line as u32;
    match &data.value {
        NodeValue::Text(t) => push(text, lines, t, line),
        NodeValue::SoftBreak | NodeValue::LineBreak => push(text, lines, "\n", line),
        // Verbatim content: a link written here is a sample, not a link. The
        // newline is a barrier so an unbalanced `[[` cannot span the gap.
        NodeValue::Code(_)
        | NodeValue::CodeBlock(_)
        | NodeValue::HtmlBlock(_)
        | NodeValue::HtmlInline(_)
        | NodeValue::FrontMatter(_) => {
            push(text, lines, "\n", line);
            return;
        }
        _ => {}
    }
    for child in node.children() {
        collect_text(child, text, lines);
    }
    if data.value.block() {
        push(text, lines, "\n", line);
    }
}

/// Append `s` to the text stream, recording the source line for every byte so
/// a match can be reported at the line the user has to open.
fn push(text: &mut String, lines: &mut Vec<u32>, s: &str, line: u32) {
    text.push_str(s);
    lines.resize(text.len(), line);
}

/// Scan the extracted text for wikilinks, skipping `%%comment%%` spans.
///
/// Byte-wise is safe: every delimiter is ASCII, so a slice boundary can never
/// land inside a multi-byte character.
fn scan(text: &str, lines: &[u32]) -> Vec<(String, u32)> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut in_comment = false;
    let mut i = 0;

    while i + 1 < b.len() {
        if b[i] == b'%' && b[i + 1] == b'%' {
            in_comment = !in_comment;
            i += 2;
            continue;
        }
        if in_comment || b[i] != b'[' || b[i + 1] != b'[' {
            i += 1;
            continue;
        }
        // A wikilink never spans a line, so a newline ends the search.
        let mut j = i + 2;
        let close = loop {
            if j + 1 >= b.len() || b[j] == b'\n' {
                break None;
            }
            if b[j] == b']' && b[j + 1] == b']' {
                break Some(j);
            }
            j += 1;
        };
        match close {
            Some(close) => {
                if let Some(target) = normalize_target(&text[i + 2..close]) {
                    out.push((target, lines.get(i).copied().unwrap_or(1)));
                }
                i = close + 2;
            }
            None => i += 2,
        }
    }
    out
}

/// `Folder/Note.md#Heading|alias` → `Note`. Returns `None` for a target that
/// names no note — an empty link or a bare `[[#Heading]]` self-reference.
fn normalize_target(raw: &str) -> Option<String> {
    let target = raw
        .split('|')
        .next()
        .unwrap_or_default()
        .split('#')
        .next()
        .unwrap_or_default()
        .trim();
    let target = target.rsplit('/').next().unwrap_or_default();
    let target = target.strip_suffix(".md").unwrap_or(target).trim();
    (!target.is_empty()).then(|| target.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets(md: &str) -> Vec<String> {
        extract_wikilinks(md).into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn extracts_plain_alias_and_heading_forms() {
        let md = "See [[Alpha]], [[Beta|the beta note]] and [[Gamma#Section]].\n";
        assert_eq!(targets(md), ["Alpha", "Beta", "Gamma"]);
    }

    #[test]
    fn strips_folder_path_and_md_suffix() {
        let md = "Refer to [[notes/Deep/Alpha.md]] and [[Beta.md#Top|b]].\n";
        assert_eq!(targets(md), ["Alpha", "Beta"]);
    }

    #[test]
    fn ignores_links_in_fenced_and_inline_code() {
        let md = "Real [[Alpha]].\n\n```\nfake [[Beta]]\n```\n\nAlso `[[Gamma]]` inline.\n";
        assert_eq!(targets(md), ["Alpha"]);
    }

    #[test]
    fn ignores_links_in_obsidian_comments() {
        let md = "Real [[Alpha]] %% hidden [[Beta]] %% and [[Gamma]].\n";
        assert_eq!(targets(md), ["Alpha", "Gamma"]);
    }

    #[test]
    fn ignores_links_in_frontmatter() {
        let md = "---\nrelated: \"[[Alpha]]\"\n---\n\nBody links [[Beta]].\n";
        assert_eq!(targets(md), ["Beta"]);
    }

    #[test]
    fn skips_self_heading_and_empty_targets() {
        assert!(targets("Jump to [[#Section]] or [[]] or [[|x]].\n").is_empty());
    }

    #[test]
    fn reports_the_source_line() {
        let md = "# Title\n\nFirst line.\n\nA link to [[Alpha]] here.\n";
        assert_eq!(extract_wikilinks(md), [("Alpha".to_string(), 5)]);
    }

    #[test]
    fn unterminated_link_does_not_swallow_the_document() {
        let md = "Broken [[Alpha and then\n\n[[Beta]] later.\n";
        assert_eq!(targets(md), ["Beta"]);
    }
}
