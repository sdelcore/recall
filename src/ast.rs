//! Markdown AST-based chunker.
//!
//! Parses a markdown document with comrak and groups top-level blocks into
//! chunks split on heading boundaries (any level) and a soft size cap.
//! Unlike the previous line-based splitter, this never tears a code block,
//! list, or table — chunks always end on a block boundary.

use comrak::nodes::{AstNode, NodeValue};
use comrak::{parse_document, Arena, ComrakOptions};

/// A chunk produced by the AST chunker. Date / project / memory_type are
/// file-level concerns added by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawChunk {
    pub content: String,
    /// 1-based inclusive line of first source line in the chunk
    pub start_line: i64,
    /// 1-based inclusive line of last source line in the chunk
    pub end_line: i64,
    /// Most recent heading at any level when this chunk was emitted
    pub section: Option<String>,
}

/// Chunk markdown content along block boundaries with a soft max-char cap.
///
/// Splits on:
/// - any heading (resets section to that heading's text)
/// - cumulative size exceeding `max_chars` (only at block boundaries)
pub fn chunk_markdown_ast(content: &str, max_chars: usize) -> Vec<RawChunk> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    let arena = Arena::new();
    let opts = ComrakOptions::default();
    let root = parse_document(&arena, content, &opts);
    let lines: Vec<&str> = content.lines().collect();

    let mut out = Vec::new();
    let mut current_section: Option<String> = None;
    let mut buf: Vec<&str> = Vec::new();
    let mut buf_start: Option<usize> = None;
    let mut buf_end: usize = 0;
    let mut buf_chars: usize = 0;

    let flush = |out: &mut Vec<RawChunk>,
                 buf: &mut Vec<&str>,
                 buf_start: &mut Option<usize>,
                 buf_end: &mut usize,
                 buf_chars: &mut usize,
                 section: &Option<String>| {
        if buf.is_empty() {
            return;
        }
        let content = buf.join("\n");
        if !content.trim().is_empty() {
            out.push(RawChunk {
                content,
                start_line: buf_start.unwrap_or(1) as i64,
                end_line: *buf_end as i64,
                section: section.clone(),
            });
        }
        buf.clear();
        *buf_start = None;
        *buf_end = 0;
        *buf_chars = 0;
    };

    for child in root.children() {
        let data = child.data.borrow();
        let pos = data.sourcepos;
        let start = pos.start.line; // 1-based
        let end = pos.end.line.max(start);

        // Block source slice
        let block_lines = &lines[start.saturating_sub(1)..end.min(lines.len())];
        let block_chars: usize = block_lines.iter().map(|l| l.len() + 1).sum();

        let is_heading = matches!(data.value, NodeValue::Heading(_));

        if is_heading {
            // Heading: flush current, then start a new chunk that begins with
            // the heading line and update the running section name.
            flush(
                &mut out,
                &mut buf,
                &mut buf_start,
                &mut buf_end,
                &mut buf_chars,
                &current_section,
            );
            current_section =
                Some(extract_text(child).trim().to_string()).filter(|s| !s.is_empty());
            buf.extend_from_slice(block_lines);
            buf_start = Some(start);
            buf_end = end;
            buf_chars = block_chars;
            continue;
        }

        // Non-heading: would adding this overflow? If yes and buf non-empty, flush first.
        if !buf.is_empty() && buf_chars + block_chars > max_chars {
            flush(
                &mut out,
                &mut buf,
                &mut buf_start,
                &mut buf_end,
                &mut buf_chars,
                &current_section,
            );
        }

        if buf.is_empty() {
            buf_start = Some(start);
        }
        buf.extend_from_slice(block_lines);
        buf_end = end;
        buf_chars += block_chars;
    }

    flush(
        &mut out,
        &mut buf,
        &mut buf_start,
        &mut buf_end,
        &mut buf_chars,
        &current_section,
    );

    out
}

/// Recursively collect text content from a node's descendants.
fn extract_text<'a>(node: &'a AstNode<'a>) -> String {
    let mut s = String::new();
    collect(node, &mut s);
    s
}

fn collect<'a>(node: &'a AstNode<'a>, out: &mut String) {
    let data = node.data.borrow();
    match &data.value {
        NodeValue::Text(t) => out.push_str(t),
        NodeValue::Code(c) => out.push_str(&c.literal),
        _ => {}
    }
    for child in node.children() {
        collect(child, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_no_chunks() {
        assert!(chunk_markdown_ast("", 1000).is_empty());
        assert!(chunk_markdown_ast("   \n  \n", 1000).is_empty());
    }

    #[test]
    fn single_paragraph_is_one_chunk() {
        let chunks = chunk_markdown_ast("Hello world.\n", 1000);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("Hello world"));
        assert_eq!(chunks[0].section, None);
    }

    #[test]
    fn headings_split_chunks_and_set_section() {
        let md = "# Top\n\nIntro paragraph.\n\n## Coffee\n\nEspresso details.\n\n## Tea\n\nMatcha details.\n";
        let chunks = chunk_markdown_ast(md, 10_000);
        assert_eq!(chunks.len(), 3, "got {chunks:#?}");
        assert_eq!(chunks[0].section.as_deref(), Some("Top"));
        assert_eq!(chunks[1].section.as_deref(), Some("Coffee"));
        assert_eq!(chunks[2].section.as_deref(), Some("Tea"));
    }

    #[test]
    fn code_block_is_not_split_mid_block() {
        // Code block alone is bigger than max_chars: it must still come out as one chunk.
        let big_code: String = (0..50)
            .map(|i| format!("line-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let md = format!("```\n{big_code}\n```\n");
        let chunks = chunk_markdown_ast(&md, 50);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("line-0"));
        assert!(chunks[0].content.contains("line-49"));
    }

    #[test]
    fn size_cap_splits_between_paragraphs() {
        let para = "x".repeat(120);
        let md = format!("{para}\n\n{para}\n\n{para}\n");
        let chunks = chunk_markdown_ast(&md, 150);
        assert!(
            chunks.len() >= 2,
            "expected size-based split; got {chunks:#?}"
        );
        // No chunk fragment should be empty
        for c in &chunks {
            assert!(!c.content.trim().is_empty());
        }
    }

    #[test]
    fn section_persists_across_paragraphs_within_section() {
        let md = "## Coffee\n\nFirst paragraph.\n\nSecond paragraph.\n";
        let chunks = chunk_markdown_ast(md, 10_000);
        assert!(chunks
            .iter()
            .all(|c| c.section.as_deref() == Some("Coffee")));
    }

    #[test]
    fn line_numbers_are_one_based_and_cover_block() {
        let md = "Line 1\n\n# Heading\n\nBody line.\n";
        let chunks = chunk_markdown_ast(md, 10_000);
        // First chunk: paragraph at line 1
        assert_eq!(chunks[0].start_line, 1);
        // Heading chunk starts at line 3
        assert_eq!(chunks[1].start_line, 3);
        assert!(chunks[1].end_line >= 5);
    }

    #[test]
    fn frontmatter_ignored_at_text_level() {
        // comrak treats YAML frontmatter as paragraphs by default; we don't
        // do anything special with it but we shouldn't drop it.
        let md = "---\ntitle: Test\n---\n\n# Hello\n\nBody.\n";
        let chunks = chunk_markdown_ast(md, 10_000);
        // Either 1 or 2 chunks depending on parsing, but never 0.
        assert!(!chunks.is_empty());
    }

    #[test]
    fn list_stays_intact() {
        let md = "Some intro.\n\n- one\n- two\n- three\n\nTrailing line.\n";
        let chunks = chunk_markdown_ast(md, 10_000);
        // The list is one block — the bullet text should appear together.
        let has_full_list = chunks.iter().any(|c| {
            c.content.contains("- one")
                && c.content.contains("- two")
                && c.content.contains("- three")
        });
        assert!(has_full_list, "got {chunks:#?}");
    }
}
