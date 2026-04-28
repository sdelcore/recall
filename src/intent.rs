//! Heuristic query intent classifier.
//!
//! Maps a raw query string to one of four buckets that downstream search
//! code can use to set sensible defaults:
//!
//! - `Lookup`: short, specific terms — BM25 is plenty.
//! - `Exploratory`: long / natural-language / question-shaped — hybrid +
//!   reranker pays off.
//! - `Temporal`: query references a date or time window — caller can
//!   auto-set the `after:` filter from the extracted year.
//! - `Structural`: query references a file path / section / glob —
//!   caller can route to file_pattern / section filters.
//!
//! Pure-heuristic by design (no LLM call) so search latency stays
//! predictable. The classifier never *changes* behavior on its own; it
//! only suggests parameter routing the caller can opt into.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Intent {
    #[default]
    Lookup,
    Exploratory,
    Temporal,
    Structural,
}

impl Intent {
    pub fn as_str(self) -> &'static str {
        match self {
            Intent::Lookup => "lookup",
            Intent::Exploratory => "exploratory",
            Intent::Temporal => "temporal",
            Intent::Structural => "structural",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Classified {
    pub intent: Intent,
    /// Year extracted from the query, if any (e.g. `Some(2025)` for "in 2025")
    pub year: Option<i32>,
}

/// Classify a query into an `Intent`. Order of checks matters: structural
/// and temporal hints win over surface length, because they map to harder
/// filter routing the caller probably wants applied.
pub fn classify(query: &str) -> Classified {
    let q = query.trim();
    if q.is_empty() {
        return Classified::default();
    }
    let lower = q.to_ascii_lowercase();

    // 1. Structural: explicit field prefix or glob characters.
    if has_structural_marker(&lower) {
        return Classified {
            intent: Intent::Structural,
            year: extract_year(&lower),
        };
    }

    // 2. Temporal: a date / year / relative time word.
    if let Some(year) = extract_year(&lower) {
        return Classified {
            intent: Intent::Temporal,
            year: Some(year),
        };
    }
    if has_relative_time(&lower) {
        return Classified {
            intent: Intent::Temporal,
            year: None,
        };
    }

    // 3. Exploratory vs Lookup by surface shape.
    let token_count = q.split_whitespace().count();
    if q.contains('?') || token_count >= 5 || starts_with_question_word(&lower) {
        return Classified {
            intent: Intent::Exploratory,
            year: None,
        };
    }

    Classified {
        intent: Intent::Lookup,
        year: None,
    }
}

fn has_structural_marker(lower: &str) -> bool {
    lower.contains('*')
        || lower.starts_with("file:")
        || lower.starts_with("section:")
        || lower.starts_with("path:")
        || lower.contains(".md")
}

fn has_relative_time(lower: &str) -> bool {
    const HINTS: &[&str] = &[
        "yesterday",
        "today",
        "last week",
        "last month",
        "last year",
        "this week",
        "this month",
        "this year",
        "recently",
    ];
    HINTS.iter().any(|h| lower.contains(h))
}

fn starts_with_question_word(lower: &str) -> bool {
    const Q: &[&str] = &["what ", "how ", "why ", "when ", "who ", "where ", "which "];
    Q.iter().any(|p| lower.starts_with(p))
}

/// Pulls the first 4-digit year between 2000 and 2099 inclusive.
fn extract_year(lower: &str) -> Option<i32> {
    let bytes = lower.as_bytes();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        if bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
        {
            // Make sure it's not part of a longer digit run (e.g. "20240")
            let prev_ok = i == 0 || !bytes[i - 1].is_ascii_digit();
            let next_ok = i + 4 == bytes.len() || !bytes[i + 4].is_ascii_digit();
            if prev_ok && next_ok {
                let year = std::str::from_utf8(&bytes[i..i + 4])
                    .ok()
                    .and_then(|s| s.parse::<i32>().ok())?;
                if (2000..=2099).contains(&year) {
                    return Some(year);
                }
            }
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_lookup() {
        assert_eq!(classify("").intent, Intent::Lookup);
        assert_eq!(classify("   ").intent, Intent::Lookup);
    }

    #[test]
    fn short_terms_are_lookup() {
        assert_eq!(classify("Paxos").intent, Intent::Lookup);
        assert_eq!(classify("rust async").intent, Intent::Lookup);
        assert_eq!(classify("nginx config").intent, Intent::Lookup);
    }

    #[test]
    fn question_mark_is_exploratory() {
        assert_eq!(classify("how does Raft work?").intent, Intent::Exploratory);
    }

    #[test]
    fn long_natural_language_is_exploratory() {
        assert_eq!(
            classify("ideas for improving the meeting cadence next quarter").intent,
            Intent::Exploratory
        );
    }

    #[test]
    fn question_word_is_exploratory() {
        assert_eq!(classify("what is RRF").intent, Intent::Exploratory);
        assert_eq!(
            classify("why we picked Postgres").intent,
            Intent::Exploratory
        );
    }

    #[test]
    fn explicit_year_is_temporal_with_year() {
        let c = classify("notes from 2025 retro");
        assert_eq!(c.intent, Intent::Temporal);
        assert_eq!(c.year, Some(2025));
    }

    #[test]
    fn relative_time_is_temporal_no_year() {
        let c = classify("what did I write last week");
        assert_eq!(c.intent, Intent::Temporal);
        assert_eq!(c.year, None);
    }

    #[test]
    fn structural_glob_or_prefix() {
        assert_eq!(classify("file:meetings/*.md").intent, Intent::Structural);
        assert_eq!(classify("section: Coffee").intent, Intent::Structural);
        assert_eq!(classify("*.md drafts").intent, Intent::Structural);
        assert_eq!(classify("README.md changes").intent, Intent::Structural);
    }

    #[test]
    fn structural_beats_temporal_when_both_present() {
        // "*.md from 2025" has both structural and temporal markers.
        // Structural should win because it implies a hard filter route.
        let c = classify("*.md from 2025");
        assert_eq!(c.intent, Intent::Structural);
        assert_eq!(c.year, Some(2025));
    }

    #[test]
    fn out_of_range_year_does_not_trigger_temporal() {
        // 1999 isn't in the 2000-2099 window — should fall back to lookup/exploratory.
        let c = classify("notes from 1999");
        assert_ne!(c.intent, Intent::Temporal);
        assert_eq!(c.year, None);
    }

    #[test]
    fn five_token_borderline_is_exploratory() {
        let c = classify("rust async tokio runtime example");
        assert_eq!(c.intent, Intent::Exploratory);
    }
}
