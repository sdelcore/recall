//! Temporal query detection.
//!
//! Two questions, both answered by string inspection alone — no LLM call, so
//! search latency stays predictable:
//!
//! - [`year`]: does the query name a year? If so [`crate::search`] uses it as
//!   a lower bound on `date`, unless the caller set one explicitly.
//! - [`is_temporal`]: does the query ask about a point in time at all? If so
//!   recency decay is skipped, because a query reaching for old material must
//!   not have age counted against it.
//!
//! This used to classify into four buckets. Two of them routed nothing, and
//! the `structural` bucket was checked first and actively harmful: `*.md from
//! 2025` matched it, which suppressed both the year bound and the decay skip
//! the query plainly wants.

/// Pulls the first 4-digit year between 2000 and 2099 inclusive.
pub fn year(query: &str) -> Option<i32> {
    let bytes = query.as_bytes();
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

/// True when the query names a year or a relative time window.
pub fn is_temporal(query: &str) -> bool {
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
    if year(query).is_some() {
        return true;
    }
    let lower = query.to_ascii_lowercase();
    HINTS.iter().any(|h| lower.contains(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_year_is_temporal_and_is_extracted() {
        assert_eq!(year("notes from 2025 retro"), Some(2025));
        assert!(is_temporal("notes from 2025 retro"));
    }

    #[test]
    fn relative_time_is_temporal_without_a_year() {
        assert!(is_temporal("what did I write last week"));
        assert_eq!(year("what did I write last week"), None);
        assert!(
            is_temporal("Recently Touched Notes"),
            "match is case-folded"
        );
    }

    #[test]
    fn a_plain_query_is_not_temporal() {
        assert!(!is_temporal(""));
        assert!(!is_temporal("Paxos"));
        assert!(!is_temporal("how does Raft work?"));
    }

    #[test]
    fn a_year_outside_the_window_is_not_a_year() {
        assert_eq!(year("notes from 1999"), None);
        assert!(!is_temporal("notes from 1999"));
        assert_eq!(year("id 20240 rollout"), None);
    }

    /// A glob or a `.md` suffix used to short-circuit the classifier into a
    /// `structural` bucket that suppressed both the year bound and the decay
    /// skip. There is no bucket now, so the year still wins.
    #[test]
    fn a_glob_no_longer_hides_the_year() {
        assert_eq!(year("*.md from 2025"), Some(2025));
        assert!(is_temporal("*.md from 2025"));
    }
}
