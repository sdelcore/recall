//! LLM-based reranking for search results.
//!
//! One adapter: `ClaudeCodeReranker` (claude-agent-sdk, no API key). The
//! `Reranker` trait stays because it is the test seam — orchestration is
//! covered by unit tests against a fake adapter, never against the SDK.
//!
//! `rerank` is the orchestration entry point: it calls `score`, validates the
//! result, sorts, and truncates. It does **not** degrade to RRF order on
//! failure. A caller that asked for reranking and got unreranked results has
//! no way to tell — over MCP the warning never reaches the model — so every
//! failure propagates as an error instead.

use anyhow::{bail, ensure, Context, Result};
use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::store::SearchResult;

/// Model the reranker asks. Haiku is the cheapest model that can hold twenty
/// snippets and emit twenty numbers; a bigger one costs latency the caller is
/// already paying seconds for.
const RERANK_MODEL: &str = "haiku";

/// Score `candidates` for relevance to `query`. Returns one f64 in `[0, 10]`
/// per candidate, in the same order. Caller is responsible for sorting and
/// truncation.
#[async_trait]
pub trait Reranker: Send + Sync {
    async fn score(&self, query: &str, candidates: &[SearchResult]) -> Result<Vec<f64>>;
}

/// Rerank search results with an LLM.
///
/// Takes `candidates` (pre-sorted by RRF), scores each for relevance to
/// `query`, and returns the top `top_k` re-sorted by LLM score. Errors if the
/// model is unreachable or answers with anything other than one score per
/// candidate. How many candidates arrive is the pipeline's decision, not this
/// module's — `search.rs` over-fetches for exactly this reason.
pub async fn rerank(
    query: &str,
    candidates: Vec<SearchResult>,
    top_k: usize,
) -> Result<Vec<SearchResult>> {
    rerank_with(&ClaudeCodeReranker, query, candidates, top_k).await
}

/// Orchestration variant taking an explicit reranker. Exposed so the search
/// pipeline can hold one adapter and so tests can inject a fake.
pub async fn rerank_with(
    reranker: &dyn Reranker,
    query: &str,
    candidates: Vec<SearchResult>,
    top_k: usize,
) -> Result<Vec<SearchResult>> {
    if candidates.is_empty() {
        return Ok(candidates);
    }

    let top_k = top_k.min(candidates.len());

    let scores = reranker
        .score(query, &candidates)
        .await
        .context("Reranking failed")?;

    ensure!(
        scores.len() == candidates.len(),
        "Reranker returned {} scores for {} candidates — expected equal count",
        scores.len(),
        candidates.len()
    );

    let mut scored: Vec<(f64, SearchResult)> = scores.into_iter().zip(candidates).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    info!(
        "Reranked {} candidates → top {} (scores: {:.1}..{:.1})",
        scored.len(),
        top_k,
        scored.first().map(|s| s.0).unwrap_or(0.0),
        scored.last().map(|s| s.0).unwrap_or(0.0),
    );

    Ok(scored
        .into_iter()
        .take(top_k)
        .map(|(score, mut r)| {
            r.score = score;
            r
        })
        .collect())
}

/// Build the batched reranking prompt.
/// Returns a single prompt that asks the LLM for comma-separated scores.
fn build_rerank_prompt(query: &str, candidates: &[SearchResult]) -> String {
    let doc_list: String = candidates
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let truncated: String = r.content.chars().take(500).collect();
            format!("Document {}: {}", i + 1, truncated)
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        "Rate the relevance of each document to the query on a scale of 0-10.\n\
         Reply with ONLY the scores as a comma-separated list (e.g., 7,3,9,1,5).\n\
         You MUST return exactly {} scores. No other text.\n\n\
         Query: {}\n\n{}",
        candidates.len(),
        query,
        doc_list
    )
}

/// Parse comma-separated scores from LLM response.
/// Returns an error with diagnostic info if parsing fails.
fn parse_scores(response: &str, expected_count: usize) -> Result<Vec<f64>> {
    let trimmed = response.trim();
    if trimmed.is_empty() {
        bail!("Reranker returned empty response");
    }

    let tokens: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();

    if tokens.len() != expected_count {
        bail!(
            "Expected {} comma-separated scores, got {} tokens. Raw response: {:?}",
            expected_count,
            tokens.len(),
            trimmed
        );
    }

    let mut scores = Vec::with_capacity(tokens.len());
    for (i, token) in tokens.iter().enumerate() {
        let score: f64 = token.parse().with_context(|| {
            format!(
                "Failed to parse score {} ({:?}) as number. Full response: {:?}",
                i, token, trimmed
            )
        })?;
        if !(0.0..=10.0).contains(&score) {
            warn!(
                "Score {} = {} is outside [0, 10] range, clamping. Full response: {:?}",
                i, score, trimmed
            );
        }
        scores.push(score.clamp(0.0, 10.0));
    }

    Ok(scores)
}

// ── Claude Code SDK adapter ───────────────────────────────────────────────

pub struct ClaudeCodeReranker;

#[async_trait]
impl Reranker for ClaudeCodeReranker {
    async fn score(&self, query: &str, candidates: &[SearchResult]) -> Result<Vec<f64>> {
        use claude_agent_sdk::{ClaudeAgentOptions, ContentBlock, Message, PermissionMode};
        use futures::StreamExt;

        debug!(
            "Reranking {} candidates via claude-code SDK (model={})",
            candidates.len(),
            RERANK_MODEL
        );

        let mut options = ClaudeAgentOptions::builder()
            .permission_mode(PermissionMode::BypassPermissions)
            .build();
        options.model = Some(RERANK_MODEL.to_string());

        let prompt = build_rerank_prompt(query, candidates);

        let stream = claude_agent_sdk::query(&prompt, Some(options))
            .await
            .context("Failed to create claude-code query for reranking")?;

        let mut stream = Box::pin(stream);
        let mut text = String::new();
        let mut got_assistant_message = false;
        let mut skipped_errors = 0u32;

        while let Some(result) = stream.next().await {
            match result {
                Ok(Message::Assistant { message, .. }) => {
                    got_assistant_message = true;
                    for block in &message.content {
                        if let ContentBlock::Text { text: t } = block {
                            text.push_str(t);
                        }
                    }
                }
                Ok(Message::Result { .. }) => break,
                Ok(_) => {}
                Err(e) => {
                    let err_str = e.to_string();
                    if err_str.contains("unknown variant") || err_str.contains("parse") {
                        skipped_errors += 1;
                        debug!("Skipping non-fatal SDK parse error: {}", err_str);
                        if skipped_errors > 50 && !got_assistant_message {
                            bail!(
                                "SDK stream produced {} parse errors with no assistant messages — \
                                 likely broken. Last error: {}",
                                skipped_errors,
                                err_str
                            );
                        }
                        continue;
                    }
                    bail!("Fatal SDK stream error during reranking: {}", e);
                }
            }
        }

        if !got_assistant_message {
            bail!(
                "SDK stream completed with no assistant messages \
                 (skipped {} parse errors)",
                skipped_errors
            );
        }

        if text.trim().is_empty() {
            bail!(
                "SDK returned {} assistant messages but no text content \
                 (skipped {} parse errors)",
                if got_assistant_message { "some" } else { "no" },
                skipped_errors
            );
        }

        if skipped_errors > 0 {
            debug!(
                "Reranking completed with {} skipped parse errors",
                skipped_errors
            );
        }

        parse_scores(&text, candidates.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn make_result(path: &str, content: &str, score: f64) -> SearchResult {
        SearchResult {
            file_path: path.into(),
            start_line: 1,
            end_line: 5,
            content: content.into(),
            score,
            date: None,
            date_source: None,
            section: None,
            status: None,
            collection_name: None,
            collection_description: None,
        }
    }

    fn make_candidates(n: usize) -> Vec<SearchResult> {
        (0..n)
            .map(|i| make_result(&format!("doc{}.md", i), &format!("content {}", i), 1.0))
            .collect()
    }

    type FakeBehavior = dyn FnMut(&str, &[SearchResult]) -> Result<Vec<f64>> + Send;

    /// Fake reranker driven by a closure so each test states its own behaviour.
    struct FakeReranker {
        behavior: Mutex<Box<FakeBehavior>>,
    }

    impl FakeReranker {
        fn new<F>(f: F) -> Self
        where
            F: FnMut(&str, &[SearchResult]) -> Result<Vec<f64>> + Send + 'static,
        {
            Self {
                behavior: Mutex::new(Box::new(f)),
            }
        }
    }

    #[async_trait]
    impl Reranker for FakeReranker {
        async fn score(&self, query: &str, candidates: &[SearchResult]) -> Result<Vec<f64>> {
            let mut f = self.behavior.lock().unwrap();
            f(query, candidates)
        }
    }

    #[test]
    fn test_parse_scores_valid() {
        let scores = parse_scores("7,3,9,1,5", 5).unwrap();
        assert_eq!(scores, vec![7.0, 3.0, 9.0, 1.0, 5.0]);
    }

    #[test]
    fn test_parse_scores_with_whitespace() {
        let scores = parse_scores("  7 , 3 , 9 , 1 , 5  ", 5).unwrap();
        assert_eq!(scores, vec![7.0, 3.0, 9.0, 1.0, 5.0]);
    }

    #[test]
    fn test_parse_scores_wrong_count() {
        let err = parse_scores("7,3,9", 5).unwrap_err();
        assert!(
            err.to_string().contains("Expected 5"),
            "Error should mention expected count: {}",
            err
        );
    }

    #[test]
    fn test_parse_scores_empty() {
        let err = parse_scores("", 5).unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "Error should mention empty: {}",
            err
        );
    }

    #[test]
    fn test_parse_scores_non_numeric() {
        let err = parse_scores("7,three,9,1,5", 5).unwrap_err();
        assert!(
            err.to_string().contains("three"),
            "Error should show the bad token: {}",
            err
        );
    }

    #[test]
    fn test_parse_scores_clamps_out_of_range() {
        let scores = parse_scores("15,-3,9,1,5", 5).unwrap();
        assert_eq!(scores, vec![10.0, 0.0, 9.0, 1.0, 5.0]);
    }

    #[test]
    fn test_build_rerank_prompt_contains_all_docs() {
        let candidates = vec![
            make_result("a.md", "Doc one content", 1.0),
            make_result("b.md", "Doc two content", 0.5),
        ];
        let prompt = build_rerank_prompt("test query", &candidates);
        assert!(prompt.contains("Document 1: Doc one content"));
        assert!(prompt.contains("Document 2: Doc two content"));
        assert!(prompt.contains("test query"));
        assert!(prompt.contains("exactly 2 scores"));
    }

    #[tokio::test]
    async fn rerank_with_sorts_by_score_descending_and_truncates_to_top_k() {
        // Five candidates, scores [3, 9, 1, 7, 5], top_k=3 → expect docs at idx 1, 3, 4.
        let fake = FakeReranker::new(|_, c| {
            assert_eq!(c.len(), 5);
            Ok(vec![3.0, 9.0, 1.0, 7.0, 5.0])
        });
        let candidates = make_candidates(5);

        let out = rerank_with(&fake, "q", candidates, 3).await.unwrap();

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].file_path, "doc1.md");
        assert_eq!(out[0].score, 9.0);
        assert_eq!(out[1].file_path, "doc3.md");
        assert_eq!(out[1].score, 7.0);
        assert_eq!(out[2].file_path, "doc4.md");
        assert_eq!(out[2].score, 5.0);
    }

    #[tokio::test]
    async fn rerank_with_errors_on_score_count_mismatch() {
        let fake = FakeReranker::new(|_, _| Ok(vec![1.0, 2.0])); // expected 5
        let candidates = make_candidates(5);

        let err = rerank_with(&fake, "q", candidates, 3)
            .await
            .expect_err("count mismatch must not degrade to RRF order");

        assert!(
            err.to_string().contains("2 scores for 5 candidates"),
            "Error should report the counts: {}",
            err
        );
    }

    #[tokio::test]
    async fn rerank_with_propagates_provider_error() {
        let fake = FakeReranker::new(|_, _| bail!("boom"));
        let candidates = make_candidates(4);

        let err = rerank_with(&fake, "q", candidates, 2)
            .await
            .expect_err("provider failure must not degrade to RRF order");

        assert_eq!(err.to_string(), "Reranking failed");
        assert_eq!(err.root_cause().to_string(), "boom");
    }

    #[tokio::test]
    async fn rerank_with_returns_empty_for_no_candidates() {
        let fake = FakeReranker::new(|_, _| panic!("should not be called"));
        let out = rerank_with(&fake, "q", vec![], 5).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn rerank_with_scores_every_candidate_it_is_given() {
        // The pipeline decides the candidate count by over-fetching; this
        // function never silently drops part of what it was handed.
        let fake = FakeReranker::new(|_, c| {
            assert_eq!(c.len(), 10, "every candidate must be scored");
            Ok((0..10).map(|i| i as f64).collect())
        });
        let candidates = make_candidates(10);

        let out = rerank_with(&fake, "q", candidates, 2).await.unwrap();

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].file_path, "doc9.md");
        assert_eq!(out[1].file_path, "doc8.md");
    }
}
