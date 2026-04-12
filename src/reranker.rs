//! LLM-based reranking for search results.
//!
//! Supports three providers:
//! - `claude-code`: Uses claude-agent-sdk (default, no API key needed)
//! - `anthropic`: Direct Anthropic Messages API (needs API key, parallel calls)
//! - `ollama`: Local Ollama model (offline fallback)

use anyhow::{bail, Context, Result};
use tracing::{debug, info, warn};

use crate::config::RerankConfig;
use crate::store::SearchResult;

/// Rerank search results using the configured LLM provider.
///
/// Takes `candidates` (pre-sorted by RRF), scores each for relevance to `query`,
/// and returns the top `top_k` results re-sorted by LLM score.
///
/// On failure, logs a warning and returns the original candidates truncated to `top_k`
/// (never silently degrades — always logs why reranking was skipped).
pub async fn rerank(
    query: &str,
    candidates: Vec<SearchResult>,
    config: &RerankConfig,
) -> Vec<SearchResult> {
    if candidates.is_empty() {
        return candidates;
    }

    let top_k = config.top_k.min(candidates.len());
    let to_rerank = config.candidates.min(candidates.len());

    let rerank_input: Vec<SearchResult> = candidates.into_iter().take(to_rerank).collect();

    let result = match config.provider.as_str() {
        "claude-code" => rerank_claude_code(query, &rerank_input, config).await,
        "anthropic" => rerank_anthropic(query, &rerank_input, config).await,
        "ollama" => rerank_ollama(query, &rerank_input, config).await,
        other => {
            warn!(
                "Unknown reranking provider {:?}, skipping reranking. \
                 Valid providers: claude-code, anthropic, ollama",
                other
            );
            return rerank_input.into_iter().take(top_k).collect();
        }
    };

    match result {
        Ok(scores) => {
            if scores.len() != rerank_input.len() {
                warn!(
                    "Reranker returned {} scores for {} candidates — expected equal count. \
                     Falling back to RRF order.",
                    scores.len(),
                    rerank_input.len()
                );
                return rerank_input.into_iter().take(top_k).collect();
            }

            // Check for all-invalid scores
            let valid_count = scores.iter().filter(|s| **s >= 0.0).count();
            if valid_count == 0 {
                warn!(
                    "All {} reranker scores were invalid (<0). \
                     Falling back to RRF order. Scores: {:?}",
                    scores.len(),
                    scores
                );
                return rerank_input.into_iter().take(top_k).collect();
            }

            // Pair scores with results, sort descending, take top_k
            let mut scored: Vec<(f64, SearchResult)> = scores
                .into_iter()
                .zip(rerank_input)
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            info!(
                "Reranked {} candidates → top {} (scores: {:.1}..{:.1})",
                scored.len(),
                top_k,
                scored.first().map(|s| s.0).unwrap_or(0.0),
                scored.last().map(|s| s.0).unwrap_or(0.0),
            );

            scored.into_iter().take(top_k).map(|(score, mut r)| {
                r.score = score;
                r
            }).collect()
        }
        Err(e) => {
            warn!(
                "Reranking failed (provider={:?}): {}. Falling back to RRF order.",
                config.provider, e
            );
            rerank_input.into_iter().take(top_k).collect()
        }
    }
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

// ── Claude Code SDK provider ──────────────────────────────────────────────

async fn rerank_claude_code(
    query: &str,
    candidates: &[SearchResult],
    config: &RerankConfig,
) -> Result<Vec<f64>> {
    use claude_agent_sdk::{ClaudeAgentOptions, ContentBlock, Message, PermissionMode};
    use futures::StreamExt;

    let model = config
        .claude_code
        .as_ref()
        .map(|c| c.model.as_str())
        .unwrap_or("haiku");

    debug!("Reranking {} candidates via claude-code SDK (model={})", candidates.len(), model);

    let mut options = ClaudeAgentOptions::builder()
        .permission_mode(PermissionMode::BypassPermissions)
        .build();
    options.model = Some(model.to_string());

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
        debug!("Reranking completed with {} skipped parse errors", skipped_errors);
    }

    parse_scores(&text, candidates.len())
}

// ── Anthropic API provider ────────────────────────────────────────────────

async fn rerank_anthropic(
    query: &str,
    candidates: &[SearchResult],
    config: &RerankConfig,
) -> Result<Vec<f64>> {
    let api_config = config
        .anthropic
        .as_ref()
        .context("Reranking provider is 'anthropic' but [reranking.anthropic] config is missing")?;

    let api_key_env = api_config
        .api_key_env
        .as_deref()
        .unwrap_or("ANTHROPIC_API_KEY");
    let api_key = std::env::var(api_key_env).with_context(|| {
        format!(
            "Reranking provider 'anthropic' requires {} environment variable",
            api_key_env
        )
    })?;

    let model = api_config
        .model
        .as_deref()
        .unwrap_or("claude-haiku-4-5-20251001");
    let max_concurrent = api_config.max_concurrent.unwrap_or(10);

    debug!(
        "Reranking {} candidates via Anthropic API (model={}, max_concurrent={})",
        candidates.len(),
        model,
        max_concurrent
    );

    let client = reqwest::Client::new();
    let mut tasks = tokio::task::JoinSet::new();

    for (i, result) in candidates.iter().enumerate() {
        let prompt = format!(
            "Rate relevance 0-10. Reply with ONLY the number.\n\n\
             Query: {}\nDocument: {}",
            query,
            result.content.chars().take(500).collect::<String>()
        );
        let client = client.clone();
        let key = api_key.clone();
        let model = model.to_string();

        tasks.spawn(async move {
            let resp = client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", &key)
                .header("anthropic-version", "2023-06-01")
                .json(&serde_json::json!({
                    "model": model,
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": prompt}]
                }))
                .send()
                .await
                .with_context(|| format!("Anthropic API request failed for candidate {}", i))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                bail!(
                    "Anthropic API returned {} for candidate {}: {}",
                    status,
                    i,
                    body
                );
            }

            let body = resp
                .json::<serde_json::Value>()
                .await
                .with_context(|| format!("Failed to parse Anthropic response for candidate {}", i))?;

            let score_text = body["content"][0]["text"]
                .as_str()
                .with_context(|| {
                    format!(
                        "Anthropic response for candidate {} missing content[0].text: {:?}",
                        i, body
                    )
                })?;

            let score: f64 = score_text.trim().parse().with_context(|| {
                format!(
                    "Failed to parse Anthropic score for candidate {} ({:?})",
                    i, score_text
                )
            })?;

            Ok::<(usize, f64), anyhow::Error>((i, score.clamp(0.0, 10.0)))
        });
    }

    let mut scores = vec![0.0f64; candidates.len()];
    let mut errors = Vec::new();

    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok((idx, score))) => {
                scores[idx] = score;
            }
            Ok(Err(e)) => {
                errors.push(format!("{}", e));
            }
            Err(e) => {
                errors.push(format!("Task panicked: {}", e));
            }
        }
    }

    if !errors.is_empty() {
        let total = candidates.len();
        let failed = errors.len();
        if failed == total {
            bail!(
                "All {} Anthropic API rerank calls failed. First error: {}",
                total,
                errors[0]
            );
        }
        warn!(
            "{}/{} Anthropic rerank calls failed. First error: {}",
            failed, total, errors[0]
        );
    }

    Ok(scores)
}

// ── Ollama provider ───────────────────────────────────────────────────────

async fn rerank_ollama(
    query: &str,
    candidates: &[SearchResult],
    config: &RerankConfig,
) -> Result<Vec<f64>> {
    let ollama_config = config
        .ollama
        .as_ref()
        .context("Reranking provider is 'ollama' but [reranking.ollama] config is missing")?;

    let url = ollama_config
        .url
        .as_deref()
        .unwrap_or("http://localhost:11434");
    let model = ollama_config
        .model
        .as_deref()
        .unwrap_or("qwen2.5:1.5b");

    debug!(
        "Reranking {} candidates via Ollama (url={}, model={})",
        candidates.len(),
        url,
        model
    );

    let prompt = build_rerank_prompt(query, candidates);

    let client = reqwest::Client::new();
    let resp = client
        .post(&format!("{}/api/generate", url))
        .json(&serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
        }))
        .send()
        .await
        .context("Ollama reranking request failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("Ollama returned {} for reranking: {}", status, body);
    }

    let body = resp
        .json::<serde_json::Value>()
        .await
        .context("Failed to parse Ollama response")?;

    let text = body["response"]
        .as_str()
        .context("Ollama response missing 'response' field")?;

    parse_scores(text, candidates.len())
}

#[cfg(test)]
mod tests {
    use super::*;

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
            SearchResult {
                file_path: "a.md".into(),
                start_line: 1,
                end_line: 5,
                content: "Doc one content".into(),
                score: 1.0,
                date: None,
                section: None,
                project: None,
            },
            SearchResult {
                file_path: "b.md".into(),
                start_line: 1,
                end_line: 3,
                content: "Doc two content".into(),
                score: 0.5,
                date: None,
                section: None,
                project: None,
            },
        ];
        let prompt = build_rerank_prompt("test query", &candidates);
        assert!(prompt.contains("Document 1: Doc one content"));
        assert!(prompt.contains("Document 2: Doc two content"));
        assert!(prompt.contains("test query"));
        assert!(prompt.contains("exactly 2 scores"));
    }
}
