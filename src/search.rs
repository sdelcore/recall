//! End-to-end search pipeline: intent classification, BM25 / vector / RRF
//! fusion, optional LLM reranking, and recency decay. One entry point —
//! [`search`] — that both the CLI and the MCP server call. Renders nothing;
//! callers decide how to display the [`SearchOutcome`] (compact, JSON, MCP
//! envelope, etc.).
//!
//! Keeping the pipeline in one module is the whole point: the rules
//! ("fetch 3× candidates when reranking", "fall back to BM25 when no
//! embeddings exist", "carry traces through reranking") live here, not in
//! both call sites.

use std::collections::HashMap;

use anyhow::{Context, Result};
use chrono::{NaiveDate, Utc};
use tracing::warn;

use crate::config::Config;
use crate::embedder::Embedder;
use crate::intent::{self, Classified, Intent};
use crate::reranker;
use crate::store::{SearchOptions, SearchResult, SearchTrace, Store};

/// Inputs to a single search invocation. The caller fills in everything;
/// `--auto` routing happens inside [`search`].
pub struct SearchRequest {
    pub query: String,
    pub limit: usize,
    pub collection: Option<String>,
    pub after: Option<String>,
    pub before: Option<String>,
    pub project: Option<String>,
    pub file_pattern: Option<String>,
    pub hybrid: bool,
    pub rerank: bool,
    /// Override `config.reranking.provider` for this single call. CLI uses it
    /// for `--rerank-provider`; MCP doesn't expose the knob.
    pub rerank_provider_override: Option<String>,
    pub auto: bool,
}

/// Outcome of a search: the original query, the classified intent (always
/// computed, used for `--auto` routing and surfaced in `--trace`), and the
/// ranked results paired with their per-result trace.
pub struct SearchOutcome {
    pub query: String,
    pub intent: Classified,
    pub results: Vec<(SearchResult, SearchTrace)>,
}

/// Run the full pipeline against a fresh store handle. Convenience entry point
/// for callers that don't already hold a `Store`.
pub async fn search(config: &Config, request: SearchRequest) -> Result<SearchOutcome> {
    let store = Store::open()?;
    search_with_store(&store, config, request).await
}

/// Same as [`search`] but takes an explicit store. Exposed so callers that
/// already hold one (or want to share it with index/status/etc.) can avoid
/// re-opening the database.
pub async fn search_with_store(
    store: &Store,
    config: &Config,
    mut request: SearchRequest,
) -> Result<SearchOutcome> {
    let classified = intent::classify(&request.query);

    if request.auto {
        apply_auto_routing(&classified, &mut request);
    }

    let do_rerank = request.rerank || config.reranking.enabled;

    let collection_id = match request.collection.as_deref() {
        Some(name) => Some(
            store
                .get_collection(name)?
                .with_context(|| format!("Collection {:?} not found", name))?
                .id,
        ),
        None => None,
    };

    let fetch_limit = if do_rerank {
        config.reranking.candidates.max(request.limit)
    } else {
        request.limit
    };

    let options = SearchOptions {
        after: request.after.clone(),
        before: request.before.clone(),
        project: request.project.clone(),
        file_pattern: request.file_pattern.clone(),
        collection_id,
    };

    let mut traced = run_retrieval(
        store,
        config,
        &request.query,
        fetch_limit,
        &options,
        request.hybrid,
    )
    .await?;

    if do_rerank && !traced.is_empty() {
        traced = apply_reranking(
            traced,
            &request.query,
            &config.reranking,
            request.rerank_provider_override.as_deref(),
            request.limit,
        )
        .await;
    } else {
        traced.truncate(request.limit);
    }

    apply_decay(store, config, &classified, &mut traced)?;

    Ok(SearchOutcome {
        query: request.query,
        intent: classified,
        results: traced,
    })
}

fn apply_auto_routing(classified: &Classified, request: &mut SearchRequest) {
    match classified.intent {
        Intent::Exploratory => {
            request.hybrid = true;
            request.rerank = true;
        }
        Intent::Temporal => {
            if let (None, Some(year)) = (request.after.as_deref(), classified.year) {
                request.after = Some(format!("{year}-01-01"));
            }
        }
        Intent::Lookup | Intent::Structural => {}
    }
}

async fn run_retrieval(
    store: &Store,
    config: &Config,
    query: &str,
    fetch_limit: usize,
    options: &SearchOptions,
    hybrid: bool,
) -> Result<Vec<(SearchResult, SearchTrace)>> {
    if !hybrid {
        return store.search_fts_traced(query, fetch_limit, config.search.rrf_k, options);
    }

    let (embedded, _) = store.get_embedding_stats()?;
    if embedded == 0 {
        warn!("No embeddings found; falling back to BM25. Run 'recall embed' to enable hybrid.");
        return store.search_fts_traced(query, fetch_limit, config.search.rrf_k, options);
    }

    let embedder = Embedder::new_with_config(config);
    let query_embedding = embedder.embed(query).await?;
    store.search_hybrid_traced(
        query,
        &query_embedding,
        fetch_limit,
        config.search.rrf_k,
        options,
    )
}

/// Floored multiplicative recency factor:
/// `f = 0.5 + 0.5 * exp(-ln2 * age / half_life)`, range `[0.5, 1.0]`.
///
/// The floor is the point of the design. Recency multiplies the relevance
/// score instead of adding to it, and it can never remove more than half of
/// that score, so it separates results the retriever already considers
/// comparable without ever letting a fresh weak match outrank a strong old
/// one. A non-positive half-life is not a valid setting; it returns a neutral
/// factor rather than dividing by zero.
fn recency_factor(age_days: f64, half_life_days: f64) -> f64 {
    if half_life_days <= 0.0 {
        return 1.0;
    }
    let age = age_days.max(0.0);
    0.5 + 0.5 * (-std::f64::consts::LN_2 * age / half_life_days).exp()
}

/// Multiply the recency factor into the final scores and re-sort.
///
/// Runs last, on the already-reranked and truncated list, so the arithmetic
/// is the only recency authority in the pipeline — dates are deliberately
/// kept out of the rerank prompt, where an LLM would apply its own
/// uncontrolled recency bias on top.
fn apply_decay(
    store: &Store,
    config: &Config,
    classified: &Classified,
    traced: &mut [(SearchResult, SearchTrace)],
) -> Result<()> {
    if !config.decay.enabled || traced.is_empty() {
        return Ok(());
    }
    // A temporal query ("when did I switch to Postgres", "notes from 2023")
    // asks for old material on purpose. Demoting age would answer the
    // opposite question, so decay is skipped outright.
    if classified.intent == Intent::Temporal {
        return Ok(());
    }

    let half_lives: HashMap<String, Option<f64>> = store
        .list_collections()?
        .into_iter()
        .map(|c| (c.name, c.half_life_days))
        .collect();
    let today = Utc::now().date_naive();

    for (result, trace) in traced.iter_mut() {
        // No usable date means no evidence of age. Treat the chunk as
        // neutral (factor 1.0) rather than guessing: an undated chunk is
        // never demoted, only out-ranked by something genuinely fresher.
        let Some(age_days) = result.date.as_deref().and_then(|d| age_in_days(d, today)) else {
            continue;
        };
        let half_life = result
            .collection_name
            .as_deref()
            .and_then(|name| half_lives.get(name).copied())
            .flatten()
            .unwrap_or(config.decay.default_half_life_days);

        let factor = recency_factor(age_days, half_life);
        trace.pre_decay_score = Some(result.score);
        trace.decay_factor = Some(factor);
        result.score *= factor;
    }

    traced.sort_by(|a, b| {
        b.0.score
            .partial_cmp(&a.0.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(())
}

fn age_in_days(date: &str, today: NaiveDate) -> Option<f64> {
    NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .ok()
        .map(|d| (today - d).num_days() as f64)
}

async fn apply_reranking(
    traced: Vec<(SearchResult, SearchTrace)>,
    query: &str,
    rerank_cfg: &crate::config::RerankConfig,
    provider_override: Option<&str>,
    limit: usize,
) -> Vec<(SearchResult, SearchTrace)> {
    let mut rerank_config = rerank_cfg.clone();
    if let Some(provider) = provider_override {
        rerank_config.provider = provider.to_string();
    }
    rerank_config.top_k = limit;

    // Snapshot pre-rerank traces by chunk identity so we can re-pair after
    // the reranker reorders + truncates.
    let mut trace_by_key: HashMap<(String, i64, i64), SearchTrace> = HashMap::new();
    for (r, t) in &traced {
        trace_by_key.insert((r.file_path.clone(), r.start_line, r.end_line), t.clone());
    }

    let plain: Vec<SearchResult> = traced.into_iter().map(|(r, _)| r).collect();
    let reranked = reranker::rerank(query, plain, &rerank_config).await;

    reranked
        .into_iter()
        .map(|r| {
            let key = (r.file_path.clone(), r.start_line, r.end_line);
            let mut t = trace_by_key.remove(&key).unwrap_or_default();
            t.rerank_score = Some(r.score);
            (r, t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_is_one_at_age_zero() {
        assert!((recency_factor(0.0, 90.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn factor_is_three_quarters_at_one_half_life() {
        assert!((recency_factor(90.0, 90.0) - 0.75).abs() < 1e-12);
        assert!((recency_factor(19.0, 19.0) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn factor_never_falls_below_the_floor() {
        assert!(recency_factor(1e6, 90.0) >= 0.5);
        assert!((recency_factor(1e6, 90.0) - 0.5).abs() < 1e-9);
        // A future date must not be rewarded above 1.0.
        assert!((recency_factor(-500.0, 90.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn factor_decreases_monotonically_with_age() {
        let mut previous = f64::INFINITY;
        for age in 0..400 {
            let factor = recency_factor(age as f64, 90.0);
            assert!(factor < previous, "factor did not decrease at age {age}");
            previous = factor;
        }
    }

    #[test]
    fn non_positive_half_life_disables_decay() {
        assert_eq!(recency_factor(1000.0, 0.0), 1.0);
        assert_eq!(recency_factor(1000.0, -30.0), 1.0);
    }

    #[test]
    fn age_in_days_counts_whole_days_and_ignores_junk() {
        let today = NaiveDate::from_ymd_opt(2026, 1, 11).unwrap();
        assert_eq!(age_in_days("2026-01-01", today), Some(10.0));
        assert_eq!(age_in_days("2026-01-11", today), Some(0.0));
        assert_eq!(age_in_days("not-a-date", today), None);
    }
}
