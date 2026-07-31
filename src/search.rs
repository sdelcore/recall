//! End-to-end search pipeline: BM25 / vector / RRF fusion, optional LLM
//! reranking, and recency decay. One entry point — [`search`] — that both the
//! CLI and the MCP server call. Renders nothing; callers decide how to display
//! the [`SearchOutcome`] (compact, JSON, MCP envelope, etc.).
//!
//! Keeping the pipeline in one module is the whole point: the rules
//! ("fetch `RERANK_CANDIDATES` when reranking", "fall back to BM25 when no
//! embeddings exist", "carry traces through reranking") live here, not in
//! both call sites.

use std::collections::HashMap;

use anyhow::{Context, Result};
use chrono::{NaiveDate, Utc};
use tracing::warn;

use crate::embedder::Embedder;
use crate::intent;
use crate::reranker;
use crate::store::{SearchOptions, SearchResult, SearchTrace, Store};

/// Half-life for a collection that has no `half_life_days` of its own. Ninety
/// days is a working default for a notes vault; `recall collection half-life`
/// is the one place to say otherwise, because corpora age at wildly different
/// rates and one number cannot serve them all.
const DEFAULT_HALF_LIFE_DAYS: f64 = 90.0;

/// How many RRF candidates a reranking search fetches. Twenty is what fits in
/// one Haiku prompt without truncating snippets; the number is a property of
/// the prompt, so it lives next to the code that builds it.
const RERANK_CANDIDATES: usize = 20;

/// Inputs to a single search invocation. The caller fills in everything;
/// the year bound that a temporal query implies is filled in by [`search`].
pub struct SearchRequest {
    pub query: String,
    pub limit: usize,
    pub collection: Option<String>,
    pub after: Option<String>,
    pub before: Option<String>,
    pub rerank: bool,
}

/// Outcome of a search: the original query and the ranked results paired with
/// their per-result trace.
pub struct SearchOutcome {
    pub query: String,
    pub results: Vec<(SearchResult, SearchTrace)>,
}

/// Run the full pipeline against a fresh store handle. Convenience entry point
/// for callers that don't already hold a `Store`.
pub async fn search(request: SearchRequest) -> Result<SearchOutcome> {
    let store = Store::open()?;
    search_with_store(&store, request).await
}

/// Same as [`search`] but takes an explicit store. Exposed so callers that
/// already hold one (or want to share it with index/status/etc.) can avoid
/// re-opening the database.
pub async fn search_with_store(store: &Store, mut request: SearchRequest) -> Result<SearchOutcome> {
    // A query that names a year gets that year as a lower bound, unless the
    // caller already set one. Reranking is deliberately never routed: it costs
    // seconds of LLM latency, so only an explicit `--rerank` (or the MCP
    // `rerank` argument) may turn it on.
    if request.after.is_none() {
        if let Some(year) = intent::year(&request.query) {
            request.after = Some(format!("{year}-01-01"));
        }
    }

    let collection_id = match request.collection.as_deref() {
        Some(name) => Some(
            store
                .get_collection(name)?
                .with_context(|| format!("Collection {:?} not found", name))?
                .id,
        ),
        None => None,
    };

    let fetch_limit = if request.rerank {
        RERANK_CANDIDATES.max(request.limit)
    } else {
        request.limit
    };

    let options = SearchOptions {
        after: request.after.clone(),
        before: request.before.clone(),
        collection_id,
    };

    let mut traced = run_retrieval(store, &request.query, fetch_limit, &options).await?;

    if request.rerank && !traced.is_empty() {
        traced = apply_reranking(traced, &request.query, request.limit).await?;
    } else {
        traced.truncate(request.limit);
    }

    apply_decay(store, &request.query, &mut traced)?;

    Ok(SearchOutcome {
        query: request.query,
        results: traced,
    })
}

/// Hybrid retrieval, degrading to BM25 when the index has no vectors. There
/// is no keyword-only mode to ask for: the fusion is strictly better when
/// embeddings exist, and this is the only place that can tell whether they do.
///
/// An empty vector table is a normal state, not a broken one — `recall index`
/// writes chunks and `recall embed` writes vectors, so a freshly indexed vault
/// lands here until the second command runs. The model itself is in-process
/// and always available, so the warning names the missing step rather than
/// suggesting anything is wrong with the host.
async fn run_retrieval(
    store: &Store,
    query: &str,
    fetch_limit: usize,
    options: &SearchOptions,
) -> Result<Vec<(SearchResult, SearchTrace)>> {
    let (embedded, _) = store.get_embedding_stats()?;
    if embedded == 0 {
        warn!("This index has no embeddings yet, so this search is keyword-only. Run 'recall embed' to add them.");
        return store.search_fts_traced(query, fetch_limit, options);
    }

    let embedder = Embedder::load()?;
    let query_embedding = embedder.embed_query(query)?;
    store.search_hybrid_traced(query, &query_embedding, fetch_limit, options)
}

/// Floored multiplicative recency factor:
/// `f = 0.5 + 0.5 * exp(-ln2 * age / half_life)`, range `[0.5, 1.0]`.
///
/// The floor is the point of the design. Recency multiplies the relevance
/// score instead of adding to it, and it can never remove more than half of
/// that score, so it separates results the retriever already considers
/// comparable without ever letting a fresh weak match outrank a strong old
/// one. The half-life is validated where it is written — `recall collection
/// half-life` rejects a non-positive value — so there is no guard here.
fn recency_factor(age_days: f64, half_life_days: f64) -> f64 {
    let age = age_days.max(0.0);
    0.5 + 0.5 * (-std::f64::consts::LN_2 * age / half_life_days).exp()
}

/// Multiply the recency factor into the final scores and re-sort.
///
/// Always on. It was a config switch defaulting to off, which meant the
/// ranking every user actually got was the worse one — the labeled query set
/// in `tests/ranking.rs` scores 14/16 without decay and 16/16 with it, and the
/// `[0.5, 1.0]` floor means a fresh weak match can never outrank a strong old
/// one. There is nothing left to hedge.
///
/// Runs last, on the already-reranked and truncated list, so the arithmetic
/// is the only recency authority in the pipeline — dates are deliberately
/// kept out of the rerank prompt, where an LLM would apply its own
/// uncontrolled recency bias on top.
fn apply_decay(
    store: &Store,
    query: &str,
    traced: &mut [(SearchResult, SearchTrace)],
) -> Result<()> {
    if traced.is_empty() {
        return Ok(());
    }
    // A temporal query ("when did I switch to Postgres", "notes from 2023")
    // asks for old material on purpose. Demoting age would answer the
    // opposite question, so decay is skipped outright.
    if intent::is_temporal(query) {
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
            .unwrap_or(DEFAULT_HALF_LIFE_DAYS);

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
    limit: usize,
) -> Result<Vec<(SearchResult, SearchTrace)>> {
    // Snapshot pre-rerank traces by chunk identity so we can re-pair after
    // the reranker reorders + truncates.
    let mut trace_by_key: HashMap<(String, i64, i64), SearchTrace> = HashMap::new();
    for (r, t) in &traced {
        trace_by_key.insert((r.file_path.clone(), r.start_line, r.end_line), t.clone());
    }

    let plain: Vec<SearchResult> = traced.into_iter().map(|(r, _)| r).collect();
    let reranked = reranker::rerank(query, plain, limit).await?;

    Ok(reranked
        .into_iter()
        .map(|r| {
            let key = (r.file_path.clone(), r.start_line, r.end_line);
            let mut t = trace_by_key.remove(&key).unwrap_or_default();
            t.rerank_score = Some(r.score);
            (r, t)
        })
        .collect())
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
    fn age_in_days_counts_whole_days_and_ignores_junk() {
        let today = NaiveDate::from_ymd_opt(2026, 1, 11).unwrap();
        assert_eq!(age_in_days("2026-01-01", today), Some(10.0));
        assert_eq!(age_in_days("2026-01-11", today), Some(0.0));
        assert_eq!(age_in_days("not-a-date", today), None);
    }
}
