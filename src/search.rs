//! End-to-end search pipeline: intent classification, BM25 / vector / RRF
//! fusion, and optional LLM reranking. One entry point — [`search`] — that
//! both the CLI and the MCP server call. Renders nothing; callers decide how
//! to display the [`SearchOutcome`] (compact, JSON, MCP envelope, etc.).
//!
//! Keeping the pipeline in one module is the whole point: the rules
//! ("fetch 3× candidates when reranking", "fall back to BM25 when no
//! embeddings exist", "carry traces through reranking") live here, not in
//! both call sites.

use std::collections::HashMap;

use anyhow::{Context, Result};
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
        options.collection_id,
    )
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
