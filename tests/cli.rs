//! End-to-end CLI tests. Exercises `recall index`, `recall search`,
//! and `recall status` against a tempdir vault using BM25 only — no Ollama
//! or LLM reranker required, so this runs unchanged on CI.

mod common;

use common::{write_fixture_vault, RecallSandbox};
use predicates::prelude::*;
use tempfile::tempdir;

#[test]
fn index_then_bm25_search_returns_expected_hit() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());

    sandbox
        .cmd()
        .args(["index", "--path"])
        .arg(vault.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed").and(predicate::str::contains("chunks")));

    // Term unique to gamma.md
    sandbox
        .cmd()
        .args(["search", "Paxos", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("gamma.md"));

    // Term unique to alpha.md
    sandbox
        .cmd()
        .args(["search", "fox", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("alpha.md"));
}

#[test]
fn status_json_reports_indexed_counts() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());

    sandbox
        .cmd()
        .args(["index", "--path"])
        .arg(vault.path())
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["status", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v["file_count"].as_i64().unwrap() >= 3 && v["chunk_count"].as_i64().unwrap() >= 3
        }));
}

#[test]
fn search_with_no_index_returns_empty_results() {
    let sandbox = RecallSandbox::new();

    sandbox
        .cmd()
        .args(["search", "anything", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"results\""));
}

#[test]
fn search_with_trace_emits_trace_object() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());

    sandbox
        .cmd()
        .args(["index", "--path"])
        .arg(vault.path())
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["search", "Paxos", "--trace"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            let r = &v["results"][0];
            // gamma.md has the only Paxos hit
            r["file"].as_str().unwrap().contains("gamma.md")
                // BM25-only path: bm25_rank=0, vec_rank=null, rrf_score>0, rerank_score=null
                && r["trace"]["bm25_rank"].as_i64() == Some(0)
                && r["trace"]["vec_rank"].is_null()
                && r["trace"]["rrf_score"].as_f64().unwrap() > 0.0
                && r["trace"]["rerank_score"].is_null()
        }));
}
