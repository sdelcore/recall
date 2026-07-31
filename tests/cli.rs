//! End-to-end CLI tests. Exercises `recall index`, `recall search`,
//! and `recall status` against a tempdir vault using BM25 only — no model
//! weights or LLM reranker required, so this runs unchanged on CI.

mod common;

use common::{write_fixture_vault, RecallSandbox};
use predicates::prelude::*;
use tempfile::tempdir;

/// Register a collection and seed it with the fixture vault.
fn add_and_index(sandbox: &RecallSandbox, vault: &std::path::Path, name: &str) {
    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault)
        .args(["--name", name])
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["index", "--collection", name])
        .assert()
        .success();
}

#[test]
fn index_then_bm25_search_returns_expected_hit() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());
    add_and_index(&sandbox, vault.path(), "test");

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

/// Status reports the counts *and* whether they can be trusted — the
/// integrity check and orphan counts that used to be `maintenance check`.
///
/// `collection_roots` is the answer to "what does this index cover". It used
/// to be `index_paths` / `watch_paths` read out of the config file, which
/// reported `~/Obsidian` no matter where the collections actually pointed.
#[test]
fn status_json_reports_indexed_counts_and_health() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());
    add_and_index(&sandbox, vault.path(), "test");
    // `collection add` stores the canonical root, so compare against the
    // canonical path. On macOS `tempdir()` hands back `/var/folders/...`,
    // which resolves to `/private/var/folders/...`; on Linux the two are
    // identical, which is why comparing the raw path passed there and failed
    // only on macOS.
    let root = vault
        .path()
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .to_string();

    sandbox
        .cmd()
        .args(["status", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(move |out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v["collection_roots"] == serde_json::json!([root])
                && v.get("config_path").is_none()
                && v["file_count"].as_i64().unwrap() >= 3
                && v["chunk_count"].as_i64().unwrap() >= 3
                && v["integrity"] == "ok"
                && v["orphans"]["chunks"].as_i64() == Some(0)
                && v["orphans"]["files"].as_i64() == Some(0)
                && v["orphans"]["embeddings"].as_i64() == Some(0)
                && v["healthy"].as_bool() == Some(true)
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
    add_and_index(&sandbox, vault.path(), "test");

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
