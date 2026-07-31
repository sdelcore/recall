//! Recency decay e2e tests. BM25-only (no Ollama needed), driven through
//! `recall search --trace` so the assertions read the same numbers a user
//! debugging their ranking would see.
//!
//! Decay is unconditional. There is no configuration to toggle here, and that
//! is the point of these tests: a bare `recall search` against a bare sandbox
//! must already be applying the factor.

mod common;

use common::RecallSandbox;
use predicates::prelude::*;
use tempfile::{tempdir, TempDir};

/// A vault with one note dated 2020 in its frontmatter and one undated note
/// that therefore falls back to today's mtime. Both match "quokka"; only the
/// old one carries the literal "2020" that makes a query temporal.
fn vault_with_old_and_new_notes() -> TempDir {
    let vault = tempdir().unwrap();
    std::fs::write(
        vault.path().join("old.md"),
        "---\ndate: 2020-01-02\n---\n\n# Old\n\nquokka archive from 2020.\n",
    )
    .unwrap();
    std::fs::write(
        vault.path().join("new.md"),
        "# New\n\nquokka sighting today.\n",
    )
    .unwrap();
    vault
}

/// An indexed sandbox. There is nothing to configure — decay is on because
/// the code says so, not because a fixture switched it on.
fn setup() -> (RecallSandbox, TempDir) {
    let sandbox = RecallSandbox::new();
    let vault = vault_with_old_and_new_notes();

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes"])
        .assert()
        .success();
    sandbox.cmd().args(["index"]).assert().success();
    (sandbox, vault)
}

/// Map file basename -> its trace object.
fn traces(out: &str) -> Vec<(String, serde_json::Value)> {
    let v: serde_json::Value = serde_json::from_str(out).expect("json");
    v["results"]
        .as_array()
        .expect("results array")
        .iter()
        .map(|r| {
            let file = r["file"].as_str().unwrap().rsplit('/').next().unwrap();
            (file.to_string(), r["trace"].clone())
        })
        .collect()
}

fn search_trace(sandbox: &RecallSandbox, query: &str) -> String {
    let out = sandbox
        .cmd()
        .args(["search", query, "--trace"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    String::from_utf8(out).expect("utf8")
}

/// Decay needs no configuration to run, and it demotes the stale note: a 2020
/// note is many half-lives past the 90-day default, so it sits on the 0.5
/// floor, while the undated note dates from its mtime — indexed seconds ago.
#[test]
fn decay_demotes_the_older_note_with_no_configuration() {
    let (sandbox, _vault) = setup();
    let out = search_trace(&sandbox, "quokka");
    let rows = traces(&out);

    let factor = |name: &str| {
        rows.iter()
            .find(|(f, _)| f == name)
            .and_then(|(_, t)| t["decay_factor"].as_f64())
            .unwrap_or_else(|| panic!("no decay factor for {name}: {rows:?}"))
    };
    assert!((factor("old.md") - 0.5).abs() < 1e-6, "{rows:?}");
    assert!((factor("new.md") - 1.0).abs() < 1e-6, "{rows:?}");
}

#[test]
fn decay_records_the_pre_decay_score() {
    let (sandbox, _vault) = setup();
    sandbox
        .cmd()
        .args(["search", "quokka", "--trace"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v["results"].as_array().unwrap().iter().all(|r| {
                let pre = r["trace"]["pre_decay_score"].as_f64().unwrap();
                let factor = r["trace"]["decay_factor"].as_f64().unwrap();
                let score = r["score"].as_f64().unwrap();
                (pre * factor - score).abs() < 1e-9
            })
        }));
}

/// A query that names a year asks for old material on purpose, so decay is
/// skipped outright — the one behavioural exception that survives.
#[test]
fn temporal_queries_skip_decay() {
    let (sandbox, _vault) = setup();
    let out = search_trace(&sandbox, "quokka 2020");
    let rows = traces(&out);
    assert!(!rows.is_empty());
    assert!(
        rows.iter().all(|(_, t)| t["decay_factor"].is_null()),
        "{rows:?}"
    );
}

/// The collection half-life is the only override left, and it must beat the
/// compiled-in default.
#[test]
fn a_collection_half_life_overrides_the_default() {
    let (sandbox, _vault) = setup();

    let old_note_factor = |sandbox: &RecallSandbox| -> f64 {
        let out = search_trace(sandbox, "quokka");
        traces(&out)
            .into_iter()
            .find(|(f, _)| f == "old.md")
            .and_then(|(_, t)| t["decay_factor"].as_f64())
            .expect("old.md decay factor")
    };

    // At the 90-day default a 2020 note sits on the floor. A ten-year
    // half-life has to lift it well clear of that.
    let with_default = old_note_factor(&sandbox);
    sandbox
        .cmd()
        .args(["collection", "half-life", "notes", "3650"])
        .assert()
        .success();
    let with_collection_half_life = old_note_factor(&sandbox);

    assert!(
        with_collection_half_life > with_default + 0.05,
        "collection half-life did not override the default: {with_default} -> {with_collection_half_life}"
    );
}
