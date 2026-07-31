//! Recency decay e2e tests. BM25-only (no Ollama needed), driven through
//! `recall search --trace` so the assertions read the same numbers a user
//! debugging their ranking would see.

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

fn setup(decay_config: &str) -> (RecallSandbox, TempDir) {
    let sandbox = RecallSandbox::new();
    let vault = vault_with_old_and_new_notes();
    std::fs::write(sandbox.config_path(), decay_config).unwrap();

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

#[test]
fn decay_is_off_by_default() {
    let (sandbox, _vault) = setup("");
    sandbox
        .cmd()
        .args(["search", "quokka", "--trace"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let rows = traces(out);
            !rows.is_empty() && rows.iter().all(|(_, t)| t["decay_factor"].is_null())
        }));
}

#[test]
fn enabled_decay_penalizes_the_older_note() {
    let (sandbox, _vault) = setup("[decay]\nenabled = true\ndefault_half_life_days = 30.0\n");
    sandbox
        .cmd()
        .args(["search", "quokka", "--trace"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let rows = traces(out);
            let factor = |name: &str| {
                rows.iter()
                    .find(|(f, _)| f == name)
                    .and_then(|(_, t)| t["decay_factor"].as_f64())
            };
            let (old, new) = match (factor("old.md"), factor("new.md")) {
                (Some(o), Some(n)) => (o, n),
                _ => return false,
            };
            // A 2020 note is many half-lives old: pinned to the floor. The
            // undated note dates from its mtime — indexed seconds ago.
            (old - 0.5).abs() < 1e-6 && (new - 1.0).abs() < 1e-6
        }));
}

#[test]
fn decay_records_the_pre_decay_score() {
    let (sandbox, _vault) = setup("[decay]\nenabled = true\ndefault_half_life_days = 30.0\n");
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

#[test]
fn temporal_queries_skip_decay() {
    let (sandbox, _vault) = setup("[decay]\nenabled = true\ndefault_half_life_days = 30.0\n");
    sandbox
        .cmd()
        .args(["search", "quokka 2020", "--trace"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            if v["intent"]["kind"] != "temporal" {
                return false;
            }
            let rows = traces(out);
            !rows.is_empty() && rows.iter().all(|(_, t)| t["decay_factor"].is_null())
        }));
}

#[test]
fn a_collection_half_life_overrides_the_config_default() {
    let (sandbox, _vault) = setup("[decay]\nenabled = true\ndefault_half_life_days = 30.0\n");

    let old_note_factor = |sandbox: &RecallSandbox| -> f64 {
        let out = sandbox
            .cmd()
            .args(["search", "quokka", "--trace"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let out = String::from_utf8(out).unwrap();
        traces(&out)
            .into_iter()
            .find(|(f, _)| f == "old.md")
            .and_then(|(_, t)| t["decay_factor"].as_f64())
            .expect("old.md decay factor")
    };

    // At the 30-day config default a 2020 note sits on the floor. A much
    // longer collection half-life has to lift it well clear of that.
    let with_config_default = old_note_factor(&sandbox);
    sandbox
        .cmd()
        .args(["collection", "half-life", "notes", "3650"])
        .assert()
        .success();
    let with_collection_half_life = old_note_factor(&sandbox);

    assert!(
        with_collection_half_life > with_config_default + 0.05,
        "collection half-life did not override the config default: {with_config_default} -> {with_collection_half_life}"
    );
}
