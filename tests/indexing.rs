//! Indexing metadata e2e tests: the frontmatter → filename → mtime date
//! cascade as it lands in search results, and the index fingerprint that
//! cold-rebuilds the DB when the pipeline changes.

mod common;

use common::RecallSandbox;
use predicates::prelude::*;
use tempfile::tempdir;

fn first_result(out: &str) -> serde_json::Value {
    let v: serde_json::Value = serde_json::from_str(out).expect("json");
    v["results"][0].clone()
}

#[test]
fn frontmatter_metadata_reaches_search_results() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    std::fs::write(
        vault.path().join("2020-01-01.md"),
        "---\ndate: 2026-03-04\nstatus: active\ntype: project\n---\n\n# Note\n\nUniqueZeta token.\n",
    )
    .unwrap();

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes"])
        .assert()
        .success();
    sandbox.cmd().args(["index"]).assert().success();

    sandbox
        .cmd()
        .args(["search", "UniqueZeta", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let r = first_result(out);
            // Frontmatter beats the YYYY-MM-DD filename.
            r["date"] == "2026-03-04"
                && r["date_source"] == "frontmatter"
                && r["status"] == "active"
                && r["memory_type"] == "semantic"
        }));
}

#[test]
fn undated_files_fall_back_to_mtime() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    std::fs::write(
        vault.path().join("plain.md"),
        "# Plain\n\nUniqueEta token.\n",
    )
    .unwrap();

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes"])
        .assert()
        .success();
    sandbox.cmd().args(["index"]).assert().success();

    sandbox
        .cmd()
        .args(["search", "UniqueEta", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let r = first_result(out);
            r["date_source"] == "mtime" && r["date"].as_str().unwrap().len() == 10
        }));
}

#[test]
fn changing_the_embedding_model_cold_rebuilds_the_index() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    std::fs::write(vault.path().join("a.md"), "# A\n\nUniqueTheta token.\n").unwrap();

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes"])
        .assert()
        .success();
    sandbox.cmd().args(["index"]).assert().success();

    // A different embedding model invalidates the stored vectors, so the
    // fingerprint no longer matches and the indexed data is discarded.
    std::fs::write(
        sandbox.config_path(),
        "[embeddings]\nmodel = \"some-other-model\"\n",
    )
    .unwrap();

    sandbox
        .cmd()
        .args(["status", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v["chunk_count"] == 0 && v["file_count"] == 0
        }));

    // Collections survive the rebuild; re-indexing refills the chunks.
    sandbox
        .cmd()
        .args(["collection", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("notes"));
    sandbox.cmd().args(["index"]).assert().success();
    sandbox
        .cmd()
        .args(["search", "UniqueTheta", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("a.md"));
}

#[test]
fn collection_half_life_round_trips() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    std::fs::write(vault.path().join("a.md"), "# A\n\nBody.\n").unwrap();

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes", "--half-life-days", "30"])
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["collection", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v[0]["half_life_days"] == 30.0
        }));

    sandbox
        .cmd()
        .args(["collection", "half-life", "notes", "180"])
        .assert()
        .success();
    sandbox
        .cmd()
        .args(["collection", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v[0]["half_life_days"] == 180.0
        }));

    // No value clears it back to NULL.
    sandbox
        .cmd()
        .args(["collection", "half-life", "notes"])
        .assert()
        .success();
    sandbox
        .cmd()
        .args(["collection", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v[0]["half_life_days"].is_null()
        }));
}

#[test]
fn half_life_on_unknown_collection_fails() {
    let sandbox = RecallSandbox::new();
    sandbox
        .cmd()
        .args(["collection", "half-life", "missing", "30"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

/// `--after` / `--before` are separate CLI flags that both land in
/// `SearchOptions`. `run_search` takes them as adjacent positional
/// `Option<String>` arguments, so a mis-ordered call site still compiles and
/// would silently swap the two bounds. This pins each flag to its own edge.
#[test]
fn search_date_bound_flags_are_wired_to_the_right_edges() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    std::fs::write(
        vault.path().join("2020-01-02.md"),
        "# Old\n\nUniqueOmicron token.\n",
    )
    .unwrap();
    std::fs::write(
        vault.path().join("2026-01-02.md"),
        "# New\n\nUniqueOmicron token.\n",
    )
    .unwrap();

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes"])
        .assert()
        .success();
    sandbox.cmd().args(["index"]).assert().success();

    let paths = |args: &[&str]| -> Vec<String> {
        let out = sandbox
            .cmd()
            .args(args)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let v: serde_json::Value = serde_json::from_slice(&out).expect("json");
        v["results"]
            .as_array()
            .expect("results")
            .iter()
            .map(|r| r["file"].as_str().unwrap_or_default().to_string())
            .collect()
    };

    let base = ["search", "UniqueOmicron", "--format", "json"];
    let all = paths(&base);
    assert_eq!(all.len(), 2, "both notes match before any bound: {all:?}");

    let recent = paths(&[&base[..], &["--after", "2025-01-01"]].concat());
    assert_eq!(recent.len(), 1);
    assert!(recent[0].ends_with("2026-01-02.md"), "{recent:?}");

    let old = paths(&[&base[..], &["--before", "2025-01-01"]].concat());
    assert_eq!(old.len(), 1);
    assert!(old[0].ends_with("2020-01-02.md"), "{old:?}");

    // Both bounds together describe a window that excludes everything.
    let none = paths(
        &[
            &base[..],
            &["--after", "2021-01-01", "--before", "2022-01-01"],
        ]
        .concat(),
    );
    assert!(none.is_empty(), "{none:?}");
}
