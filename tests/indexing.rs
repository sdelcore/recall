//! Indexing metadata e2e tests: the frontmatter → filename → mtime date
//! cascade as it lands in search results, and the exclusion list that the
//! indexer and the linter both obey.

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
        "---\ndate: 2026-03-04\nstatus: active\n---\n\n# Note\n\nUniqueZeta token.\n",
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

/// `recall index` reconciles, it does not only add. Deleting a note used to
/// leave its chunks searchable forever for anyone going through the MCP tool
/// or the watcher, because only the CLI's "full" mode dropped rows first.
#[test]
fn re_indexing_forgets_a_deleted_file() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    std::fs::write(
        vault.path().join("keep.md"),
        "# Keep\n\nUniqueKappa token.\n",
    )
    .unwrap();
    let doomed = vault.path().join("doomed.md");
    std::fs::write(&doomed, "# Doomed\n\nUniqueLambda token.\n").unwrap();

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes"])
        .assert()
        .success();
    sandbox.cmd().args(["index"]).assert().success();

    let hits = |token: &str| -> usize {
        let out = sandbox
            .cmd()
            .args(["search", token, "--format", "json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let v: serde_json::Value = serde_json::from_slice(&out).expect("json");
        v["results"].as_array().expect("results").len()
    };
    assert_eq!(hits("UniqueLambda"), 1, "the doomed note must index first");

    std::fs::remove_file(&doomed).unwrap();
    sandbox.cmd().args(["index"]).assert().success();

    assert_eq!(hits("UniqueLambda"), 0, "deleted note is still searchable");
    assert_eq!(hits("UniqueKappa"), 1, "the surviving note was collateral");
    sandbox
        .cmd()
        .args(["status", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v["file_count"] == 1 && v["orphans"]["chunks"] == 0 && v["healthy"] == true
        }));
}

/// A query that names a year gets that year as an `after` bound. The wildcard
/// form is the regression: a `*` used to classify the query as "structural",
/// which was checked first and suppressed the bound the query obviously wants.
#[test]
fn a_year_in_the_query_bounds_the_results() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    std::fs::write(
        vault.path().join("2020-05-05.md"),
        "# Old\n\nUniqueNu rollout notes for 2026 planning.\n",
    )
    .unwrap();
    std::fs::write(
        vault.path().join("2026-05-05.md"),
        "# New\n\nUniqueNu rollout notes for 2026 planning.\n",
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

    let paths = |query: &str| -> Vec<String> {
        let out = sandbox
            .cmd()
            .args(["search", query, "--format", "json"])
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

    let both = paths("UniqueNu rollout");
    assert_eq!(both.len(), 2, "{both:?}");

    for query in ["UniqueNu 2026", "UniqueNu* 2026"] {
        let bounded = paths(query);
        assert_eq!(bounded.len(), 1, "{query:?} -> {bounded:?}");
        assert!(bounded[0].ends_with("2026-05-05.md"), "{bounded:?}");
    }
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
        .args(["--name", "notes"])
        .assert()
        .success();

    // A new collection has no half-life until one is set.
    sandbox
        .cmd()
        .args(["collection", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v[0]["half_life_days"].is_null()
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

/// `collection half-life` is the only writer of the column, so its validation
/// is the only thing standing between a typo and a division by zero in
/// `recency_factor`.
#[test]
fn a_non_positive_half_life_is_rejected() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    std::fs::write(vault.path().join("a.md"), "# A\n\nBody.\n").unwrap();

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes"])
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["collection", "half-life", "notes", "0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must be positive"));
}

/// `--after` / `--before` are separate CLI flags that both land in
/// `SearchOptions`, adjacent to each other in `SearchRequest` and identically
/// typed. This pins each flag to its own edge.
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

/// The indexer and the linter walk the same roots, so they must agree on what
/// a note is. They used to read two different lists — the indexer's globs were
/// hardcoded and the linter read `[index] exclude`, which the indexer ignored —
/// so an excluded note was unsearchable *and* reported as an orphan.
#[test]
fn the_indexer_and_the_linter_share_one_exclusion_list() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    std::fs::create_dir_all(vault.path().join("Templates")).unwrap();
    std::fs::write(
        vault.path().join("Templates/daily.md"),
        "# Template\n\nUniqueKappa placeholder.\n",
    )
    .unwrap();
    std::fs::write(
        vault.path().join("real.md"),
        "# Real\n\nUniqueKappa content.\n",
    )
    .unwrap();
    std::fs::write(
        vault.path().join("note.sync-conflict-20260101-ABCDEF.md"),
        "# Conflict\n\nUniqueKappa duplicate.\n",
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

    // Only the real note is indexed.
    sandbox
        .cmd()
        .args(["search", "UniqueKappa", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            let files = v["results"].as_array().unwrap();
            files.len() == 1 && files[0]["file"].as_str().unwrap().ends_with("real.md")
        }));

    // And the linter scanned exactly the same one note.
    sandbox
        .cmd()
        .args(["lint", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v["notes_scanned"] == 1 && v["orphans"].as_array().unwrap().len() == 1
        }));
}
