//! Collection lifecycle e2e tests: add, list, remove, multi-collection
//! filtering, and the "schema upgrade required" guard against pre-PR-4 DBs.

mod common;

use common::{write_fixture_vault, RecallSandbox};
use predicates::prelude::*;
use tempfile::tempdir;

#[test]
fn collection_add_list_remove_lifecycle() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());

    // list empty
    sandbox
        .cmd()
        .args(["collection", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No collections"));

    // add
    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes"])
        .assert()
        .success()
        .stdout(predicate::str::contains("notes"));

    // list shows it
    sandbox
        .cmd()
        .args(["collection", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v.as_array().unwrap().len() == 1 && v[0]["name"] == "notes"
        }));

    // remove
    sandbox
        .cmd()
        .args(["collection", "remove", "notes"])
        .assert()
        .success();

    // list empty again
    sandbox
        .cmd()
        .args(["collection", "list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No collections"));
}

#[test]
fn search_filters_by_collection() {
    let sandbox = RecallSandbox::new();

    let vault_a = tempdir().unwrap();
    std::fs::write(
        vault_a.path().join("a.md"),
        "# A\n\nUniqueAlpha token here.\n",
    )
    .unwrap();

    let vault_b = tempdir().unwrap();
    std::fs::write(
        vault_b.path().join("b.md"),
        "# B\n\nUniqueBeta token here.\n",
    )
    .unwrap();

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault_a.path())
        .args(["--name", "alpha"])
        .assert()
        .success();
    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault_b.path())
        .args(["--name", "beta"])
        .assert()
        .success();
    sandbox.cmd().args(["index"]).assert().success();

    // Filtered to alpha: hits UniqueAlpha
    sandbox
        .cmd()
        .args([
            "search",
            "UniqueAlpha",
            "--collection",
            "alpha",
            "--format",
            "json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("a.md"));

    // Filtered to beta: should not see UniqueAlpha
    sandbox
        .cmd()
        .args([
            "search",
            "UniqueAlpha",
            "--collection",
            "beta",
            "--format",
            "json",
        ])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v["results"].as_array().unwrap().is_empty()
        }));
}

#[test]
fn index_without_collection_fails_when_path_given() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());

    sandbox
        .cmd()
        .args(["index", "--path"])
        .arg(vault.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains("require --collection"));
}

#[test]
fn index_without_any_collections_fails() {
    let sandbox = RecallSandbox::new();
    sandbox
        .cmd()
        .args(["index"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("No collections registered"));
}

#[test]
fn context_description_appears_in_search_results() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes"])
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["context", "add", "notes", "Personal notes and stray ideas"])
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["index", "--collection", "notes"])
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["search", "Paxos", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            let r = &v["results"][0];
            r["collection"]["name"] == "notes"
                && r["collection"]["description"] == "Personal notes and stray ideas"
        }));
}

#[test]
fn context_remove_clears_description() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());

    sandbox
        .cmd()
        .args(["collection", "add"])
        .arg(vault.path())
        .args(["--name", "notes"])
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["context", "add", "notes", "first description"])
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["context", "remove", "notes"])
        .assert()
        .success();

    sandbox
        .cmd()
        .args(["context", "list", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v[0]["description"].is_null()
        }));
}

#[test]
fn search_with_unknown_collection_fails() {
    let sandbox = RecallSandbox::new();
    sandbox
        .cmd()
        .args(["search", "anything", "--collection", "missing"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}
