//! Maintenance subcommand e2e: check / vacuum / rebuild-fts.

mod common;

use common::{write_fixture_vault, RecallSandbox};
use predicates::prelude::*;
use tempfile::tempdir;

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
fn maintenance_check_reports_healthy_db() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());
    add_and_index(&sandbox, vault.path(), "test");

    sandbox
        .cmd()
        .args(["maintenance", "check", "--json"])
        .assert()
        .success()
        .stdout(predicate::function(|out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap();
            v["integrity"] == "ok"
                && v["orphans"]["chunks"].as_i64() == Some(0)
                && v["orphans"]["files"].as_i64() == Some(0)
                && v["orphans"]["embeddings"].as_i64() == Some(0)
                && v["healthy"].as_bool() == Some(true)
        }));
}

#[test]
fn maintenance_default_action_is_check() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());
    add_and_index(&sandbox, vault.path(), "test");

    sandbox
        .cmd()
        .args(["maintenance"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Integrity:").and(predicate::str::contains("healthy")));
}

#[test]
fn maintenance_vacuum_succeeds() {
    let sandbox = RecallSandbox::new();
    sandbox
        .cmd()
        .args(["maintenance", "vacuum"])
        .assert()
        .success()
        .stdout(predicate::str::contains("VACUUM complete"));
}

#[test]
fn maintenance_rebuild_fts_then_search_still_works() {
    let sandbox = RecallSandbox::new();
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());
    add_and_index(&sandbox, vault.path(), "test");

    sandbox
        .cmd()
        .args(["maintenance", "rebuild-fts"])
        .assert()
        .success()
        .stdout(predicate::str::contains("FTS5 index rebuilt"));

    // Sanity-check: search still hits after rebuild
    sandbox
        .cmd()
        .args(["search", "Paxos", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("gamma.md"));
}
