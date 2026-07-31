//! `recall lint` e2e tests. Filesystem-only — lint never reads the index, so
//! these tests never index anything and never need Ollama.

mod common;

use common::RecallSandbox;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::{tempdir, TempDir};

/// Two collections. `notes` links inside itself, into `work`, and at one note
/// that does not exist; `work` holds the cross-collection target.
fn two_vaults() -> (RecallSandbox, TempDir, TempDir) {
    let notes = tempdir().unwrap();
    let work = tempdir().unwrap();

    write(
        notes.path().join("index.md"),
        "# Index\n\n\
         Local link to [[Alpha]] and a piped one to [[Beta|the beta note]].\n\n\
         By alias: [[Codename Alpha]].\n\n\
         Cross-project: [[Roadmap#Q3]].\n\n\
         Broken: [[Missing Note]].\n\n\
         ```\n\
         sample [[Ghost In Code]]\n\
         ```\n\n\
         %% draft [[Ghost In Comment]] %%\n",
    );
    write(
        notes.path().join("alpha.md"),
        "---\naliases: [Codename Alpha]\n---\n\n# Alpha\n\nBody.\n",
    );
    write(notes.path().join("beta.md"), "# Beta\n\nBody.\n");
    write(
        notes.path().join("stray.md"),
        "# Stray\n\nNo links at all.\n",
    );
    write(
        notes.path().join("2026-07-30.md"),
        "# Daily\n\nNo links either.\n",
    );
    write(
        work.path().join("roadmap.md"),
        "# Roadmap\n\n## Q3\n\nWork.\n",
    );

    let sandbox = RecallSandbox::new();
    for (name, path) in [("notes", notes.path()), ("work", work.path())] {
        sandbox
            .cmd()
            .args(["collection", "add"])
            .arg(path)
            .args(["--name", name])
            .assert()
            .success();
    }
    (sandbox, notes, work)
}

fn write(path: std::path::PathBuf, contents: &str) {
    std::fs::write(&path, contents).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Run `recall lint --json [--collection <name>]` and parse the report.
fn lint_json(sandbox: &RecallSandbox, collection: Option<&str>) -> Value {
    let mut cmd = sandbox.cmd();
    cmd.args(["lint", "--json"]);
    if let Some(name) = collection {
        cmd.args(["--collection", name]);
    }
    let out = cmd.assert().success().get_output().stdout.clone();
    serde_json::from_slice(&out).expect("lint json")
}

fn targets(report: &Value, key: &str) -> Vec<String> {
    report[key]
        .as_array()
        .expect("array")
        .iter()
        .map(|l| l["target"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn dangling_link_is_the_only_unresolved_finding() {
    let (sandbox, _notes, _work) = two_vaults();
    let report = lint_json(&sandbox, Some("notes"));

    assert_eq!(targets(&report, "unresolved"), ["Missing Note"]);
    assert_eq!(report["unresolved"][0]["state"], "unresolved");
    assert_eq!(report["unresolved"][0]["line"].as_u64(), Some(9));
}

#[test]
fn cross_collection_link_resolves_as_foreign() {
    let (sandbox, _notes, _work) = two_vaults();
    let report = lint_json(&sandbox, Some("notes"));

    assert_eq!(targets(&report, "foreign"), ["Roadmap"]);
    assert_eq!(report["foreign"][0]["state"], "resolved-foreign");
    assert_eq!(report["foreign"][0]["resolved_in"], "work");
}

#[test]
fn links_in_code_blocks_and_comments_are_ignored() {
    let (sandbox, _notes, _work) = two_vaults();
    let report = lint_json(&sandbox, Some("notes"));

    let all: Vec<String> = targets(&report, "unresolved")
        .into_iter()
        .chain(targets(&report, "foreign"))
        .collect();
    assert!(
        !all.iter().any(|t| t.starts_with("Ghost")),
        "code-block / comment links leaked into the report: {all:?}"
    );
    // Alpha, Beta, Codename Alpha, Roadmap, Missing Note — and nothing else.
    assert_eq!(report["links_total"].as_u64(), Some(5));
}

#[test]
fn frontmatter_alias_resolves_as_local() {
    let (sandbox, _notes, _work) = two_vaults();
    let report = lint_json(&sandbox, Some("notes"));

    assert!(
        !targets(&report, "unresolved").contains(&"Codename Alpha".to_string()),
        "alias link reported as dangling"
    );
    // Alpha, Beta, and the alias hit on alpha.md.
    assert_eq!(report["resolved_local"].as_u64(), Some(3));
}

#[test]
fn orphans_exclude_daily_notes_by_default() {
    let (sandbox, notes, _work) = two_vaults();
    let report = lint_json(&sandbox, Some("notes"));

    let orphans: Vec<&str> = report["orphans"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    assert_eq!(orphans.len(), 1, "got {orphans:?}");
    assert!(orphans[0].ends_with("stray.md"));
    assert!(
        !orphans[0].contains("2026-07-30"),
        "daily note should be excluded"
    );
    // Sanity: the daily note really is on disk and was scanned.
    assert!(notes.path().join("2026-07-30.md").exists());
    assert_eq!(report["notes_scanned"].as_u64(), Some(5));
}

#[test]
fn lint_is_warn_only_unless_asked_to_fail() {
    let (sandbox, _notes, _work) = two_vaults();

    sandbox
        .cmd()
        .args(["lint"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Unresolved links:"));

    sandbox
        .cmd()
        .args(["lint", "--fail-on-unresolved"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("1 unresolved link"));
}

#[test]
fn report_file_may_not_land_inside_a_collection() {
    let (sandbox, notes, _work) = two_vaults();

    sandbox
        .cmd()
        .args(["lint", "--out"])
        .arg(notes.path().join("lint-report.md"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("Refusing to write"));
    assert!(!notes.path().join("lint-report.md").exists());

    let outside = tempdir().unwrap();
    let dest = outside.path().join("lint-report.txt");
    sandbox
        .cmd()
        .args(["lint", "--out"])
        .arg(&dest)
        .assert()
        .success();
    assert!(std::fs::read_to_string(&dest)
        .unwrap()
        .contains("Recall Lint"));
}

#[test]
fn lint_without_collections_explains_itself() {
    let sandbox = RecallSandbox::new();
    sandbox
        .cmd()
        .args(["lint"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("No collections registered"));
}
