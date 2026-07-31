//! Retrieval regression harness: a fixed fixture vault, a fixed query set,
//! and two ways of catching a ranking change.
//!
//! Ranking moved three ways in this branch — date filters that were silently
//! dropped in hybrid mode now run, recency decay multiplies the final score,
//! and the MCP server routes intent. None of that is attributable without a
//! baseline, so this file supplies one:
//!
//! 1. **Differential snapshot.** Every query's ranked chunk list is compared
//!    against `tests/snapshots/ranking.json`, once with decay off and once
//!    with it on. A ranking change becomes a readable diff instead of a
//!    surprise. Regenerate deliberately with `RECALL_UPDATE_SNAPSHOTS=1`.
//! 2. **Labeled queries.** Sixteen queries with a known-correct top result,
//!    scored as a hit rate under both profiles. Two of them are supersession
//!    pairs — an old, keyword-dense decision record against the short note
//!    that overturned it — and they are the entire reason decay exists. If
//!    the hit rate does not improve when decay is switched on, decay is not
//!    working.
//!
//! Hermetic and BM25-only: temp dirs throughout (via `RecallSandbox`), no
//! embeddings, no Ollama, no LLM. Every fixture date is written relative to
//! today, so the corpus ages with the calendar and the snapshot does not rot.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{Duration, Utc};
use common::RecallSandbox;
use tempfile::{tempdir, TempDir};

/// Results captured per query. Deep enough to show a reorder below the
/// winner, shallow enough to keep the snapshot readable.
const LIMIT: &str = "5";

/// Long enough that the year-old reference notes are not yet pinned to the
/// 0.5 floor, short enough that a superseded decision record is.
const DECAY_CONFIG: &str = "[decay]\nenabled = true\ndefault_half_life_days = 90.0\n";

/// Queries whose correct top result is known. The answer is what a person
/// asking that question wants *today*, which is why `primary datastore`
/// resolves to the Postgres note and `mongodb replica` still resolves to the
/// superseded one — naming the old thing explicitly must keep working.
const LABELED: &[(&str, &str)] = &[
    ("primary datastore", "decisions/datastore-postgres.md"),
    ("mongodb replica", "decisions/datastore-mongodb.md"),
    ("postgres joins", "decisions/datastore-postgres.md"),
    ("deploy target", "decisions/deploy-nixos.md"),
    ("heroku dynos", "decisions/deploy-heroku.md"),
    ("nixos flake", "decisions/deploy-nixos.md"),
    ("half-life decay", "projects/recall-overview.md"),
    ("orchestrator persona", "projects/aria-overview.md"),
    ("reciprocal rank fusion", "reference/rank-fusion.md"),
    ("unicode61 tokenizer", "reference/sqlite-fts5.md"),
    ("nomic embed", "reference/ollama-embeddings.md"),
    ("batch size", "sessions/embedding-tuning.md"),
    ("trigger sync", "sessions/fts-debugging.md"),
    ("dentist appointment", "notes/standup.md"),
    ("grocery list", "notes/inbox.md"),
    ("thinkpad dock", "archive/laptop-setup.md"),
];

/// The two labeled queries decay is expected to fix, and only these two.
/// Both ask about a topic by its generic name, where the superseded record
/// is the keyword-dense one.
const DECAY_SENSITIVE: &[&str] = &["primary datastore", "deploy target"];

/// The snapshot set: every labeled query plus eight broader ones that return
/// several competing chunks, so a reorder below rank 1 still shows up.
/// Queries stay under five tokens and carry no `?`, year, glob, or `.md`, to
/// keep the intent classifier on `Lookup` and the reranker (which would need
/// a network) switched off.
///
/// `runs` is the widest age spread in the corpus — an eight-year-old note, a
/// five-year-old one, and a one-year-old one — so it is the most sensitive
/// entry to a change in the decay curve. `status current` matches the
/// *frontmatter* text, which comrak still parses as body: the snapshot
/// records that leak so that fixing it shows up as a diff rather than as a
/// silent recall change.
const EXTRA_QUERIES: &[&str] = &[
    "datastore",
    "deploy",
    "schema",
    "vault",
    "index",
    "rank",
    "runs",
    "status current",
];

/// One profile's snapshot: query -> ranked `relative/path.md:start_line`.
type Ranking = BTreeMap<String, Vec<String>>;

fn snapshot_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots/ranking.json")
}

fn snapshot_queries() -> impl Iterator<Item = &'static str> {
    LABELED
        .iter()
        .map(|(query, _)| *query)
        .chain(EXTRA_QUERIES.iter().copied())
}

fn days_ago(n: i64) -> String {
    (Utc::now().date_naive() - Duration::days(n))
        .format("%Y-%m-%d")
        .to_string()
}

/// An indexed fixture vault plus the sandbox that owns its database.
///
/// The vault is indexed once; `set_decay` only rewrites `config.toml`, which
/// every subsequent `recall` process re-reads. `[decay]` is not part of the
/// index fingerprint, so flipping it never triggers a rebuild.
struct Harness {
    sandbox: RecallSandbox,
    /// Kept alive so the vault outlives the harness; the indexed paths live
    /// in `root`.
    _vault: TempDir,
    /// Canonical vault root with a trailing separator. `collection add`
    /// canonicalizes what it stores, and on macOS a tempdir is reached
    /// through a symlink (`/var` -> `/private/var`), so stripping the
    /// un-canonicalized path would leave absolute paths in the snapshot.
    root: String,
}

impl Harness {
    fn new() -> Self {
        let sandbox = RecallSandbox::new();
        let vault = tempdir().expect("vault tempdir");
        write_ranking_vault(vault.path());
        std::fs::write(sandbox.config_path(), "").expect("config");

        sandbox
            .cmd()
            .args(["collection", "add"])
            .arg(vault.path())
            .args(["--name", "vault"])
            .assert()
            .success();
        sandbox.cmd().args(["index"]).assert().success();

        let canonical = std::fs::canonicalize(vault.path()).expect("canonical vault");
        let root = format!(
            "{}{}",
            canonical.to_string_lossy(),
            std::path::MAIN_SEPARATOR
        );
        Self {
            sandbox,
            _vault: vault,
            root,
        }
    }

    fn set_decay(&self, enabled: bool) {
        let body = if enabled { DECAY_CONFIG } else { "" };
        std::fs::write(self.sandbox.config_path(), body).expect("config");
    }

    /// Ranked chunk identities for one query: `relative/path.md:start_line`.
    /// The line number matters — a note splits into several chunks, and a
    /// change in *which* chunk of a note wins is a ranking change too.
    fn ranked(&self, query: &str) -> Vec<String> {
        let output = self
            .sandbox
            .cmd()
            .args(["search", query, "--format", "json", "--limit", LIMIT])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let json: serde_json::Value =
            serde_json::from_slice(&output).expect("search must emit JSON");
        json["results"]
            .as_array()
            .expect("results array")
            .iter()
            .map(|r| {
                let file = r["file"].as_str().expect("file");
                let relative = file
                    .strip_prefix(&self.root)
                    .unwrap_or_else(|| panic!("{file} is not under {}", self.root));
                let line = r["lines"].as_str().expect("lines");
                let start = line.split('-').next().unwrap_or(line);
                format!("{relative}:{start}")
            })
            .collect()
    }

    /// Relative path of the top result, ignoring which chunk of it won.
    fn top(&self, query: &str) -> Option<String> {
        self.ranked(query)
            .first()
            .and_then(|entry| entry.rsplit_once(':').map(|(path, _)| path.to_string()))
    }

    /// Best rank any chunk of `path` reached, panicking if the note is
    /// missing — an absent note would otherwise read as a very good rank.
    fn rank_of(&self, query: &str, path: &str) -> usize {
        let ranked = self.ranked(query);
        ranked
            .iter()
            .position(|entry| entry.starts_with(&format!("{path}:")))
            .unwrap_or_else(|| panic!("{query:?} returned no chunk of {path}: {ranked:?}"))
    }

    fn capture(&self) -> Ranking {
        snapshot_queries()
            .map(|q| (q.to_string(), self.ranked(q)))
            .collect()
    }

    /// Fraction of labeled queries whose top result is the expected note,
    /// plus the queries that missed.
    fn hit_rate(&self) -> (usize, Vec<&'static str>) {
        let mut hits = 0;
        let mut missed = Vec::new();
        for (query, expected) in LABELED {
            if self.top(query).as_deref() == Some(*expected) {
                hits += 1;
            } else {
                missed.push(*query);
            }
        }
        (hits, missed)
    }
}

/// Fourteen notes covering the metadata the ranker reads: frontmatter dates
/// spanning eight years, `status`, `type`, and one note with no frontmatter
/// at all so the mtime rung of the date cascade is exercised.
///
/// Dates are relative to today rather than literal, so the corpus keeps the
/// same ages forever and the decay snapshot stays valid. Filenames are
/// deliberately *not* date-shaped — a `YYYY-MM-DD.md` name would change
/// every day and churn the snapshot; `tests/indexing.rs` covers that rung.
///
/// The two `decisions/` pairs are the point of the corpus: in each, the
/// superseded note repeats the generic term ("primary datastore", "deploy
/// target") and the current note mentions it once.
fn write_ranking_vault(root: &Path) {
    let files: &[(&str, String)] = &[
        (
            "decisions/datastore-mongodb.md",
            format!(
                "---\ndate: {}\nstatus: superseded\ntype: project\n---\n\n\
                 # Datastore choice\n\n\
                 MongoDB is the primary datastore. The primary datastore holds user\n\
                 documents, and its replica set runs three nodes across two racks.\n",
                days_ago(2700)
            ),
        ),
        (
            "decisions/datastore-postgres.md",
            format!(
                "---\ndate: {}\nstatus: current\ntype: project\n---\n\n\
                 # Datastore choice, revisited\n\n\
                 Postgres replaced MongoDB as the primary datastore. Joins and\n\
                 constraints turned out to be worth more than a loose schema.\n",
                days_ago(4)
            ),
        ),
        (
            "decisions/deploy-heroku.md",
            format!(
                "---\ndate: {}\nstatus: superseded\ntype: project\n---\n\n\
                 # Deploy target\n\n\
                 Heroku is the deploy target. The deploy target runs two web dynos and\n\
                 one worker dyno, and the deploy target scales by sliding a dyno slider.\n",
                days_ago(1850)
            ),
        ),
        (
            "decisions/deploy-nixos.md",
            format!(
                "---\ndate: {}\nstatus: current\ntype: project\n---\n\n\
                 # Deploy target, revisited\n\n\
                 A NixOS flake is the deploy target now. Rebuilds are reproducible and\n\
                 a rollback is one generation away.\n",
                days_ago(12)
            ),
        ),
        (
            "projects/recall-overview.md",
            format!(
                "---\ndate: {}\nstatus: current\ntype: project\n---\n\n\
                 # Recall\n\n\
                 Recall ranks vault chunks by relevance and then applies a half-life\n\
                 decay so that a stale note loses ground to a fresh one. The decay floor\n\
                 keeps recency from overwhelming relevance.\n",
                days_ago(60)
            ),
        ),
        (
            "projects/aria-overview.md",
            format!(
                "---\ndate: {}\nstatus: current\ntype: project\n---\n\n\
                 # Aria\n\n\
                 Aria routes every turn through an orchestrator that delegates work to a\n\
                 persona. Gateways carry messages in, and the vault carries knowledge out.\n",
                days_ago(45)
            ),
        ),
        (
            "reference/rank-fusion.md",
            format!(
                "---\ndate: {}\nstatus: current\n---\n\n\
                 # Rank fusion\n\n\
                 Reciprocal rank fusion merges two ranked lists by summing one over k\n\
                 plus rank. It needs no score calibration between the two retrievers.\n",
                days_ago(300)
            ),
        ),
        (
            "reference/sqlite-fts5.md",
            format!(
                "---\ndate: {}\nstatus: current\n---\n\n\
                 # FTS5\n\n\
                 The unicode61 tokenizer folds diacritics and splits on punctuation.\n\
                 Ranking uses the built in bm25 function, which returns negative scores.\n",
                days_ago(600)
            ),
        ),
        (
            "reference/ollama-embeddings.md",
            format!(
                "---\ndate: {}\nstatus: current\n---\n\n\
                 # Ollama embeddings\n\n\
                 The nomic-embed-text model returns a 768 dimension vector per input and\n\
                 runs entirely on the local machine.\n",
                days_ago(400)
            ),
        ),
        (
            "sessions/embedding-tuning.md",
            format!(
                "---\ndate: {}\ntype: session\n---\n\n\
                 # Embedding tuning\n\n\
                 Raised the batch size from eight to sixty four. Throughput tripled and\n\
                 the queue drained before the next sweep started.\n",
                days_ago(167)
            ),
        ),
        (
            "sessions/fts-debugging.md",
            format!(
                "---\ndate: {}\ntype: session\n---\n\n\
                 # FTS debugging\n\n\
                 The trigger that keeps the index in sync fired before the row landed, so\n\
                 a delete left an orphan row behind. Rewrote the trigger as an after clause.\n",
                days_ago(270)
            ),
        ),
        (
            "notes/standup.md",
            format!(
                "---\ndate: {}\ntype: scratchpad\n---\n\n\
                 # Standup\n\n\
                 Booked a dentist appointment for Thursday afternoon and moved the review\n\
                 to Friday.\n",
                days_ago(11)
            ),
        ),
        (
            // No frontmatter: the date cascade falls through to the mtime,
            // which is now, so this note never decays.
            "notes/inbox.md",
            "# Inbox\n\nGrocery list: oat milk, rye bread, tinned tomatoes, and a bag of coffee.\n"
                .to_string(),
        ),
        (
            "archive/laptop-setup.md",
            format!(
                "---\ndate: {}\nstatus: archived\n---\n\n\
                 # Laptop setup\n\n\
                 The old Thinkpad drives two monitors through a dock. Suspend needs a\n\
                 kernel parameter to survive a lid close.\n",
                days_ago(3000)
            ),
        ),
    ];

    for (relative, body) in files {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(&path, body).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
}

fn diff(profile: &str, expected: &Ranking, actual: &Ranking) -> Vec<String> {
    let mut lines = Vec::new();
    for query in expected.keys().chain(actual.keys()).collect::<Vec<_>>() {
        let (want, got) = (expected.get(query), actual.get(query));
        if want != got {
            lines.push(format!(
                "[{profile}] {query:?}\n  snapshot: {:?}\n  actual:   {:?}",
                want.map(Vec::as_slice).unwrap_or(&[]),
                got.map(Vec::as_slice).unwrap_or(&[])
            ));
        }
    }
    lines.sort();
    lines.dedup();
    lines
}

#[test]
fn ranked_output_matches_the_snapshot() {
    let harness = Harness::new();

    harness.set_decay(false);
    let baseline = harness.capture();
    harness.set_decay(true);
    let decayed = harness.capture();

    let path = snapshot_path();
    let actual: BTreeMap<&str, &Ranking> = [("baseline", &baseline), ("decay", &decayed)]
        .into_iter()
        .collect();

    if std::env::var_os("RECALL_UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        let mut json = serde_json::to_string_pretty(&actual).expect("serialize");
        json.push('\n');
        std::fs::write(&path, json).expect("write snapshot");
        return;
    }

    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing snapshot {}: {e}\nregenerate with RECALL_UPDATE_SNAPSHOTS=1 cargo test",
            path.display()
        )
    });
    let expected: BTreeMap<String, Ranking> = serde_json::from_str(&raw).expect("parse snapshot");

    let mut lines = diff(
        "baseline",
        expected.get("baseline").expect("baseline profile"),
        &baseline,
    );
    lines.extend(diff(
        "decay",
        expected.get("decay").expect("decay profile"),
        &decayed,
    ));
    assert!(
        lines.is_empty(),
        "ranking changed against {}:\n{}\n\nIf the change is intended, regenerate with \
         RECALL_UPDATE_SNAPSHOTS=1 cargo test --test ranking",
        path.display(),
        lines.join("\n")
    );
}

#[test]
fn decay_improves_the_labeled_hit_rate() {
    let harness = Harness::new();

    harness.set_decay(false);
    let (baseline_hits, baseline_missed) = harness.hit_rate();
    harness.set_decay(true);
    let (decayed_hits, decayed_missed) = harness.hit_rate();

    assert!(
        decayed_missed.is_empty(),
        "decay must answer every labeled query; missed {decayed_missed:?}"
    );
    assert_eq!(decayed_hits, LABELED.len());

    // The baseline must miss exactly the supersession pairs. A different
    // miss list means the fixture drifted and the comparison is no longer
    // measuring decay.
    assert_eq!(
        baseline_missed.as_slice(),
        DECAY_SENSITIVE,
        "baseline missed {baseline_missed:?}, expected exactly {DECAY_SENSITIVE:?}"
    );
    assert_eq!(baseline_hits, LABELED.len() - DECAY_SENSITIVE.len());
}

#[test]
fn a_superseded_note_outranks_current_truth_until_decay_is_enabled() {
    let harness = Harness::new();
    let pairs = [
        (
            "primary datastore",
            "decisions/datastore-mongodb.md",
            "decisions/datastore-postgres.md",
        ),
        (
            "deploy target",
            "decisions/deploy-heroku.md",
            "decisions/deploy-nixos.md",
        ),
    ];

    for (query, stale, current) in pairs {
        harness.set_decay(false);
        assert!(
            harness.rank_of(query, stale) < harness.rank_of(query, current),
            "{query:?}: the stale note must win on relevance alone, otherwise this \
             fixture proves nothing about decay"
        );

        harness.set_decay(true);
        assert_eq!(
            harness.rank_of(query, current),
            0,
            "{query:?}: decay must lift {current} over {stale}"
        );
    }
}
