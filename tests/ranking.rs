//! Retrieval regression harness: a fixed fixture vault, a fixed query set,
//! and two ways of catching a ranking change.
//!
//! Ranking moved three ways in this branch — date filters that were silently
//! dropped in hybrid mode now run, recency decay multiplies the final score,
//! and a query naming a year gets a date bound. None of that is attributable
//! without a baseline, so this file supplies one:
//!
//! 1. **Differential snapshot.** Every query's ranked chunk list is compared
//!    against `tests/snapshots/ranking.json`. A ranking change becomes a
//!    readable diff instead of a surprise. Regenerate deliberately with
//!    `RECALL_UPDATE_SNAPSHOTS=1`.
//! 2. **Labeled queries.** Sixteen queries with a known-correct top result,
//!    scored as a hit rate. Two of them are supersession pairs — an old,
//!    keyword-dense decision record against the short note that overturned it
//!    — and they are the entire reason decay exists.
//!
//! Decay is unconditional, so there is no "decay off" profile to run. The
//! comparison it used to provide comes from `--trace` instead: `pre_decay_score`
//! is the score the pipeline would have returned without decay, so sorting on
//! it reconstructs the pre-decay ranking from the same single search. That is
//! a stronger baseline than the old config toggle — it is measured from the
//! shipped code path rather than from a second one.
//!
//! Hermetic and BM25-only: temp dirs throughout (via `RecallSandbox`), no
//! embeddings, no model weights, no LLM. Every fixture date is written relative to
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
    ("bge small", "reference/local-embeddings.md"),
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
/// Queries stay under five tokens and carry no year or relative time word, so
/// no query picks up an implicit `after` bound or skips decay. Reranking needs
/// a network and is never routed, so it stays off.
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

    /// One search, two orderings.
    ///
    /// `.0` is the ranking as returned — decay applied. `.1` is the same
    /// chunks re-sorted by `pre_decay_score`, which is what the pipeline would
    /// have returned had decay not run. Both come from a single `--trace`
    /// search, so the baseline is measured from the shipped code path.
    ///
    /// `pre_decay_score` is null for a chunk decay skipped (no usable date);
    /// its score is already the pre-decay one, so it stands in unchanged.
    fn rankings(&self, query: &str) -> (Vec<String>, Vec<String>) {
        let output = self
            .sandbox
            .cmd()
            .args(["search", query, "--trace", "--limit", LIMIT])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let json: serde_json::Value =
            serde_json::from_slice(&output).expect("search must emit JSON");

        // `relative/path.md:start_line` — the line number matters, because a
        // note splits into several chunks and a change in *which* chunk of a
        // note wins is a ranking change too.
        let mut scored: Vec<(String, f64)> = json["results"]
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
                let pre = r["trace"]["pre_decay_score"]
                    .as_f64()
                    .unwrap_or_else(|| r["score"].as_f64().expect("score"));
                (format!("{relative}:{start}"), pre)
            })
            .collect();

        let ranked: Vec<String> = scored.iter().map(|(id, _)| id.clone()).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("comparable scores"));
        let pre_decay: Vec<String> = scored.into_iter().map(|(id, _)| id).collect();
        (ranked, pre_decay)
    }

    fn ranked(&self, query: &str) -> Vec<String> {
        self.rankings(query).0
    }

    /// Relative path of the top result, ignoring which chunk of it won.
    fn top_of(ranked: &[String]) -> Option<String> {
        ranked
            .first()
            .and_then(|entry| entry.rsplit_once(':').map(|(path, _)| path.to_string()))
    }

    /// Best rank any chunk of `path` reached, panicking if the note is
    /// missing — an absent note would otherwise read as a very good rank.
    fn rank_of(ranked: &[String], path: &str) -> usize {
        ranked
            .iter()
            .position(|entry| entry.starts_with(&format!("{path}:")))
            .unwrap_or_else(|| panic!("no chunk of {path} in {ranked:?}"))
    }

    fn capture(&self) -> Ranking {
        snapshot_queries()
            .map(|q| (q.to_string(), self.ranked(q)))
            .collect()
    }

    /// Labeled queries whose top result is not the expected note, scored
    /// twice from the same searches: as ranked, and as the pre-decay order
    /// would have ranked them.
    fn missed(&self) -> (Vec<&'static str>, Vec<&'static str>) {
        let mut missed = Vec::new();
        let mut missed_pre_decay = Vec::new();
        for (query, expected) in LABELED {
            let (ranked, pre_decay) = self.rankings(query);
            if Self::top_of(&ranked).as_deref() != Some(*expected) {
                missed.push(*query);
            }
            if Self::top_of(&pre_decay).as_deref() != Some(*expected) {
                missed_pre_decay.push(*query);
            }
        }
        (missed, missed_pre_decay)
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
            "reference/local-embeddings.md",
            format!(
                "---\ndate: {}\nstatus: current\n---\n\n\
                 # Local embeddings\n\n\
                 The bge-small-en-v1.5 model returns a 384 dimension vector per input and\n\
                 runs entirely in process on the local machine.\n",
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

fn diff(expected: &Ranking, actual: &Ranking) -> Vec<String> {
    let mut lines = Vec::new();
    for query in expected.keys().chain(actual.keys()).collect::<Vec<_>>() {
        let (want, got) = (expected.get(query), actual.get(query));
        if want != got {
            lines.push(format!(
                "{query:?}\n  snapshot: {:?}\n  actual:   {:?}",
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
    let actual = harness.capture();
    let path = snapshot_path();

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
    let expected: Ranking = serde_json::from_str(&raw).expect("parse snapshot");

    let lines = diff(&expected, &actual);
    assert!(
        lines.is_empty(),
        "ranking changed against {}:\n{}\n\nIf the change is intended, regenerate with \
         RECALL_UPDATE_SNAPSHOTS=1 cargo test --test ranking",
        path.display(),
        lines.join("\n")
    );
}

/// The headline measurement: decay answers every labeled query, and the
/// pre-decay order — the ranking recall would return without it — misses
/// exactly the two supersession pairs. A different miss list means the fixture
/// drifted and the comparison no longer measures decay.
#[test]
fn decay_answers_every_labeled_query_and_the_pre_decay_order_does_not() {
    let harness = Harness::new();
    let (missed, missed_pre_decay) = harness.missed();

    assert!(
        missed.is_empty(),
        "decay must answer every labeled query; missed {missed:?}"
    );
    assert_eq!(
        missed_pre_decay.as_slice(),
        DECAY_SENSITIVE,
        "pre-decay order missed {missed_pre_decay:?}, expected exactly {DECAY_SENSITIVE:?}"
    );
}

/// The two supersession pairs, chunk by chunk: relevance alone puts the stale
/// note first, and decay is what lifts current truth to rank 1.
#[test]
fn decay_lifts_current_truth_over_a_superseded_note() {
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
        let (ranked, pre_decay) = harness.rankings(query);

        assert!(
            Harness::rank_of(&pre_decay, stale) < Harness::rank_of(&pre_decay, current),
            "{query:?}: the stale note must win on relevance alone, otherwise this \
             fixture proves nothing about decay"
        );
        assert_eq!(
            Harness::rank_of(&ranked, current),
            0,
            "{query:?}: decay must lift {current} over {stale}"
        );
    }
}
