#![allow(dead_code)]
//! Shared test helpers: hermetic recall sandbox + fixture vault builders.
//!
//! Every integration test routes through `RecallSandbox`, which sets
//! `RECALL_DB_PATH` and `RECALL_CONFIG_PATH` to per-test tempdirs so the
//! suite cannot touch a developer's real database or config.

use assert_cmd::Command as AssertCommand;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

pub struct RecallSandbox {
    pub home: TempDir,
}

impl RecallSandbox {
    pub fn new() -> Self {
        Self {
            home: tempfile::tempdir().expect("tempdir"),
        }
    }

    pub fn db_path(&self) -> PathBuf {
        self.home.path().join("memory.sqlite")
    }

    pub fn config_path(&self) -> PathBuf {
        self.home.path().join("config.toml")
    }

    /// `recall <args>` with hermetic env vars applied.
    pub fn cmd(&self) -> AssertCommand {
        let mut c = AssertCommand::cargo_bin("recall").expect("recall binary built");
        c.env("RECALL_DB_PATH", self.db_path())
            .env("RECALL_CONFIG_PATH", self.config_path())
            .env("RUST_LOG", "warn");
        c
    }

    /// Same env vars but as `std::process::Command` for tests that need
    /// piped stdio (e.g. MCP server).
    pub fn raw_cmd(&self) -> std::process::Command {
        let exe = AssertCommand::cargo_bin("recall")
            .expect("recall binary built")
            .get_program()
            .to_owned();
        let mut c = std::process::Command::new(exe);
        c.env("RECALL_DB_PATH", self.db_path())
            .env("RECALL_CONFIG_PATH", self.config_path())
            .env("RUST_LOG", "warn");
        c
    }
}

/// Write a fixture markdown vault into `dir` and return the path.
/// Files are deliberately simple so BM25 hits are predictable.
pub fn write_fixture_vault(dir: &Path) {
    write(
        dir.join("alpha.md"),
        "# Alpha\n\nThe quick brown fox jumps over the lazy dog.\n",
    );
    write(
        dir.join("beta.md"),
        "# Beta\n\n## Coffee\n\nEspresso brewing requires consistent grind size.\n\n## Tea\n\nMatcha pairs with morning meditation.\n",
    );
    write(
        dir.join("gamma.md"),
        "# Gamma\n\nDistributed consensus protocols like Raft and Paxos.\n",
    );
}

fn write(path: PathBuf, contents: &str) {
    std::fs::write(&path, contents).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}
