//! MCP (stdio JSON-RPC) integration tests.
//!
//! Spawns `recall serve --mode mcp`, drives the protocol over its stdin/stdout,
//! and asserts on response shape. No network, no LLM — the sandbox has no
//! embeddings, so search falls back to BM25 and every query used here
//! classifies as `Lookup`, which never turns the reranker on.

mod common;

use common::{write_fixture_vault, RecallSandbox};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::time::{Duration, Instant};
use tempfile::{tempdir, TempDir};

struct McpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl McpClient {
    fn spawn(sandbox: &RecallSandbox) -> Self {
        let mut child = sandbox
            .raw_cmd()
            .args(["serve", "--mode", "mcp"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn recall serve");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn request(&mut self, method: &str, id: i64, params: Value) -> Value {
        let req = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let mut line = serde_json::to_string(&req).unwrap();
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).expect("write");
        self.stdin.flush().expect("flush");

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut buf = String::new();
        loop {
            buf.clear();
            self.stdout.read_line(&mut buf).expect("read_line");
            if buf.is_empty() {
                panic!("MCP server closed stdout before responding");
            }
            let v: Value = serde_json::from_str(buf.trim())
                .unwrap_or_else(|e| panic!("non-JSON line {buf:?}: {e}"));
            // Skip notifications; match by id.
            if v.get("id").and_then(|x| x.as_i64()) == Some(id) {
                return v;
            }
            if Instant::now() > deadline {
                panic!("timed out waiting for response to {method}");
            }
        }
    }

    fn call_tool(&mut self, id: i64, name: &str, arguments: Value) -> Value {
        self.request(
            "tools/call",
            id,
            json!({"name": name, "arguments": arguments}),
        )
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Register the fixture vault as a collection and index it. Returns the
/// tempdir so the caller keeps the vault alive for the test's duration.
fn indexed_sandbox(sandbox: &RecallSandbox) -> TempDir {
    let vault = tempdir().unwrap();
    write_fixture_vault(vault.path());
    // A filename-dated note, so the `before` / `after` bounds have something
    // unambiguous to bite on (the fixture files fall back to mtime = today).
    std::fs::write(
        vault.path().join("2020-01-02.md"),
        "# Old note\n\nA quokka sighting on Rottnest Island.\n",
    )
    .unwrap();

    add_and_index(sandbox, vault.path(), "test");
    vault
}

fn add_and_index(sandbox: &RecallSandbox, vault: &Path, name: &str) {
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

fn search_tool(client: &mut McpClient, id: i64) -> Value {
    let resp = client.request("tools/list", id, json!({}));
    resp["result"]["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .find(|t| t["name"] == "recall_search")
        .cloned()
        .expect("recall_search tool")
}

#[test]
fn mcp_initialize_handshake() {
    let sandbox = RecallSandbox::new();
    let mut client = McpClient::spawn(&sandbox);

    let resp = client.request("initialize", 1, json!({}));
    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["result"]["serverInfo"]["name"], "recall");
    assert!(resp["result"]["protocolVersion"].is_string());
}

#[test]
fn mcp_initialize_returns_instructions_with_the_retrieval_contract() {
    let sandbox = RecallSandbox::new();
    let mut client = McpClient::spawn(&sandbox);

    let resp = client.request("initialize", 1, json!({}));
    let instructions = resp["result"]["instructions"]
        .as_str()
        .expect("instructions string");

    assert!(
        instructions.contains("newer date wins"),
        "instructions must state the retrieval contract: {instructions}"
    );
    assert!(
        instructions.contains("verified live"),
        "instructions must warn about volatile facts: {instructions}"
    );
    // Empty sandbox: the gap it must announce is "nothing indexed".
    assert!(
        instructions.contains("index is empty"),
        "empty index must be announced: {instructions}"
    );
}

#[test]
fn mcp_initialize_instructions_announce_the_bm25_only_gap() {
    let sandbox = RecallSandbox::new();
    let _vault = indexed_sandbox(&sandbox);
    let mut client = McpClient::spawn(&sandbox);

    let resp = client.request("initialize", 1, json!({}));
    let instructions = resp["result"]["instructions"]
        .as_str()
        .expect("instructions string");

    assert!(
        instructions.contains("No vector embeddings; BM25-only"),
        "an unembedded index must announce its degraded mode: {instructions}"
    );
    assert!(
        instructions.contains("recall embed"),
        "the gap must come with its remedy: {instructions}"
    );
    assert!(
        instructions.contains("Collections (scope with the `collection` parameter): test"),
        "collection names must be listed: {instructions}"
    );
}

#[test]
fn mcp_tools_list_returns_three_tools() {
    let sandbox = RecallSandbox::new();
    let mut client = McpClient::spawn(&sandbox);

    let _ = client.request("initialize", 1, json!({}));
    let resp = client.request("tools/list", 2, json!({}));

    let tools = resp["result"]["tools"].as_array().expect("tools array");
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"recall_search"), "names={names:?}");
    assert!(names.contains(&"recall_index"), "names={names:?}");
    assert!(names.contains(&"recall_status"), "names={names:?}");
}

#[test]
fn mcp_search_schema_drops_hybrid_and_adds_before() {
    let sandbox = RecallSandbox::new();
    let mut client = McpClient::spawn(&sandbox);
    let _ = client.request("initialize", 1, json!({}));

    let tool = search_tool(&mut client, 2);
    let props = tool["inputSchema"]["properties"]
        .as_object()
        .expect("properties");

    for expected in ["query", "limit", "rerank", "after", "before", "collection"] {
        assert!(props.contains_key(expected), "missing param {expected}");
    }
    // Retrieval strategy is config, not a tool parameter.
    for banned in ["hybrid", "rerank_provider", "project", "file_pattern"] {
        assert!(!props.contains_key(banned), "param {banned} must not exist");
    }
    assert!(props["after"]["description"]
        .as_str()
        .unwrap()
        .contains("YYYY-MM-DD"));
    assert!(props["before"]["description"]
        .as_str()
        .unwrap()
        .contains("YYYY-MM-DD"));
}

#[test]
fn mcp_search_declares_annotations_and_output_schema() {
    let sandbox = RecallSandbox::new();
    let mut client = McpClient::spawn(&sandbox);
    let _ = client.request("initialize", 1, json!({}));

    let tool = search_tool(&mut client, 2);
    assert_eq!(tool["annotations"]["readOnlyHint"], true);
    assert_eq!(tool["annotations"]["openWorldHint"], false);

    let item_props = &tool["outputSchema"]["properties"]["results"]["items"]["properties"];
    for field in [
        "path",
        "line",
        "date",
        "date_source",
        "status",
        "memory_type",
        "collection",
        "score",
    ] {
        assert!(!item_props[field].is_null(), "outputSchema lacks {field}");
    }
    assert_eq!(item_props["line"]["type"], "integer");

    // The description is the tool's prompt surface; keep it substantial.
    let description = tool["description"].as_str().unwrap();
    assert!(
        description.len() > 800,
        "description is {} chars",
        description.len()
    );
    assert!(description.contains("Read(path, offset=line-20, limit=80)"));
}

#[test]
fn mcp_status_declares_annotations() {
    let sandbox = RecallSandbox::new();
    let mut client = McpClient::spawn(&sandbox);
    let _ = client.request("initialize", 1, json!({}));

    let resp = client.request("tools/list", 2, json!({}));
    let tool = resp["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "recall_status")
        .cloned()
        .unwrap();
    assert_eq!(tool["annotations"]["readOnlyHint"], true);
    assert_eq!(tool["annotations"]["openWorldHint"], false);
}

#[test]
fn mcp_search_emits_both_channels_with_the_full_payload() {
    let sandbox = RecallSandbox::new();
    let _vault = indexed_sandbox(&sandbox);
    let mut client = McpClient::spawn(&sandbox);
    let _ = client.request("initialize", 1, json!({}));

    let resp = client.call_tool(2, "recall_search", json!({"query": "Paxos"}));
    let result = &resp["result"];

    // Channel 1: text, for clients that forward only content[].text.
    let text = result["content"][0]["text"].as_str().expect("text channel");
    assert!(text.contains("gamma.md"), "text channel: {text}");
    assert!(text.contains("newer date wins"), "text channel: {text}");

    // Channel 2: structuredContent, for the Claude Code CLI.
    let results = result["structuredContent"]["results"]
        .as_array()
        .expect("structuredContent.results");
    assert_eq!(results.len(), 1, "expected one Paxos hit: {results:?}");
    let hit = &results[0];

    assert!(hit["path"].as_str().unwrap().ends_with("gamma.md"));
    // `line` must be a number the agent can compute with, not "12-40".
    assert!(hit["line"].is_i64(), "line must be an integer: {hit}");
    assert!(hit["line"].as_i64().unwrap() >= 1);
    assert_eq!(hit["date_source"], "mtime");
    assert_eq!(hit["collection"], "test");
    assert!(hit["score"].is_number());
    assert!(hit["content"].as_str().unwrap().contains("Paxos"));
    // Unknowns are explicit nulls, never omitted or synthesized.
    assert!(hit["status"].is_null(), "status should be null: {hit}");
}

#[test]
fn mcp_search_no_results_still_answers_on_both_channels() {
    let sandbox = RecallSandbox::new();
    let _vault = indexed_sandbox(&sandbox);
    let mut client = McpClient::spawn(&sandbox);
    let _ = client.request("initialize", 1, json!({}));

    let resp = client.call_tool(2, "recall_search", json!({"query": "zzzznotaword"}));
    assert!(resp["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("No results"));
    assert_eq!(resp["result"]["structuredContent"]["result_count"], 0);
}

#[test]
fn mcp_search_honors_before_and_after_bounds() {
    let sandbox = RecallSandbox::new();
    let _vault = indexed_sandbox(&sandbox);
    let mut client = McpClient::spawn(&sandbox);
    let _ = client.request("initialize", 1, json!({}));

    let unbounded = client.call_tool(2, "recall_search", json!({"query": "quokka"}));
    assert_eq!(unbounded["result"]["structuredContent"]["result_count"], 1);
    assert_eq!(
        unbounded["result"]["structuredContent"]["results"][0]["date"],
        "2020-01-02"
    );
    assert_eq!(
        unbounded["result"]["structuredContent"]["results"][0]["date_source"],
        "filename"
    );

    // `before` excludes it: the note is dated after the bound.
    let bounded = client.call_tool(
        3,
        "recall_search",
        json!({"query": "quokka", "before": "2019-12-31"}),
    );
    assert_eq!(bounded["result"]["structuredContent"]["result_count"], 0);

    // A window that contains it keeps it.
    let windowed = client.call_tool(
        4,
        "recall_search",
        json!({"query": "quokka", "after": "2020-01-01", "before": "2020-12-31"}),
    );
    assert_eq!(windowed["result"]["structuredContent"]["result_count"], 1);
}

#[test]
fn mcp_status_emits_both_channels() {
    let sandbox = RecallSandbox::new();
    let _vault = indexed_sandbox(&sandbox);
    let mut client = McpClient::spawn(&sandbox);
    let _ = client.request("initialize", 1, json!({}));

    let resp = client.call_tool(2, "recall_status", json!({}));
    let text = resp["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("recall index:"), "text channel: {text}");

    let structured = &resp["result"]["structuredContent"];
    assert_eq!(structured["vector_search_available"], false);
    assert!(structured["files"].as_i64().unwrap() >= 4);
    assert_eq!(structured["collections"][0]["name"], "test");
}

#[test]
fn mcp_unknown_method_returns_error() {
    let sandbox = RecallSandbox::new();
    let mut client = McpClient::spawn(&sandbox);

    let resp = client.request("definitely/not/a/method", 1, json!({}));
    assert_eq!(resp["error"]["code"], -32601);
}
