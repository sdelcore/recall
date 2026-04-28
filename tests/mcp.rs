//! MCP (stdio JSON-RPC) integration tests.
//!
//! Spawns `recall serve --mode mcp`, drives the protocol over its stdin/stdout,
//! and asserts on response shape. No network, no LLM — pure handshake coverage.

mod common;

use common::RecallSandbox;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::time::{Duration, Instant};

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
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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
fn mcp_unknown_method_returns_error() {
    let sandbox = RecallSandbox::new();
    let mut client = McpClient::spawn(&sandbox);

    let resp = client.request("definitely/not/a/method", 1, json!({}));
    assert_eq!(resp["error"]["code"], -32601);
}
