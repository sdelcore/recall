use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// The embedding model, and therefore half of the index fingerprint. Changing
/// it makes every stored vector incomparable and cold-rebuilds the whole
/// index, so it is a code change with a consequence, not a setting.
pub const EMBEDDING_MODEL: &str = "nomic-embed-text";

/// Where Ollama listens. The one value with a plausible second setting — a
/// remote GPU box — so it stays overridable, by environment variable rather
/// than by a config file that would exist only for this.
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Ollama embedding client
pub struct Embedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

impl Embedder {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: ollama_url(),
            model: EMBEDDING_MODEL.to_string(),
        }
    }

    /// The URL this client talks to, for the connectivity message `recall
    /// embed` prints before it does any work.
    pub fn url(&self) -> &str {
        &self.base_url
    }

    /// Generate embedding for a single text
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);

        let request = EmbeddingRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send embedding request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Embedding request failed: {} - {}", status, body);
        }

        let result: EmbeddingResponse = response
            .json()
            .await
            .context("Failed to parse embedding response")?;

        Ok(result.embedding)
    }

    /// Check if Ollama is available
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);

        match self.client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Ensure the configured model is available, pulling it if needed
    pub async fn ensure_model(&self) -> Result<()> {
        let url = format!("{}/api/show", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&serde_json::json!({"name": self.model}))
            .send()
            .await
            .context("Failed to check model availability")?;

        if response.status().is_success() {
            return Ok(());
        }

        // Model not found — pull it
        eprintln!("Model '{}' not found locally, pulling...", self.model);
        let pull_url = format!("{}/api/pull", self.base_url);
        let pull_response = self
            .client
            .post(&pull_url)
            .json(&serde_json::json!({"name": self.model}))
            .send()
            .await
            .context("Failed to start model pull")?;

        if !pull_response.status().is_success() {
            let body = pull_response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to pull model '{}': {}", self.model, body);
        }

        // The pull endpoint streams progress as NDJSON — read until complete
        let body = pull_response.text().await.unwrap_or_default();
        for line in body.lines() {
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(status) = obj.get("status").and_then(|s| s.as_str()) {
                    eprintln!("  {}", status);
                }
            }
        }

        eprintln!("Model '{}' pulled successfully.", self.model);
        Ok(())
    }
}

/// `RECALL_OLLAMA_URL`, or localhost.
fn ollama_url() -> String {
    std::env::var("RECALL_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string())
}
