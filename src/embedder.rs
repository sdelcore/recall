use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::Config;

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
    /// Create a new embedder with default settings
    pub fn new() -> Self {
        Self::with_url_and_model("http://localhost:11434", "nomic-embed-text")
    }

    /// Create embedder from Config
    pub fn new_with_config(config: &Config) -> Self {
        Self::with_url_and_model(&config.embeddings.ollama_url, &config.embeddings.model)
    }

    /// Create embedder with custom URL and model
    pub fn with_url_and_model(base_url: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
            model: model.to_string(),
        }
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

    /// Generate embeddings for multiple texts (batched)
    #[allow(dead_code)]
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = self.embed(text).await?;
            embeddings.push(embedding);
        }

        Ok(embeddings)
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

    /// Get embedding dimensions for the configured model
    #[allow(dead_code)]
    pub fn dimensions(&self) -> usize {
        // nomic-embed-text produces 768-dimensional embeddings
        768
    }
}

