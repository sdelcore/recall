use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default)]
    pub embeddings: EmbeddingsConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub watch: WatchConfig,
    #[serde(default)]
    pub reranking: RerankConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankConfig {
    /// Enable LLM reranking
    #[serde(default)]
    pub enabled: bool,
    /// Provider: "claude-code" (default), "anthropic", "ollama"
    #[serde(default = "default_rerank_provider")]
    pub provider: String,
    /// How many RRF candidates to send to the reranker
    #[serde(default = "default_rerank_candidates")]
    pub candidates: usize,
    /// How many results to return after reranking
    #[serde(default = "default_rerank_top_k")]
    pub top_k: usize,
    /// Claude Code SDK settings
    #[serde(default)]
    pub claude_code: Option<ClaudeCodeRerankConfig>,
    /// Anthropic API settings
    #[serde(default)]
    pub anthropic: Option<AnthropicRerankConfig>,
    /// Ollama settings
    #[serde(default)]
    pub ollama: Option<OllamaRerankConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodeRerankConfig {
    /// Model to use (default: "haiku")
    #[serde(default = "default_haiku_model")]
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicRerankConfig {
    /// Model name
    pub model: Option<String>,
    /// Env var containing API key (default: ANTHROPIC_API_KEY)
    pub api_key_env: Option<String>,
    /// Max concurrent API calls
    pub max_concurrent: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaRerankConfig {
    /// Ollama URL
    pub url: Option<String>,
    /// Model name
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Paths to index
    #[serde(default = "default_index_paths")]
    pub paths: Vec<String>,
    /// Glob patterns to exclude
    #[serde(default = "default_exclude_patterns")]
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsConfig {
    /// Ollama server URL
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
    /// Embedding model name
    #[serde(default = "default_embedding_model")]
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Default number of results
    #[serde(default = "default_limit")]
    pub default_limit: usize,
    /// RRF constant k (higher = more weight to lower-ranked results)
    #[serde(default = "default_rrf_k")]
    pub rrf_k: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchConfig {
    /// Paths to watch for changes
    #[serde(default = "default_watch_paths")]
    pub paths: Vec<String>,
    /// Patterns to exclude from watching
    #[serde(default = "default_watch_exclude")]
    pub exclude: Vec<String>,
    /// Debounce time in milliseconds
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,
}

// Default value functions
fn default_index_paths() -> Vec<String> {
    vec![expand_home("~/Obsidian")]
}

fn default_exclude_patterns() -> Vec<String> {
    vec![
        "**/Templates/**".to_string(),
        "**/.obsidian/**".to_string(),
        "**/attachments/**".to_string(),
    ]
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

fn default_embedding_model() -> String {
    "nomic-embed-text".to_string()
}

fn default_limit() -> usize {
    5
}

fn default_rrf_k() -> u32 {
    60
}

fn default_watch_paths() -> Vec<String> {
    vec![expand_home("~/Obsidian")]
}

fn default_watch_exclude() -> Vec<String> {
    vec![
        "Templates/".to_string(),
        ".obsidian/".to_string(),
        "attachments/".to_string(),
        ".sync-conflict-".to_string(),
    ]
}

fn default_debounce_ms() -> u64 {
    1500
}

fn default_rerank_provider() -> String {
    "claude-code".to_string()
}

fn default_rerank_candidates() -> usize {
    20
}

fn default_rerank_top_k() -> usize {
    5
}

fn default_haiku_model() -> String {
    "haiku".to_string()
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: default_rerank_provider(),
            candidates: default_rerank_candidates(),
            top_k: default_rerank_top_k(),
            claude_code: None,
            anthropic: None,
            ollama: None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            index: IndexConfig::default(),
            embeddings: EmbeddingsConfig::default(),
            search: SearchConfig::default(),
            watch: WatchConfig::default(),
            reranking: RerankConfig::default(),
        }
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            paths: default_index_paths(),
            exclude: default_exclude_patterns(),
        }
    }
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            ollama_url: default_ollama_url(),
            model: default_embedding_model(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: default_limit(),
            rrf_k: default_rrf_k(),
        }
    }
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            paths: default_watch_paths(),
            exclude: default_watch_exclude(),
            debounce_ms: default_debounce_ms(),
        }
    }
}

/// Expand ~ to home directory
pub fn expand_home(path: &str) -> String {
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(&path[2..]).to_string_lossy().to_string();
        }
    }
    path.to_string()
}

impl Config {
    /// Load config from file, or return defaults if not found
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();

        if config_path.exists() {
            let contents = fs::read_to_string(&config_path)?;
            let config: Config = toml::from_str(&contents)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    /// Get the config file path
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("recall")
            .join("config.toml")
    }

    /// Get expanded watch paths
    pub fn watch_paths(&self) -> Vec<String> {
        self.watch.paths.iter().map(|p| expand_home(p)).collect()
    }

    /// Get expanded index paths
    pub fn index_paths(&self) -> Vec<String> {
        self.index.paths.iter().map(|p| expand_home(p)).collect()
    }

    /// Check if a path should be excluded from watching
    pub fn should_skip_watch(&self, path: &str) -> bool {
        self.watch.exclude.iter().any(|pattern| path.contains(pattern))
    }
}
