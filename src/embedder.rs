//! In-process embeddings on the CPU.
//!
//! candle runs the model inside the process, in pure Rust, so hybrid search
//! works anywhere the binary runs — no daemon to reach, no native math
//! library, no ONNX runtime to version-match. That is the whole reason for
//! the choice: an embedding server made hybrid search a deployment question,
//! and hosts without one silently degraded to keyword-only. The cost is a
//! ~0.6s model load and roughly 9 chunks/sec on a CPU, which is why
//! [`Embedder::load`] is separate from [`Embedder::embed_batch`]: load once,
//! then feed it work.
//!
//! ONNX was tried first and rejected. `fastembed` pins `ort` at ABI 24 and
//! nixpkgs ships onnxruntime 1.22 (ABI 22); Cargo features are additive, so
//! the pin cannot be lowered and the binary dies at runtime with `BadVersion`.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::path::{Path, PathBuf};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::store::Store;

/// The embedding model, and therefore half of the index fingerprint. Changing
/// it makes every stored vector incomparable and cold-rebuilds the whole
/// index, so it is a code change with a consequence, not a setting.
pub const EMBEDDING_MODEL: &str = "bge-small-en-v1.5";

/// Width of a stored vector. The `vec_embeddings` table is declared with this,
/// and a mismatch against the loaded model is caught at load time rather than
/// as an opaque tensor error inside a forward pass.
pub const EMBEDDING_DIM: usize = 384;

/// Where the weights come from when nothing local is pinned.
const HF_REPO: &str = "BAAI/bge-small-en-v1.5";

/// Points at a directory of pinned weights. This is the seam the Nix build
/// uses: it stages `config.json`, `tokenizer.json` and `model.safetensors` in
/// the store and sets this, so the packaged binary never touches the network.
const MODEL_PATH_ENV: &str = "RECALL_MODEL_PATH";

/// BERT's positional embeddings stop here; longer input has to be truncated
/// rather than rejected, because a chunk's size cap is measured in characters.
const MAX_TOKENS: usize = 512;

/// BGE's specified retrieval instruction, prepended to queries only.
///
/// BGE is trained for *asymmetric* retrieval: the FlagEmbedding reference
/// implementation and the `BAAI/bge-small-en-v1.5` model card both prepend
/// this exact string to queries while leaving documents bare. Applying it to
/// documents, or skipping it on queries, is a real usage error — it does not
/// error out or degrade gracefully, it just quietly retrieves worse.
const QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";

const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const WEIGHTS_FILE: &str = "model.safetensors";

/// A loaded model. Construction is expensive; hold one and reuse it.
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    source: String,
}

impl Embedder {
    /// Read the weights and build the model. ~0.6s.
    pub fn load() -> Result<Self> {
        let files = ModelFiles::resolve()?;

        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&files.config)
                .with_context(|| format!("failed to read {}", files.config.display()))?,
        )
        .with_context(|| {
            format!(
                "{} is not a BERT config.json ({EMBEDDING_MODEL} expected)",
                files.config.display()
            )
        })?;
        if config.hidden_size != EMBEDDING_DIM {
            bail!(
                "{} declares hidden_size {}, but recall stores {EMBEDDING_DIM}-dimension vectors. \
                 {MODEL_PATH_ENV} must hold {HF_REPO}, not another model.",
                files.config.display(),
                config.hidden_size
            );
        }

        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow!("failed to load {}: {e}", files.tokenizer.display()))?;
        // Pad to the longest member of each batch, and mask the padding out of
        // the pooling below, so a chunk's vector does not depend on which
        // batch it happened to land in.
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("failed to configure truncation: {e}"))?;

        // Safety: mmap of a file we opened; the model outlives the borrow.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&files.weights),
                DType::F32,
                &Device::Cpu,
            )
        }
        .with_context(|| format!("failed to read weights from {}", files.weights.display()))?;
        let model = BertModel::load(vb, &config).with_context(|| {
            format!(
                "{} does not hold {EMBEDDING_MODEL} weights",
                files.weights.display()
            )
        })?;

        Ok(Self {
            model,
            tokenizer,
            source: files.source,
        })
    }

    /// Where the weights came from, for the line `recall embed` prints before
    /// it does any work.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Embed one *document*. A batch of one; prefer [`Embedder::embed_batch`]
    /// when there is more than one. Do not use this for a search query — see
    /// [`Embedder::embed_query`], which applies BGE's required query prefix.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed_batch(&[text])?.swap_remove(0))
    }

    /// Embed one *query*. BGE is trained asymmetrically: queries need
    /// [`QUERY_INSTRUCTION`] prepended, documents do not. This is the only
    /// entry point that adds it — indexing must go through [`Embedder::embed`]
    /// or [`Embedder::embed_batch`] instead, bare.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{QUERY_INSTRUCTION}{text}");
        self.embed(&prefixed)
    }

    /// Embed a batch of *documents* in a single forward pass. The batch is
    /// where the throughput is: per-text passes spend most of their time in
    /// setup. Do not feed queries through this — see [`Embedder::embed_query`].
    ///
    /// Returns L2-normalized vectors, so a dot product is a cosine similarity.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("tokenization failed: {e}"))?;

        let device = &Device::Cpu;
        let ids = encodings
            .iter()
            .map(|e| Tensor::new(e.get_ids(), device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let masks = encodings
            .iter()
            .map(|e| Tensor::new(e.get_attention_mask(), device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let ids = Tensor::stack(&ids, 0)?;
        let mask = Tensor::stack(&masks, 0)?;
        let token_types = ids.zeros_like()?;

        let hidden = self.model.forward(&ids, &token_types, Some(&mask))?;

        // CLS pooling, then L2-normalize. BGE is trained with the [CLS] token
        // (index 0) as the sentence representation, not a mean over tokens —
        // BAAI's model card and the FlagEmbedding reference implementation
        // both pool this way, and sentence-transformers ships the bge-* models
        // with a CLS pooling config. Mean pooling runs without error but is
        // not what the model was optimized for, and measurably degrades
        // retrieval; do not "fix" this back to a masked mean.
        let cls = hidden.i((.., 0))?;
        let norms = cls.sqr()?.sum_keepdim(1)?.sqrt()?;
        Ok(cls.broadcast_div(&norms)?.to_vec2::<f32>()?)
    }
}

/// The three files a load needs, and a description of where they came from.
#[derive(Debug)]
struct ModelFiles {
    config: PathBuf,
    tokenizer: PathBuf,
    weights: PathBuf,
    source: String,
}

impl ModelFiles {
    fn resolve() -> Result<Self> {
        match std::env::var(MODEL_PATH_ENV) {
            Ok(dir) if !dir.trim().is_empty() => Self::from_dir(Path::new(dir.trim())),
            _ => Self::from_hub(),
        }
    }

    /// Pinned weights on disk. Every failure names the file, the directory and
    /// the variable that chose it — a wrong path here would otherwise surface
    /// as a shape mismatch deep inside the model load.
    fn from_dir(dir: &Path) -> Result<Self> {
        if !dir.is_dir() {
            bail!(
                "{MODEL_PATH_ENV} is set to {}, which is not a directory. \
                 It must name a directory holding {CONFIG_FILE}, {TOKENIZER_FILE} and \
                 {WEIGHTS_FILE} for {HF_REPO}.",
                dir.display()
            );
        }
        let file = |name: &str| -> Result<PathBuf> {
            let path = dir.join(name);
            if !path.is_file() {
                bail!(
                    "{name} is missing from {} (set by {MODEL_PATH_ENV}). \
                     The directory must hold {CONFIG_FILE}, {TOKENIZER_FILE} and {WEIGHTS_FILE} \
                     for {HF_REPO}; unset {MODEL_PATH_ENV} to download them instead.",
                    dir.display()
                );
            }
            Ok(path)
        };
        Ok(Self {
            config: file(CONFIG_FILE)?,
            tokenizer: file(TOKENIZER_FILE)?,
            weights: file(WEIGHTS_FILE)?,
            source: dir.display().to_string(),
        })
    }

    /// Hugging Face cache, downloading on the first run.
    fn from_hub() -> Result<Self> {
        let repo = hf_hub::api::sync::Api::new()
            .context(HUB_HINT)?
            .model(HF_REPO.to_string());
        let file = |name: &str| -> Result<PathBuf> {
            repo.get(name)
                .with_context(|| format!("failed to fetch {name} from {HF_REPO}. {HUB_HINT}"))
        };
        Ok(Self {
            config: file(CONFIG_FILE)?,
            tokenizer: file(TOKENIZER_FILE)?,
            weights: file(WEIGHTS_FILE)?,
            source: format!("{HF_REPO} (Hugging Face cache)"),
        })
    }
}

const HUB_HINT: &str = "Set RECALL_MODEL_PATH to a directory holding config.json, tokenizer.json \
                        and model.safetensors to load the model offline.";

/// How many chunks go through the model in one forward pass. The batch is
/// where the throughput is; 32 keeps the padded activations small enough to
/// stay comfortable on a laptop.
const EMBED_BATCH: usize = 32;

/// What one [`embed_pending`] call did.
pub struct EmbedProgress {
    /// Chunks the call took on. Equal to the budget when the budget ran out
    /// before the backlog did — which is how a bounded caller knows there is
    /// more work waiting.
    pub attempted: usize,
    /// Chunks whose vector is now stored.
    pub embedded: usize,
    /// Chunks the model or the database refused. They keep no vector, so the
    /// next call picks them up again.
    pub failed: usize,
}

/// The model's batch entry point, as a plain function. [`embed_pending`] takes
/// it this way so the backlog loop can be exercised without weights.
pub type BatchEmbed<'a> = dyn Fn(&[&str]) -> Result<Vec<Vec<f32>>> + 'a;

/// Embed the chunks the indexer left without a vector, lowest id first.
///
/// One loop, two callers: `recall embed` passes no budget and drains the
/// index, the watcher passes a small one so a large backlog cannot hold its
/// event loop for minutes. `on_batch` receives `(done, total)` before each
/// forward pass, for a progress line.
///
/// `embed` is the model's batch entry point; see [`BatchEmbed`]. CI never
/// downloads weights, so the loop is tested through a stand-in.
///
/// A batch the model rejects is counted and skipped rather than fatal: the
/// chunks keep no vector, so the next run retries them, and one bad batch does
/// not throw away the fifteen minutes of work behind it.
pub fn embed_pending(
    store: &Store,
    embed: &BatchEmbed<'_>,
    budget: Option<usize>,
    mut on_batch: impl FnMut(usize, usize),
) -> Result<EmbedProgress> {
    let chunks = store.get_chunks_without_embeddings(budget)?;
    let total = chunks.len();
    let mut progress = EmbedProgress {
        attempted: total,
        embedded: 0,
        failed: 0,
    };

    for (index, batch) in chunks.chunks(EMBED_BATCH).enumerate() {
        let first = index * EMBED_BATCH + 1;
        let last = first + batch.len() - 1;
        on_batch(last, total);

        let texts: Vec<&str> = batch.iter().map(|(_, content)| content.as_str()).collect();
        let embeddings = match embed(&texts) {
            Ok(embeddings) => embeddings,
            Err(e) => {
                eprintln!("\nFailed to embed chunks {first}..{last}: {e}");
                progress.failed += batch.len();
                continue;
            }
        };
        // A short batch would zip away the tail chunks: not embedded, not
        // counted, and back in the next call's backlog — which reads as
        // progress to a caller that is looping on it.
        if embeddings.len() != texts.len() {
            eprintln!(
                "\nEmbedding chunks {first}..{last} returned {} vectors for {} chunks",
                embeddings.len(),
                texts.len()
            );
            progress.failed += batch.len();
            continue;
        }

        for ((chunk_id, _), embedding) in batch.iter().zip(embeddings) {
            match store.store_embedding(*chunk_id, &embedding) {
                Ok(()) => progress.embedded += 1,
                Err(e) => {
                    eprintln!("\nFailed to store embedding for chunk {chunk_id}: {e}");
                    progress.failed += 1;
                }
            }
        }
    }

    Ok(progress)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RECALL_MODEL_PATH` is process-global, so the tests that read it take
    /// this lock rather than racing each other's environment.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn resolve_with(value: &str) -> Result<ModelFiles> {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous = std::env::var(MODEL_PATH_ENV).ok();
        std::env::set_var(MODEL_PATH_ENV, value);
        let result = ModelFiles::resolve();
        match previous {
            Some(previous) => std::env::set_var(MODEL_PATH_ENV, previous),
            None => std::env::remove_var(MODEL_PATH_ENV),
        }
        result
    }

    fn write_model_dir(dir: &Path, files: &[&str]) {
        for name in files {
            std::fs::write(dir.join(name), "{}").unwrap();
        }
    }

    /// A store holding `sections` chunks and no vectors. One heading per
    /// chunk, so the count is exact rather than a function of the size cap.
    fn store_with_chunks(dir: &Path, sections: usize) -> Store {
        let vault = dir.join("vault");
        std::fs::create_dir_all(&vault).unwrap();
        let note: String = (0..sections)
            .map(|i| format!("# Section {i}\n\nquokka zebra {i}\n\n"))
            .collect();
        std::fs::write(vault.join("note.md"), note).unwrap();

        let store = Store::open_at(dir.join("memory.sqlite")).unwrap();
        let root = vault.to_string_lossy().to_string();
        let collection = store.create_collection("t", &root).unwrap();
        store.index(collection.id, &root).unwrap();
        assert_eq!(
            store.get_embedding_stats().unwrap(),
            (0, sections as i64),
            "fixture should be one chunk per section, none embedded"
        );
        store
    }

    /// Stands in for the model. The vectors are constant — these tests are
    /// about the backlog loop, not about what it stores.
    fn fake_embed(texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.1f32; EMBEDDING_DIM]).collect())
    }

    /// The watcher embeds in slices so a large backlog cannot hold its event
    /// loop. That only works if each slice takes the *next* chunks.
    #[test]
    fn a_budget_bounds_one_call_and_the_next_call_resumes() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_chunks(tmp.path(), 40);

        let first = embed_pending(&store, &fake_embed, Some(10), |_, _| {}).unwrap();
        assert_eq!((first.attempted, first.embedded, first.failed), (10, 10, 0));

        let second = embed_pending(&store, &fake_embed, Some(10), |_, _| {}).unwrap();
        assert_eq!(
            (second.attempted, second.embedded, second.failed),
            (10, 10, 0)
        );

        assert_eq!(store.get_embedding_stats().unwrap(), (20, 40));
    }

    /// `attempted` is how a bounded caller learns there is more to do. It has
    /// to fall short of the budget exactly when the backlog is exhausted.
    #[test]
    fn attempted_falls_short_of_the_budget_only_on_the_last_call() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_chunks(tmp.path(), 15);

        assert_eq!(
            embed_pending(&store, &fake_embed, Some(10), |_, _| {})
                .unwrap()
                .attempted,
            10
        );
        assert_eq!(
            embed_pending(&store, &fake_embed, Some(10), |_, _| {})
                .unwrap()
                .attempted,
            5
        );
    }

    /// One rejected batch must not throw away the work behind it, and the
    /// chunks it skipped must stay pending for the next run.
    #[test]
    fn a_rejected_batch_is_counted_and_the_run_continues() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_chunks(tmp.path(), 40);

        let calls = std::cell::Cell::new(0);
        let flaky = |texts: &[&str]| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                bail!("out of memory")
            }
            fake_embed(texts)
        };

        let progress = embed_pending(&store, &flaky, None, |_, _| {}).unwrap();
        assert_eq!(calls.get(), 2, "both batches attempted");
        assert_eq!(
            (progress.attempted, progress.embedded, progress.failed),
            (40, 8, EMBED_BATCH)
        );

        // The failed batch kept no vector, so a retry picks it up.
        assert_eq!(
            embed_pending(&store, &fake_embed, None, |_, _| {})
                .unwrap()
                .embedded,
            EMBED_BATCH
        );
    }

    /// A short batch used to zip away: the tail chunks kept no vector but were
    /// counted as attempted, which a bounded caller reads as progress.
    #[test]
    fn a_short_batch_counts_as_failed_rather_than_silently_dropping_chunks() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_chunks(tmp.path(), 4);

        let short = |texts: &[&str]| Ok(fake_embed(texts)?.into_iter().skip(1).collect());
        let progress = embed_pending(&store, &short, None, |_, _| {}).unwrap();

        assert_eq!(
            (progress.attempted, progress.embedded, progress.failed),
            (4, 0, 4)
        );
        assert_eq!(store.get_embedding_stats().unwrap(), (0, 4));
    }

    /// The watcher sweeps on a timer forever. A sweep with nothing to do must
    /// not reach the model at all.
    #[test]
    fn nothing_pending_never_touches_the_model() {
        let tmp = tempfile::tempdir().unwrap();
        let store = store_with_chunks(tmp.path(), 4);
        embed_pending(&store, &fake_embed, None, |_, _| {}).unwrap();

        let progress = embed_pending(
            &store,
            &|_| panic!("the model must not be called with an empty backlog"),
            None,
            |_, _| panic!("no batch to report"),
        )
        .unwrap();
        assert_eq!(progress.attempted, 0);
    }

    #[test]
    fn pinned_directory_wins_over_the_hub() {
        let tmp = tempfile::tempdir().unwrap();
        write_model_dir(tmp.path(), &[CONFIG_FILE, TOKENIZER_FILE, WEIGHTS_FILE]);

        let files = resolve_with(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(files.config, tmp.path().join(CONFIG_FILE));
        assert_eq!(files.tokenizer, tmp.path().join(TOKENIZER_FILE));
        assert_eq!(files.weights, tmp.path().join(WEIGHTS_FILE));
        assert_eq!(files.source, tmp.path().display().to_string());
    }

    #[test]
    fn a_missing_file_names_the_file_the_directory_and_the_variable() {
        let tmp = tempfile::tempdir().unwrap();
        write_model_dir(tmp.path(), &[CONFIG_FILE, TOKENIZER_FILE]);

        let err = resolve_with(tmp.path().to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains(WEIGHTS_FILE), "{err}");
        assert!(err.contains(&tmp.path().display().to_string()), "{err}");
        assert!(err.contains(MODEL_PATH_ENV), "{err}");
    }

    #[test]
    fn a_path_that_is_not_a_directory_says_so() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join(CONFIG_FILE);
        std::fs::write(&file, "{}").unwrap();

        let err = resolve_with(file.to_str().unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a directory"), "{err}");
        assert!(err.contains(MODEL_PATH_ENV), "{err}");
    }

    /// Weights-gated tests only run where `RECALL_MODEL_PATH` already points
    /// at a pinned model directory, so the suite stays hermetic and offline
    /// by default. Returns `None` (and prints why) when they should skip.
    fn pinned_model_dir() -> Option<String> {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(dir) = std::env::var(MODEL_PATH_ENV) else {
            eprintln!("skipping: {MODEL_PATH_ENV} is not set");
            return None;
        };
        assert!(Path::new(&dir).is_dir());
        Some(dir)
    }

    /// Needs the real weights, so it only runs where they are already pinned.
    #[test]
    fn embeds_a_batch_to_unit_vectors() {
        let Some(_dir) = pinned_model_dir() else {
            return;
        };

        let embedder = Embedder::load().unwrap();
        let vectors = embedder
            .embed_batch(&[
                "the cat sat on the mat",
                "a kitten rested on the rug",
                "quarterly revenue guidance for the fiscal year",
            ])
            .unwrap();

        assert_eq!(vectors.len(), 3);
        for vector in &vectors {
            assert_eq!(vector.len(), EMBEDDING_DIM);
            let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "not normalized: {norm}");
        }

        // The vectors are unit length, so a dot product is a cosine. This is
        // the assertion that catches a wrong pooling or a mis-shaped mask: the
        // shapes stay valid, only the meaning goes.
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let related = dot(&vectors[0], &vectors[1]);
        let unrelated = dot(&vectors[0], &vectors[2]);
        assert!(
            related > unrelated,
            "related {related}, unrelated {unrelated}"
        );
    }

    /// `embed_query` must apply BGE's query instruction prefix and `embed`
    /// must not — mixing the two up is exactly the bug this module exists to
    /// prevent, so the same text through both paths has to land on different
    /// vectors.
    #[test]
    fn embed_query_prepends_the_instruction_and_embed_does_not() {
        let Some(_dir) = pinned_model_dir() else {
            return;
        };

        let embedder = Embedder::load().unwrap();
        let text = "what is the capital of France";

        let as_document = embedder.embed(text).unwrap();
        let as_query = embedder.embed_query(text).unwrap();
        let prefixed_by_hand = embedder
            .embed(&format!("{QUERY_INSTRUCTION}{text}"))
            .unwrap();

        assert_eq!(as_document.len(), EMBEDDING_DIM);
        assert_eq!(as_query.len(), EMBEDDING_DIM);
        assert_ne!(
            as_document, as_query,
            "embed_query must differ from embed for the same text"
        );
        assert_eq!(
            as_query, prefixed_by_hand,
            "embed_query must be exactly embed(QUERY_INSTRUCTION + text)"
        );
    }
}
