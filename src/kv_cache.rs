//! KV cache compressor for modern transformer attention architectures.
//!
//! Supports all major 2024-2026 attention variants:
//!
//! - **MHA** (Multi-Head Attention) — legacy, 1 KV head per query head
//! - **GQA** (Grouped Query Attention) — Llama 3, GPT-5, Gemma 3, Mistral
//! - **MQA** (Multi-Query Attention) — Falcon, PaLM
//! - **MLA** (Multi-head Latent Attention) — DeepSeek V3/R1
//! - **Sliding Window** — Gemma 3, Mistral (local window + global layers)
//!
//! Each key/value vector is compressed on arrival using TurboQuant,
//! enabling efficient attention over long sequences with bounded memory.

use crate::error::{validate_finite, Result, TurboQuantError};
use crate::turbo::{TurboCode, TurboQuantizer};

// ---------------------------------------------------------------------------
// KvCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for a KV cache compressor.
///
/// Fully determines the quantizer: `(head_dim, bits, projections, seed)`.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct KvCacheConfig {
    /// Dimension of each attention head's key/value vector
    pub head_dim: usize,
    /// Polar angle quantization bit width (1..=16)
    pub bits: u8,
    /// Number of QJL residual projections
    pub projections: usize,
    /// RNG seed
    pub seed: u64,
}

/// Attention architecture variant.
///
/// Determines how KV heads map to query heads and how the cache is structured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum AttentionType {
    /// Multi-Head Attention: `num_kv_heads == num_query_heads`.
    /// Each query head has its own KV head. (GPT-2, older models)
    MHA,
    /// Grouped Query Attention: `num_kv_heads < num_query_heads`.
    /// Multiple query heads share one KV head.
    /// (Llama 3, GPT-5, Gemma 3, Mistral, Claude)
    GQA {
        /// Number of query heads (e.g. 32)
        num_query_heads: usize,
    },
    /// Multi-Query Attention: `num_kv_heads == 1`.
    /// All query heads share a single KV head. (Falcon, PaLM)
    MQA {
        /// Number of query heads
        num_query_heads: usize,
    },
    /// Multi-head Latent Attention: KV compressed to latent space.
    /// (DeepSeek V3/R1)
    MLA {
        /// Latent dimension (d_c, e.g. 512). Much smaller than num_heads × head_dim.
        latent_dim: usize,
    },
}

/// Layer-level attention configuration for hybrid architectures.
///
/// Gemma 3 uses 5 local sliding-window layers per 1 global layer.
/// Kimi uses KDA (linear) + MLA (full attention) interleaving.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum LayerAttentionKind {
    /// Full attention — stores KV for all tokens.
    Global,
    /// Sliding window — only stores KV for the last `window_size` tokens.
    SlidingWindow {
        window_size: usize,
    },
    /// Linear attention — stores fixed-size recurrent state, not per-token KV.
    /// Not compressed by BitPolar (passed through as-is).
    Linear {
        state_dim: usize,
    },
}

// ---------------------------------------------------------------------------
// KvCacheCompressor (single-head, unchanged core)
// ---------------------------------------------------------------------------

/// Single-head KV cache compressor.
///
/// Stores compressed key and value vectors for one attention head.
/// Vectors are compressed on arrival using [`TurboQuantizer`].
#[derive(Debug, Clone)]
pub struct KvCacheCompressor {
    quantizer: TurboQuantizer,
    keys: Vec<TurboCode>,
    values: Vec<TurboCode>,
}

impl KvCacheCompressor {
    /// Create a new KV cache compressor from a [`KvCacheConfig`].
    pub fn new(config: &KvCacheConfig) -> Result<Self> {
        let quantizer = TurboQuantizer::new(
            config.head_dim,
            config.bits,
            config.projections,
            config.seed,
        )?;
        Ok(Self {
            quantizer,
            keys: Vec::new(),
            values: Vec::new(),
        })
    }

    /// Push a (key, value) pair into the cache.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::kv_cache::push",
            skip(self, key, value),
            fields(cache_len = self.keys.len(), dim = self.quantizer.dim())
        )
    )]
    pub fn push(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        validate_finite(key)?;
        validate_finite(value)?;
        let kc = self.quantizer.encode(key)?;
        let vc = self.quantizer.encode(value)?;
        self.keys.push(kc);
        self.values.push(vc);
        Ok(())
    }

    /// Number of (key, value) pairs stored.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Compute approximate attention scores `softmax(q · K^T / √d)`.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::kv_cache::attention_scores",
            skip(self, query),
            fields(cache_len = self.keys.len(), dim = self.quantizer.dim())
        )
    )]
    pub fn attention_scores(&self, query: &[f32]) -> Result<Vec<f32>> {
        validate_finite(query)?;
        let scale = 1.0 / crate::compat::math::sqrtf(self.quantizer.dim() as f32);
        let scores: Result<Vec<f32>> = self
            .keys
            .iter()
            .map(|kc| {
                self.quantizer
                    .inner_product_estimate(kc, query)
                    .map(|ip| ip * scale)
            })
            .collect();
        scores
    }

    /// Decode all stored values back to approximate f32 vectors.
    pub fn decode_values(&self) -> Vec<Vec<f32>> {
        self.values
            .iter()
            .map(|vc| self.quantizer.decode(vc))
            .collect()
    }

    /// Decode all stored keys back to approximate f32 vectors.
    pub fn decode_keys(&self) -> Vec<Vec<f32>> {
        self.keys
            .iter()
            .map(|kc| self.quantizer.decode(kc))
            .collect()
    }

    /// Access raw compressed key codes (for serialization/transfer).
    pub fn key_codes(&self) -> &[TurboCode] {
        &self.keys
    }

    /// Access raw compressed value codes (for serialization/transfer).
    pub fn value_codes(&self) -> &[TurboCode] {
        &self.values
    }

    /// Approximate compression ratio: original bytes / compressed bytes.
    pub fn compression_ratio(&self) -> f64 {
        let total_codes: usize = self.keys.len() + self.values.len();
        if total_codes == 0 {
            return 1.0;
        }
        let original = total_codes * self.quantizer.dim() * core::mem::size_of::<f32>();
        let compressed: usize = self
            .keys
            .iter()
            .chain(self.values.iter())
            .map(|c| c.size_bytes())
            .sum();
        if compressed == 0 {
            return 1.0;
        }
        original as f64 / compressed as f64
    }

    /// Evict tokens outside the sliding window, keeping only the last `window_size`.
    pub fn evict_outside_window(&mut self, window_size: usize) {
        if self.keys.len() > window_size {
            let start = self.keys.len() - window_size;
            self.keys = self.keys.split_off(start);
            self.values = self.values.split_off(0); // values already correct length
            // Fix: both must be trimmed identically
            let vstart = if self.values.len() > window_size {
                self.values.len() - window_size
            } else {
                0
            };
            if vstart > 0 {
                self.values = self.values.split_off(vstart);
            }
        }
    }

    /// Clear all stored keys and values.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
}

// ---------------------------------------------------------------------------
// MultiHeadKvCache (legacy MHA — backward compatible)
// ---------------------------------------------------------------------------

/// Multi-head KV cache for MHA (Multi-Head Attention).
///
/// Maintains one [`KvCacheCompressor`] per attention head.
/// For modern architectures (GQA, MLA), use [`GqaKvCache`] or [`MlaKvCache`].
#[derive(Debug, Clone)]
pub struct MultiHeadKvCache {
    heads: Vec<KvCacheCompressor>,
}

impl MultiHeadKvCache {
    /// Create a new MHA cache with one compressor per head.
    pub fn new(num_heads: usize, config: &KvCacheConfig) -> Result<Self> {
        if num_heads == 0 {
            return Err(TurboQuantError::EmptyInput("num_heads must be > 0"));
        }
        let heads: Result<Vec<_>> = (0..num_heads)
            .map(|h| {
                let head_config = KvCacheConfig {
                    seed: config.seed.wrapping_add(h as u64),
                    ..*config
                };
                KvCacheCompressor::new(&head_config)
            })
            .collect();
        Ok(Self { heads: heads? })
    }

    /// Push a token's KV vectors for all heads simultaneously.
    pub fn push_token(&mut self, keys: &[&[f32]], values: &[&[f32]]) -> Result<()> {
        if keys.len() != self.heads.len() || values.len() != self.heads.len() {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.heads.len(),
                actual: keys.len(),
            });
        }
        for (h, head) in self.heads.iter_mut().enumerate() {
            head.push(keys[h], values[h])?;
        }
        Ok(())
    }

    /// Compute attention scores for each head.
    pub fn attention_scores(&self, queries: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if queries.len() != self.heads.len() {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.heads.len(),
                actual: queries.len(),
            });
        }
        self.heads
            .iter()
            .zip(queries.iter())
            .map(|(head, q)| head.attention_scores(q))
            .collect()
    }

    /// Number of tokens stored.
    pub fn len(&self) -> usize {
        self.heads.first().map_or(0, |h| h.len())
    }

    /// Returns `true` if no tokens have been stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of KV heads.
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Access a specific head's compressor.
    pub fn head(&self, index: usize) -> Option<&KvCacheCompressor> {
        self.heads.get(index)
    }

    /// Clear all heads.
    pub fn clear(&mut self) {
        for h in &mut self.heads {
            h.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// GqaKvCache — Grouped Query Attention (Llama 3, GPT-5, Gemma, Mistral)
// ---------------------------------------------------------------------------

/// GQA KV cache: fewer KV heads shared by multiple query heads.
///
/// This is the **dominant architecture in 2025-2026**:
/// - Llama 3 (70B): 8 KV heads, 64 query heads (group_size = 8)
/// - GPT-5: GQA with unspecified ratio
/// - Gemma 3: GQA + sliding window interleaving
/// - Mistral: GQA + sliding window
///
/// Each KV head serves `group_size = num_query_heads / num_kv_heads` query heads.
/// The compressed KV cache has `num_kv_heads` compressors, not `num_query_heads`.
#[derive(Debug, Clone)]
pub struct GqaKvCache {
    /// One compressor per KV head (NOT per query head).
    kv_heads: Vec<KvCacheCompressor>,
    /// Number of query heads that share each KV head.
    group_size: usize,
    /// Total number of query heads.
    num_query_heads: usize,
}

impl GqaKvCache {
    /// Create a GQA cache.
    ///
    /// # Arguments
    /// - `num_kv_heads` — number of key/value heads (e.g. 8 for Llama 3 70B)
    /// - `num_query_heads` — number of query heads (e.g. 64 for Llama 3 70B)
    /// - `config` — quantizer configuration (head_dim, bits, etc.)
    ///
    /// `num_query_heads` must be divisible by `num_kv_heads`.
    pub fn new(
        num_kv_heads: usize,
        num_query_heads: usize,
        config: &KvCacheConfig,
    ) -> Result<Self> {
        if num_kv_heads == 0 {
            return Err(TurboQuantError::EmptyInput("num_kv_heads must be > 0"));
        }
        if num_query_heads == 0 {
            return Err(TurboQuantError::EmptyInput("num_query_heads must be > 0"));
        }
        if num_query_heads % num_kv_heads != 0 {
            return Err(TurboQuantError::DimensionMismatch {
                expected: num_kv_heads,
                actual: num_query_heads % num_kv_heads,
            });
        }

        let group_size = num_query_heads / num_kv_heads;
        let kv_heads: Result<Vec<_>> = (0..num_kv_heads)
            .map(|h| {
                let head_config = KvCacheConfig {
                    seed: config.seed.wrapping_add(h as u64),
                    ..*config
                };
                KvCacheCompressor::new(&head_config)
            })
            .collect();

        Ok(Self {
            kv_heads: kv_heads?,
            group_size,
            num_query_heads,
        })
    }

    /// Shorthand for common configurations.
    pub fn llama3_70b(config: &KvCacheConfig) -> Result<Self> {
        Self::new(8, 64, config)
    }

    /// Shorthand for Llama 3 8B (GQA: 8 KV heads, 32 query heads).
    pub fn llama3_8b(config: &KvCacheConfig) -> Result<Self> {
        Self::new(8, 32, config)
    }

    /// Push a token's KV vectors. `keys` and `values` must have `num_kv_heads` entries.
    pub fn push_token(&mut self, keys: &[&[f32]], values: &[&[f32]]) -> Result<()> {
        if keys.len() != self.kv_heads.len() || values.len() != self.kv_heads.len() {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.kv_heads.len(),
                actual: keys.len(),
            });
        }
        for (h, head) in self.kv_heads.iter_mut().enumerate() {
            head.push(keys[h], values[h])?;
        }
        Ok(())
    }

    /// Compute attention scores for ALL query heads.
    ///
    /// `queries` must have `num_query_heads` entries. Each group of `group_size`
    /// query heads shares the same KV head's keys.
    ///
    /// Returns one score vector per query head.
    pub fn attention_scores(&self, queries: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if queries.len() != self.num_query_heads {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.num_query_heads,
                actual: queries.len(),
            });
        }
        let mut all_scores = Vec::with_capacity(self.num_query_heads);
        for (q_idx, query) in queries.iter().enumerate() {
            let kv_idx = q_idx / self.group_size;
            let scores = self.kv_heads[kv_idx].attention_scores(query)?;
            all_scores.push(scores);
        }
        Ok(all_scores)
    }

    /// Number of tokens stored.
    pub fn len(&self) -> usize {
        self.kv_heads.first().map_or(0, |h| h.len())
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn num_kv_heads(&self) -> usize { self.kv_heads.len() }
    pub fn num_query_heads(&self) -> usize { self.num_query_heads }
    pub fn group_size(&self) -> usize { self.group_size }

    /// Access a specific KV head's compressor (for serialization/transfer).
    pub fn kv_head(&self, index: usize) -> Option<&KvCacheCompressor> {
        self.kv_heads.get(index)
    }

    /// Compression ratio across all KV heads.
    pub fn compression_ratio(&self) -> f64 {
        if self.kv_heads.is_empty() { return 1.0; }
        let total: f64 = self.kv_heads.iter().map(|h| h.compression_ratio()).sum();
        total / self.kv_heads.len() as f64
    }

    pub fn clear(&mut self) {
        for h in &mut self.kv_heads { h.clear(); }
    }
}

// ---------------------------------------------------------------------------
// MlaKvCache — Multi-head Latent Attention (DeepSeek V3/R1)
// ---------------------------------------------------------------------------

/// MLA KV cache: compresses latent vectors (not per-head KV).
///
/// DeepSeek V3 compresses each token's KV into a latent vector of dimension `d_c`
/// (e.g. 512 for DeepSeek V3, vs 128×128 = 16384 for full MHA).
/// The KV cache stores one latent per token, not per-head vectors.
///
/// BitPolar compresses these latent vectors further for transfer/storage.
#[derive(Debug, Clone)]
pub struct MlaKvCache {
    /// Single compressor for latent vectors.
    compressor: KvCacheCompressor,
    /// Original latent dimension (d_c).
    latent_dim: usize,
}

impl MlaKvCache {
    /// Create an MLA cache.
    ///
    /// # Arguments
    /// - `latent_dim` — latent KV dimension (d_c), e.g. 512 for DeepSeek V3
    /// - `bits` — quantization bits (1..=16)
    /// - `projections` — QJL residual projections
    /// - `seed` — RNG seed
    pub fn new(latent_dim: usize, bits: u8, projections: usize, seed: u64) -> Result<Self> {
        let config = KvCacheConfig {
            head_dim: latent_dim,
            bits,
            projections,
            seed,
        };
        let compressor = KvCacheCompressor::new(&config)?;
        Ok(Self {
            compressor,
            latent_dim,
        })
    }

    /// Shorthand for DeepSeek V3 (d_c = 512).
    pub fn deepseek_v3(bits: u8, projections: usize, seed: u64) -> Result<Self> {
        Self::new(512, bits, projections, seed)
    }

    /// Push a token's latent KV vector.
    ///
    /// In MLA, the same latent serves as both key and value source.
    /// We store it once (as both key and value in the compressor).
    pub fn push_latent(&mut self, latent: &[f32]) -> Result<()> {
        // MLA stores a single latent per token. We store it as "key" only
        // (the up-projection to K and V happens at decode time, not at storage time).
        validate_finite(latent)?;
        let code = self.compressor.quantizer_ref().encode(latent)?;
        self.compressor.keys.push(code);
        Ok(())
    }

    /// Decode all stored latent vectors.
    pub fn decode_latents(&self) -> Vec<Vec<f32>> {
        self.compressor
            .keys
            .iter()
            .map(|c| self.compressor.quantizer_ref().decode(c))
            .collect()
    }

    /// Access raw compressed latent codes (for serialization/transfer).
    pub fn latent_codes(&self) -> &[TurboCode] {
        &self.compressor.keys
    }

    /// Number of tokens stored.
    pub fn len(&self) -> usize { self.compressor.keys.len() }
    pub fn is_empty(&self) -> bool { self.compressor.keys.is_empty() }
    pub fn latent_dim(&self) -> usize { self.latent_dim }

    pub fn compression_ratio(&self) -> f64 {
        if self.compressor.keys.is_empty() { return 1.0; }
        let original = self.compressor.keys.len() * self.latent_dim * core::mem::size_of::<f32>();
        let compressed: usize = self.compressor.keys.iter().map(|c| c.size_bytes()).sum();
        if compressed == 0 { return 1.0; }
        original as f64 / compressed as f64
    }

    pub fn clear(&mut self) { self.compressor.keys.clear(); }
}

// ---------------------------------------------------------------------------
// SlidingWindowKvCache — For Gemma 3, Mistral local attention layers
// ---------------------------------------------------------------------------

/// Sliding window KV cache: only keeps the last `window_size` tokens.
///
/// Gemma 3 uses 5 local layers (window=1024) per 1 global layer.
/// Mistral uses sliding window attention on all layers.
///
/// Wraps a [`GqaKvCache`] and automatically evicts old tokens.
#[derive(Debug, Clone)]
pub struct SlidingWindowKvCache {
    inner: GqaKvCache,
    window_size: usize,
    total_tokens_seen: usize,
}

impl SlidingWindowKvCache {
    /// Create a sliding window cache.
    pub fn new(
        num_kv_heads: usize,
        num_query_heads: usize,
        window_size: usize,
        config: &KvCacheConfig,
    ) -> Result<Self> {
        if window_size == 0 {
            return Err(TurboQuantError::EmptyInput("window_size must be > 0"));
        }
        let inner = GqaKvCache::new(num_kv_heads, num_query_heads, config)?;
        Ok(Self {
            inner,
            window_size,
            total_tokens_seen: 0,
        })
    }

    /// Push a token. Automatically evicts tokens outside the window.
    pub fn push_token(&mut self, keys: &[&[f32]], values: &[&[f32]]) -> Result<()> {
        self.inner.push_token(keys, values)?;
        self.total_tokens_seen += 1;

        // Evict if cache exceeds window
        if self.inner.len() > self.window_size {
            for head in &mut self.inner.kv_heads {
                head.evict_outside_window(self.window_size);
            }
        }
        Ok(())
    }

    /// Attention scores (only over tokens in the window).
    pub fn attention_scores(&self, queries: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        self.inner.attention_scores(queries)
    }

    /// Tokens currently in the window.
    pub fn window_len(&self) -> usize { self.inner.len() }
    /// Total tokens ever pushed (including evicted).
    pub fn total_tokens_seen(&self) -> usize { self.total_tokens_seen }
    pub fn window_size(&self) -> usize { self.window_size }
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }

    pub fn clear(&mut self) {
        self.inner.clear();
        self.total_tokens_seen = 0;
    }
}

// ---------------------------------------------------------------------------
// LayerKvCache — Per-layer cache for hybrid architectures
// ---------------------------------------------------------------------------

/// Per-layer KV cache that supports different attention types per layer.
///
/// Gemma 3: layers 0-4 = SlidingWindow(1024), layer 5 = Global, repeat.
/// Kimi: 3 KDA (linear) layers per 1 MLA (full) layer.
///
/// This is the top-level cache struct for modern model serving.
#[derive(Debug, Clone)]
pub enum LayerCache {
    /// Full attention layer — stores all tokens.
    Global(GqaKvCache),
    /// Sliding window layer — evicts old tokens.
    Window(SlidingWindowKvCache),
    /// MLA layer — stores latent vectors.
    Latent(MlaKvCache),
    /// Linear attention layer — not compressed by BitPolar.
    /// Stores raw recurrent state as-is.
    Linear {
        state: Vec<f32>,
        state_dim: usize,
    },
}

/// Full model KV cache: one [`LayerCache`] per transformer layer.
#[derive(Debug, Clone)]
pub struct ModelKvCache {
    layers: Vec<LayerCache>,
}

impl ModelKvCache {
    /// Create a model cache from a per-layer configuration.
    pub fn new(layer_configs: Vec<LayerCache>) -> Self {
        Self {
            layers: layer_configs,
        }
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Access a specific layer's cache.
    pub fn layer(&self, index: usize) -> Option<&LayerCache> {
        self.layers.get(index)
    }

    /// Mutable access to a specific layer's cache.
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut LayerCache> {
        self.layers.get_mut(index)
    }

    /// Total compressed size across all layers (bytes).
    pub fn total_compressed_bytes(&self) -> usize {
        self.layers.iter().map(|layer| match layer {
            LayerCache::Global(gqa) => {
                gqa.kv_heads.iter()
                    .flat_map(|h| h.key_codes().iter().chain(h.value_codes().iter()))
                    .map(|c| c.size_bytes())
                    .sum()
            }
            LayerCache::Window(sw) => {
                sw.inner.kv_heads.iter()
                    .flat_map(|h| h.key_codes().iter().chain(h.value_codes().iter()))
                    .map(|c| c.size_bytes())
                    .sum()
            }
            LayerCache::Latent(mla) => {
                mla.latent_codes().iter().map(|c| c.size_bytes()).sum()
            }
            LayerCache::Linear { state, .. } => {
                state.len() * core::mem::size_of::<f32>()
            }
        }).sum()
    }

    /// Clear all layers.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            match layer {
                LayerCache::Global(gqa) => gqa.clear(),
                LayerCache::Window(sw) => sw.clear(),
                LayerCache::Latent(mla) => mla.clear(),
                LayerCache::Linear { state, .. } => state.clear(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Builder helpers for common model configurations
// ---------------------------------------------------------------------------

/// Create a [`ModelKvCache`] for Llama 3 70B (80 layers, all GQA).
///
/// 8 KV heads, 64 query heads, head_dim=128.
pub fn llama3_70b_cache(bits: u8, projections: usize, seed: u64) -> Result<ModelKvCache> {
    let config = KvCacheConfig { head_dim: 128, bits, projections, seed };
    let layers: Result<Vec<_>> = (0..80)
        .map(|l| {
            let layer_config = KvCacheConfig { seed: seed.wrapping_add(l * 100), ..config.clone() };
            Ok(LayerCache::Global(GqaKvCache::new(8, 64, &layer_config)?))
        })
        .collect();
    Ok(ModelKvCache::new(layers?))
}

/// Create a [`ModelKvCache`] for Llama 3 8B (32 layers, all GQA).
///
/// 8 KV heads, 32 query heads, head_dim=128.
pub fn llama3_8b_cache(bits: u8, projections: usize, seed: u64) -> Result<ModelKvCache> {
    let config = KvCacheConfig { head_dim: 128, bits, projections, seed };
    let layers: Result<Vec<_>> = (0..32)
        .map(|l| {
            let layer_config = KvCacheConfig { seed: seed.wrapping_add(l * 100), ..config.clone() };
            Ok(LayerCache::Global(GqaKvCache::new(8, 32, &layer_config)?))
        })
        .collect();
    Ok(ModelKvCache::new(layers?))
}

/// Create a [`ModelKvCache`] for Gemma 3 (interleaved sliding window + global).
///
/// Pattern: 5 local (window=1024) + 1 global, repeated.
pub fn gemma3_cache(
    num_layers: usize,
    num_kv_heads: usize,
    num_query_heads: usize,
    head_dim: usize,
    bits: u8,
    projections: usize,
    seed: u64,
) -> Result<ModelKvCache> {
    let config = KvCacheConfig { head_dim, bits, projections, seed };
    let layers: Result<Vec<_>> = (0..num_layers)
        .map(|l| {
            let layer_config = KvCacheConfig { seed: seed.wrapping_add(l as u64 * 100), ..config.clone() };
            if (l + 1) % 6 == 0 {
                // Every 6th layer is global
                Ok(LayerCache::Global(GqaKvCache::new(num_kv_heads, num_query_heads, &layer_config)?))
            } else {
                // Other layers are sliding window
                Ok(LayerCache::Window(SlidingWindowKvCache::new(
                    num_kv_heads, num_query_heads, 1024, &layer_config,
                )?))
            }
        })
        .collect();
    Ok(ModelKvCache::new(layers?))
}

/// Create a [`ModelKvCache`] for DeepSeek V3 (all MLA layers).
///
/// d_c = 512 latent dimension.
pub fn deepseek_v3_cache(
    num_layers: usize,
    bits: u8,
    projections: usize,
    seed: u64,
) -> Result<ModelKvCache> {
    let layers: Result<Vec<_>> = (0..num_layers)
        .map(|l| {
            Ok(LayerCache::Latent(MlaKvCache::new(
                512, bits, projections, seed.wrapping_add(l as u64 * 100),
            )?))
        })
        .collect();
    Ok(ModelKvCache::new(layers?))
}

// ---------------------------------------------------------------------------
// Helper: expose quantizer reference for MLA
// ---------------------------------------------------------------------------

impl KvCacheCompressor {
    /// Access the underlying quantizer (for MLA direct encoding).
    pub(crate) fn quantizer_ref(&self) -> &TurboQuantizer {
        &self.quantizer
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> KvCacheConfig {
        KvCacheConfig {
            head_dim: 8,
            bits: 4,
            projections: 16,
            seed: 42,
        }
    }

    // --- Single-head compressor tests ---

    #[test]
    fn test_push_and_len() {
        let mut cache = KvCacheCompressor::new(&sample_config()).unwrap();
        assert!(cache.is_empty());
        cache.push(&vec![0.1_f32; 8], &vec![0.2_f32; 8]).unwrap();
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_attention_scores_shape() {
        let mut cache = KvCacheCompressor::new(&sample_config()).unwrap();
        for _ in 0..5 {
            cache.push(&vec![0.1_f32; 8], &vec![0.2_f32; 8]).unwrap();
        }
        let scores = cache.attention_scores(&vec![0.1_f32; 8]).unwrap();
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn test_decode_values_shape() {
        let mut cache = KvCacheCompressor::new(&sample_config()).unwrap();
        cache.push(&vec![0.1; 8], &vec![0.2; 8]).unwrap();
        assert_eq!(cache.decode_values().len(), 1);
        assert_eq!(cache.decode_values()[0].len(), 8);
    }

    #[test]
    fn test_compression_ratio_positive() {
        let mut cache = KvCacheCompressor::new(&sample_config()).unwrap();
        cache.push(&vec![0.5; 8], &vec![0.3; 8]).unwrap();
        assert!(cache.compression_ratio() > 0.0);
    }

    #[test]
    fn test_clear() {
        let mut cache = KvCacheCompressor::new(&sample_config()).unwrap();
        cache.push(&vec![0.1; 8], &vec![0.2; 8]).unwrap();
        cache.clear();
        assert!(cache.is_empty());
    }

    // --- MHA tests ---

    #[test]
    fn test_mha_push_and_len() {
        let config = sample_config();
        let mut mhc = MultiHeadKvCache::new(4, &config).unwrap();
        let k_vecs: Vec<Vec<f32>> = (0..4).map(|_| vec![0.1_f32; 8]).collect();
        let v_vecs: Vec<Vec<f32>> = (0..4).map(|_| vec![0.2_f32; 8]).collect();
        let keys: Vec<&[f32]> = k_vecs.iter().map(|v| v.as_slice()).collect();
        let vals: Vec<&[f32]> = v_vecs.iter().map(|v| v.as_slice()).collect();
        mhc.push_token(&keys, &vals).unwrap();
        assert_eq!(mhc.len(), 1);
    }

    #[test]
    fn test_mha_attention_scores_shape() {
        let config = sample_config();
        let mut mhc = MultiHeadKvCache::new(2, &config).unwrap();
        let k_vecs: Vec<Vec<f32>> = (0..2).map(|_| vec![0.1_f32; 8]).collect();
        let v_vecs: Vec<Vec<f32>> = (0..2).map(|_| vec![0.2_f32; 8]).collect();
        mhc.push_token(
            &k_vecs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            &v_vecs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
        ).unwrap();
        let q_vecs: Vec<Vec<f32>> = (0..2).map(|_| vec![0.1_f32; 8]).collect();
        let scores = mhc.attention_scores(&q_vecs.iter().map(|v| v.as_slice()).collect::<Vec<_>>()).unwrap();
        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].len(), 1);
    }

    // --- GQA tests ---

    #[test]
    fn test_gqa_creation() {
        let config = sample_config();
        let gqa = GqaKvCache::new(2, 8, &config).unwrap();
        assert_eq!(gqa.num_kv_heads(), 2);
        assert_eq!(gqa.num_query_heads(), 8);
        assert_eq!(gqa.group_size(), 4);
    }

    #[test]
    fn test_gqa_rejects_non_divisible() {
        let config = sample_config();
        let result = GqaKvCache::new(3, 8, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_gqa_push_and_attention() {
        let config = sample_config();
        let mut gqa = GqaKvCache::new(2, 8, &config).unwrap();

        // Push 1 token: 2 KV head vectors
        let k_vecs: Vec<Vec<f32>> = (0..2).map(|_| vec![0.1_f32; 8]).collect();
        let v_vecs: Vec<Vec<f32>> = (0..2).map(|_| vec![0.2_f32; 8]).collect();
        gqa.push_token(
            &k_vecs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            &v_vecs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
        ).unwrap();
        assert_eq!(gqa.len(), 1);

        // 8 query heads, each gets scores from their shared KV head
        let q_vecs: Vec<Vec<f32>> = (0..8).map(|_| vec![0.1_f32; 8]).collect();
        let scores = gqa.attention_scores(
            &q_vecs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
        ).unwrap();
        assert_eq!(scores.len(), 8); // one per query head
        assert_eq!(scores[0].len(), 1); // 1 token stored
        // Query heads 0-3 share KV head 0, so their scores should be identical
        assert_eq!(scores[0], scores[1]);
        assert_eq!(scores[0], scores[3]);
        // Query heads 4-7 share KV head 1
        assert_eq!(scores[4], scores[5]);
    }

    #[test]
    fn test_gqa_llama3_shorthand() {
        let config = KvCacheConfig { head_dim: 128, bits: 4, projections: 32, seed: 42 };
        let gqa = GqaKvCache::llama3_8b(&config).unwrap();
        assert_eq!(gqa.num_kv_heads(), 8);
        assert_eq!(gqa.num_query_heads(), 32);
        assert_eq!(gqa.group_size(), 4);
    }

    // --- MLA tests ---

    #[test]
    fn test_mla_creation() {
        let mla = MlaKvCache::new(512, 4, 32, 42).unwrap();
        assert_eq!(mla.latent_dim(), 512);
        assert!(mla.is_empty());
    }

    #[test]
    fn test_mla_push_and_decode() {
        let mut mla = MlaKvCache::new(8, 4, 16, 42).unwrap();
        mla.push_latent(&vec![0.1_f32; 8]).unwrap();
        mla.push_latent(&vec![0.2_f32; 8]).unwrap();
        assert_eq!(mla.len(), 2);
        let decoded = mla.decode_latents();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].len(), 8);
    }

    #[test]
    fn test_mla_deepseek_shorthand() {
        let mla = MlaKvCache::deepseek_v3(4, 32, 42).unwrap();
        assert_eq!(mla.latent_dim(), 512);
    }

    // --- Sliding window tests ---

    #[test]
    fn test_sliding_window_eviction() {
        let config = sample_config();
        let mut sw = SlidingWindowKvCache::new(1, 1, 3, &config).unwrap();

        for i in 0..5 {
            let k = vec![(i as f32) * 0.1 + 0.1; 8];
            let v = vec![(i as f32) * 0.1 + 0.2; 8];
            sw.push_token(&[k.as_slice()], &[v.as_slice()]).unwrap();
        }

        assert_eq!(sw.total_tokens_seen(), 5);
        assert_eq!(sw.window_len(), 3); // only last 3 kept
    }

    // --- Model cache tests ---

    #[test]
    fn test_llama3_8b_cache_creation() {
        let cache = llama3_8b_cache(4, 16, 42).unwrap();
        assert_eq!(cache.num_layers(), 32);
    }

    #[test]
    fn test_deepseek_v3_cache_creation() {
        let cache = deepseek_v3_cache(61, 4, 32, 42).unwrap();
        assert_eq!(cache.num_layers(), 61);
    }

    #[test]
    fn test_gemma3_cache_interleaving() {
        let cache = gemma3_cache(12, 4, 16, 8, 4, 16, 42).unwrap();
        assert_eq!(cache.num_layers(), 12);
        // Layers 0-4 should be Window, layer 5 should be Global
        assert!(matches!(cache.layer(0).unwrap(), LayerCache::Window(_)));
        assert!(matches!(cache.layer(4).unwrap(), LayerCache::Window(_)));
        assert!(matches!(cache.layer(5).unwrap(), LayerCache::Global(_)));
        assert!(matches!(cache.layer(6).unwrap(), LayerCache::Window(_)));
        assert!(matches!(cache.layer(11).unwrap(), LayerCache::Global(_)));
    }
}
