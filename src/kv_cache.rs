//! KV cache compressor for transformer attention.
//!
//! Provides [`KvCacheCompressor`] for single-head KV cache compression
//! and [`MultiHeadKvCache`] for multi-head transformer attention.
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
#[cfg_attr(feature = "serde-support", derive(serde::Serialize, serde::Deserialize))]
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

// ---------------------------------------------------------------------------
// KvCacheCompressor
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
    ///
    /// # Errors
    /// - `ZeroDimension`, `OddDimension`, `InvalidBitWidth`, `ZeroProjections`
    pub fn new(config: &KvCacheConfig) -> Result<Self> {
        let quantizer =
            TurboQuantizer::new(config.head_dim, config.bits, config.projections, config.seed)?;
        Ok(Self { quantizer, keys: Vec::new(), values: Vec::new() })
    }

    /// Push a (key, value) pair into the cache.
    ///
    /// Both vectors must have length `head_dim`.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
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
    ///
    /// `query` must have length `head_dim`.
    ///
    /// Returns one unnormalized (pre-softmax) score per stored key.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
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
            .map(|kc| self.quantizer.inner_product_estimate(kc, query).map(|ip| ip * scale))
            .collect();
        scores
    }

    /// Decode all stored values back to approximate f32 vectors.
    pub fn decode_values(&self) -> Vec<Vec<f32>> {
        self.values.iter().map(|vc| self.quantizer.decode(vc)).collect()
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

    /// Clear all stored keys and values.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
}

// ---------------------------------------------------------------------------
// MultiHeadKvCache
// ---------------------------------------------------------------------------

/// Multi-head KV cache for transformer attention.
///
/// Maintains one [`KvCacheCompressor`] per attention head.
#[derive(Debug, Clone)]
pub struct MultiHeadKvCache {
    heads: Vec<KvCacheCompressor>,
}

impl MultiHeadKvCache {
    /// Create a new multi-head KV cache.
    ///
    /// # Arguments
    /// - `num_heads` — number of attention heads
    /// - `config` — shared configuration applied to all heads
    ///
    /// # Errors
    /// - Any error from [`KvCacheCompressor::new`]
    pub fn new(num_heads: usize, config: &KvCacheConfig) -> Result<Self> {
        if num_heads == 0 {
            return Err(TurboQuantError::EmptyInput("num_heads must be > 0"));
        }
        let heads: Result<Vec<_>> = (0..num_heads)
            .map(|h| {
                // Use a different seed per head to avoid correlated projections.
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
    ///
    /// `keys` and `values` must each have exactly `num_heads` slices of length `head_dim`.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
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

    /// Compute attention scores for each head given per-head query vectors.
    ///
    /// `queries` must have exactly `num_heads` slices of length `head_dim`.
    ///
    /// Returns one score vector per head.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
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

    /// Number of tokens stored (same for all heads).
    pub fn len(&self) -> usize {
        self.heads.first().map_or(0, |h| h.len())
    }

    /// Returns `true` if no tokens have been stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all heads.
    pub fn clear(&mut self) {
        for h in &mut self.heads {
            h.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> KvCacheConfig {
        KvCacheConfig { head_dim: 8, bits: 4, projections: 16, seed: 42 }
    }

    #[test]
    fn test_push_and_len() {
        let mut cache = KvCacheCompressor::new(&sample_config()).unwrap();
        assert!(cache.is_empty());
        let k = vec![0.1_f32; 8];
        let v = vec![0.2_f32; 8];
        cache.push(&k, &v).unwrap();
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_attention_scores_shape() {
        let mut cache = KvCacheCompressor::new(&sample_config()).unwrap();
        for _ in 0..5 {
            let k = vec![0.1_f32; 8];
            let v = vec![0.2_f32; 8];
            cache.push(&k, &v).unwrap();
        }
        let q = vec![0.1_f32; 8];
        let scores = cache.attention_scores(&q).unwrap();
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn test_decode_values_shape() {
        let mut cache = KvCacheCompressor::new(&sample_config()).unwrap();
        let k = vec![0.1_f32; 8];
        let v = vec![0.2_f32; 8];
        cache.push(&k, &v).unwrap();
        let decoded = cache.decode_values();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].len(), 8);
    }

    #[test]
    fn test_compression_ratio_positive() {
        let mut cache = KvCacheCompressor::new(&sample_config()).unwrap();
        let k = vec![0.5_f32; 8];
        let v = vec![0.3_f32; 8];
        cache.push(&k, &v).unwrap();
        assert!(cache.compression_ratio() > 0.0);
    }

    #[test]
    fn test_clear() {
        let mut cache = KvCacheCompressor::new(&sample_config()).unwrap();
        let k = vec![0.1_f32; 8];
        let v = vec![0.2_f32; 8];
        cache.push(&k, &v).unwrap();
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_multi_head_push_and_len() {
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
    fn test_multi_head_attention_scores_shape() {
        let config = sample_config();
        let mut mhc = MultiHeadKvCache::new(2, &config).unwrap();
        let k_vecs: Vec<Vec<f32>> = (0..2).map(|_| vec![0.1_f32; 8]).collect();
        let v_vecs: Vec<Vec<f32>> = (0..2).map(|_| vec![0.2_f32; 8]).collect();
        let keys: Vec<&[f32]> = k_vecs.iter().map(|v| v.as_slice()).collect();
        let vals: Vec<&[f32]> = v_vecs.iter().map(|v| v.as_slice()).collect();
        mhc.push_token(&keys, &vals).unwrap();

        let q_vecs: Vec<Vec<f32>> = (0..2).map(|_| vec![0.1_f32; 8]).collect();
        let queries: Vec<&[f32]> = q_vecs.iter().map(|v| v.as_slice()).collect();
        let scores = mhc.attention_scores(&queries).unwrap();
        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].len(), 1);
    }
}
