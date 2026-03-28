//! Adaptive bit-width quantization.
//!
//! Automatically selects quantization precision per-vector based on a
//! configurable classification function. High-importance vectors get
//! more bits (better accuracy), cold vectors get fewer bits (more compression).
//!
//! This extends `TieredQuantization` with dynamic tier selection and
//! promote/demote operations for changing a vector's precision level.
//!
//! # Example
//!
//! ```rust
//! use bitpolar::adaptive::AdaptiveQuantizer;
//! use bitpolar::tiered::Tier;
//!
//! let adaptive = AdaptiveQuantizer::builder(128, 42)
//!     .hot_bits(8)
//!     .warm_bits(4)
//!     .cold_bits(3)
//!     .build()
//!     .unwrap();
//!
//! let vector = vec![0.1_f32; 128];
//! let code = adaptive.encode_adaptive(&vector, Tier::Warm).unwrap();
//! ```

use crate::error::{Result, TurboQuantError};
use crate::tiered::Tier;
use crate::traits::{SerializableCode, VectorQuantizer};
use crate::TurboQuantizer;

/// Adaptive quantizer that supports per-vector bit-width selection
/// and tier promotion/demotion.
///
/// Wraps three `TurboQuantizer` instances (hot/warm/cold) and provides
/// methods to encode at a specific tier, promote to higher precision,
/// or demote to lower precision.
pub struct AdaptiveQuantizer {
    /// High-precision quantizer (e.g., 8-bit)
    hot: TurboQuantizer,
    /// Medium-precision quantizer (e.g., 4-bit)
    warm: TurboQuantizer,
    /// Low-precision quantizer (e.g., 3-bit)
    cold: TurboQuantizer,
    /// Vector dimension
    dim: usize,
}

/// Compressed code with tier metadata.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(serde::Serialize, serde::Deserialize))]
pub enum AdaptiveCode {
    /// Hot tier (highest precision)
    Hot(crate::turbo::TurboCode),
    /// Warm tier (medium precision)
    Warm(crate::turbo::TurboCode),
    /// Cold tier (lowest precision, maximum compression)
    Cold(crate::turbo::TurboCode),
}

impl AdaptiveCode {
    /// Return the tier of this code.
    pub fn tier(&self) -> Tier {
        match self {
            AdaptiveCode::Hot(_) => Tier::Hot,
            AdaptiveCode::Warm(_) => Tier::Warm,
            AdaptiveCode::Cold(_) => Tier::Cold,
        }
    }

    /// Return a reference to the inner TurboCode.
    pub fn inner(&self) -> &crate::turbo::TurboCode {
        match self {
            AdaptiveCode::Hot(c) | AdaptiveCode::Warm(c) | AdaptiveCode::Cold(c) => c,
        }
    }

    /// Return the compact byte representation.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![self.tier() as u8];
        bytes.extend(self.inner().to_compact_bytes());
        bytes
    }
}

/// Builder for `AdaptiveQuantizer`.
pub struct AdaptiveBuilder {
    dim: usize,
    seed: u64,
    hot_bits: u8,
    warm_bits: u8,
    cold_bits: u8,
    projections: Option<usize>,
}

impl AdaptiveQuantizer {
    /// Create a builder for an adaptive quantizer.
    ///
    /// # Arguments
    /// - `dim` — Vector dimension
    /// - `seed` — Random seed for deterministic quantization
    pub fn builder(dim: usize, seed: u64) -> AdaptiveBuilder {
        AdaptiveBuilder {
            dim,
            seed,
            hot_bits: 8,
            warm_bits: 4,
            cold_bits: 3,
            projections: None,
        }
    }

    /// Encode a vector at the specified tier.
    pub fn encode_adaptive(
        &self,
        vector: &[f32],
        tier: Tier,
    ) -> Result<AdaptiveCode> {
        match tier {
            Tier::Hot => Ok(AdaptiveCode::Hot(self.hot.encode(vector)?)),
            Tier::Warm => Ok(AdaptiveCode::Warm(self.warm.encode(vector)?)),
            Tier::Cold => Ok(AdaptiveCode::Cold(self.cold.encode(vector)?)),
        }
    }

    /// Decode an adaptive code back to an approximate vector.
    pub fn decode_adaptive(&self, code: &AdaptiveCode) -> Vec<f32> {
        match code {
            AdaptiveCode::Hot(c) => self.hot.decode(c),
            AdaptiveCode::Warm(c) => self.warm.decode(c),
            AdaptiveCode::Cold(c) => self.cold.decode(c),
        }
    }

    /// Estimate inner product between a code and a query vector.
    pub fn inner_product_estimate(
        &self,
        code: &AdaptiveCode,
        query: &[f32],
    ) -> Result<f32> {
        match code {
            AdaptiveCode::Hot(c) => self.hot.inner_product_estimate(c, query),
            AdaptiveCode::Warm(c) => self.warm.inner_product_estimate(c, query),
            AdaptiveCode::Cold(c) => self.cold.inner_product_estimate(c, query),
        }
    }

    /// Promote a code to a higher-precision tier.
    ///
    /// Decodes the current code and re-encodes at the next tier up.
    /// Hot codes cannot be promoted further.
    ///
    /// **Note**: Promotion operates on the approximate reconstruction, not the
    /// original vector. The result is lower quality than encoding the original
    /// vector directly at the target tier. Use when the original is unavailable.
    pub fn promote(&self, code: &AdaptiveCode) -> Result<AdaptiveCode> {
        let decoded = self.decode_adaptive(code);
        match code.tier() {
            Tier::Cold => self.encode_adaptive(&decoded, Tier::Warm),
            Tier::Warm => self.encode_adaptive(&decoded, Tier::Hot),
            Tier::Hot => Ok(code.clone()), // Already at highest tier
        }
    }

    /// Demote a code to a lower-precision tier.
    ///
    /// Decodes the current code and re-encodes at the next tier down.
    /// Cold codes cannot be demoted further.
    pub fn demote(&self, code: &AdaptiveCode) -> Result<AdaptiveCode> {
        let decoded = self.decode_adaptive(code);
        match code.tier() {
            Tier::Hot => self.encode_adaptive(&decoded, Tier::Warm),
            Tier::Warm => self.encode_adaptive(&decoded, Tier::Cold),
            Tier::Cold => Ok(code.clone()), // Already at lowest tier
        }
    }

    /// Return the vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl AdaptiveBuilder {
    /// Set hot tier bit-width (default: 8).
    pub fn hot_bits(mut self, bits: u8) -> Self {
        self.hot_bits = bits;
        self
    }

    /// Set warm tier bit-width (default: 4).
    pub fn warm_bits(mut self, bits: u8) -> Self {
        self.warm_bits = bits;
        self
    }

    /// Set cold tier bit-width (default: 3).
    pub fn cold_bits(mut self, bits: u8) -> Self {
        self.cold_bits = bits;
        self
    }

    /// Set QJL projections (default: dim/4).
    pub fn projections(mut self, proj: usize) -> Self {
        self.projections = Some(proj);
        self
    }

    /// Build the adaptive quantizer.
    ///
    /// # Errors
    /// Returns an error if bit-widths are not in descending order
    /// (hot >= warm >= cold) or if any bit-width is invalid.
    pub fn build(self) -> Result<AdaptiveQuantizer> {
        if !(self.hot_bits >= self.warm_bits && self.warm_bits >= self.cold_bits) {
            return Err(TurboQuantError::InvalidBitWidth(self.warm_bits));
        }
        let proj = self.projections.unwrap_or(self.dim / 4).max(1);
        Ok(AdaptiveQuantizer {
            hot: TurboQuantizer::new(self.dim, self.hot_bits, proj, self.seed)?,
            warm: TurboQuantizer::new(self.dim, self.warm_bits, proj, self.seed)?,
            cold: TurboQuantizer::new(self.dim, self.cold_bits, proj, self.seed)?,
            dim: self.dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_encode_decode() {
        let aq = AdaptiveQuantizer::builder(64, 42).build().unwrap();
        let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

        for tier in [Tier::Hot, Tier::Warm, Tier::Cold] {
            let code = aq.encode_adaptive(&v, tier).unwrap();
            assert_eq!(code.tier(), tier);
            let decoded = aq.decode_adaptive(&code);
            assert_eq!(decoded.len(), 64);
        }
    }

    #[test]
    fn test_promote_demote() {
        let aq = AdaptiveQuantizer::builder(64, 42).build().unwrap();
        let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();

        let cold = aq.encode_adaptive(&v, Tier::Cold).unwrap();
        assert_eq!(cold.tier(), Tier::Cold);

        let warm = aq.promote(&cold).unwrap();
        assert_eq!(warm.tier(), Tier::Warm);

        let hot = aq.promote(&warm).unwrap();
        assert_eq!(hot.tier(), Tier::Hot);

        // Promote hot → stays hot
        let still_hot = aq.promote(&hot).unwrap();
        assert_eq!(still_hot.tier(), Tier::Hot);

        // Demote hot → warm → cold
        let demoted = aq.demote(&hot).unwrap();
        assert_eq!(demoted.tier(), Tier::Warm);
        let demoted2 = aq.demote(&demoted).unwrap();
        assert_eq!(demoted2.tier(), Tier::Cold);
    }

    #[test]
    fn test_inner_product() {
        let aq = AdaptiveQuantizer::builder(64, 42).build().unwrap();
        let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let q: Vec<f32> = (0..64).map(|i| i as f32 * 0.02).collect();

        let code = aq.encode_adaptive(&v, Tier::Warm).unwrap();
        let score = aq.inner_product_estimate(&code, &q).unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_custom_bits() {
        let aq = AdaptiveQuantizer::builder(64, 42)
            .hot_bits(6)
            .warm_bits(4)
            .cold_bits(3)
            .build()
            .unwrap();
        let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let code = aq.encode_adaptive(&v, Tier::Hot).unwrap();
        assert_eq!(code.tier(), Tier::Hot);
    }
}
