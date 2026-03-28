//! Tiered quantization: hot, warm, and cold storage tiers.
//!
//! Different use cases require different accuracy/memory trade-offs.
//! `TieredQuantization` wraps three [`TurboQuantizer`] instances — one per
//! tier — so callers can select the right fidelity at encode time and
//! transparently work with codes from any tier.
//!
//! | Tier | Bits | Use case |
//! |------|------|----------|
//! | Hot  | 8    | Frequently accessed vectors, highest accuracy |
//! | Warm | 4    | Infrequently accessed vectors, balanced |
//! | Cold | 3    | Rarely accessed vectors, maximum compression |
//!
//! # Example
//!
//! ```rust
//! use bitpolar::tiered::{TieredQuantization, Tier};
//!
//! let tq = TieredQuantization::new(64, 32, 42).unwrap();
//! let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
//!
//! let hot_code = tq.encode(&v, Tier::Hot).unwrap();
//! let cold_code = tq.encode(&v, Tier::Cold).unwrap();
//!
//! // All tiers have the same byte footprint; the bit width only affects accuracy.
//! assert_eq!(tq.code_size_bytes(&hot_code), tq.code_size_bytes(&cold_code));
//!
//! // All tiers support inner product estimation.
//! let query: Vec<f32> = vec![1.0_f32; 64];
//! let _score = tq.inner_product_estimate(&hot_code, &query).unwrap();
//! ```

use crate::error::Result;
use crate::turbo::{TurboCode, TurboQuantizer};
use crate::traits::VectorQuantizer;

// ---------------------------------------------------------------------------
// Tier
// ---------------------------------------------------------------------------

/// Storage tier selector for [`TieredQuantization`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde-support", derive(serde::Serialize, serde::Deserialize))]
pub enum Tier {
    /// 8-bit quantization — highest accuracy, largest codes.
    Hot,
    /// 4-bit quantization — balanced accuracy and size.
    Warm,
    /// 3-bit quantization — maximum compression, lowest accuracy.
    Cold,
}

// ---------------------------------------------------------------------------
// TieredCode
// ---------------------------------------------------------------------------

/// A compressed vector code tagged with its storage [`Tier`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(serde::Serialize, serde::Deserialize))]
pub enum TieredCode {
    /// Hot-tier code (8-bit TurboCode).
    Hot(TurboCode),
    /// Warm-tier code (4-bit TurboCode).
    Warm(TurboCode),
    /// Cold-tier code (3-bit TurboCode).
    Cold(TurboCode),
}

impl TieredCode {
    /// Return the [`Tier`] of this code.
    pub fn tier(&self) -> Tier {
        match self {
            TieredCode::Hot(_) => Tier::Hot,
            TieredCode::Warm(_) => Tier::Warm,
            TieredCode::Cold(_) => Tier::Cold,
        }
    }
}

// ---------------------------------------------------------------------------
// TieredQuantization
// ---------------------------------------------------------------------------

/// Three-tier vector quantizer.
///
/// Internally holds one [`TurboQuantizer`] per tier (8-bit hot, 4-bit warm,
/// 3-bit cold). All three share the same dimension, projection count, and
/// seed, so they apply identical rotations and residual projection matrices —
/// the only difference is angle quantization bit width.
///
/// # Errors
///
/// Construction fails with [`crate::error::TurboQuantError`] if the dimension
/// is zero or odd, or if `projections` is zero.
#[derive(Debug, Clone)]
pub struct TieredQuantization {
    hot: TurboQuantizer,
    warm: TurboQuantizer,
    cold: TurboQuantizer,
}

impl TieredQuantization {
    /// Create a new `TieredQuantization`.
    ///
    /// # Arguments
    /// - `dim` — vector dimension (must be even and > 0)
    /// - `projections` — number of QJL residual projections (must be > 0)
    /// - `seed` — RNG seed shared across all three tiers
    ///
    /// # Errors
    /// - `ZeroDimension`, `OddDimension`, `ZeroProjections`
    pub fn new(dim: usize, projections: usize, seed: u64) -> Result<Self> {
        let hot = TurboQuantizer::new(dim, 8, projections, seed)?;
        let warm = TurboQuantizer::new(dim, 4, projections, seed)?;
        let cold = TurboQuantizer::new(dim, 3, projections, seed)?;
        Ok(Self { hot, warm, cold })
    }

    /// The vector dimension.
    pub fn dim(&self) -> usize {
        self.hot.dim()
    }

    // -----------------------------------------------------------------------
    // Encode / decode
    // -----------------------------------------------------------------------

    /// Encode `vector` at the given `tier`.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
    pub fn encode(&self, vector: &[f32], tier: Tier) -> Result<TieredCode> {
        match tier {
            Tier::Hot => Ok(TieredCode::Hot(self.hot.encode(vector)?)),
            Tier::Warm => Ok(TieredCode::Warm(self.warm.encode(vector)?)),
            Tier::Cold => Ok(TieredCode::Cold(self.cold.encode(vector)?)),
        }
    }

    /// Decode a `TieredCode` back to an approximate f32 vector.
    pub fn decode(&self, code: &TieredCode) -> Vec<f32> {
        match code {
            TieredCode::Hot(c) => self.hot.decode(c),
            TieredCode::Warm(c) => self.warm.decode(c),
            TieredCode::Cold(c) => self.cold.decode(c),
        }
    }

    // -----------------------------------------------------------------------
    // Estimation
    // -----------------------------------------------------------------------

    /// Estimate the inner product `<original_vector, query>` from a `TieredCode`.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
    pub fn inner_product_estimate(&self, code: &TieredCode, query: &[f32]) -> Result<f32> {
        match code {
            TieredCode::Hot(c) => self.hot.inner_product_estimate(c, query),
            TieredCode::Warm(c) => self.warm.inner_product_estimate(c, query),
            TieredCode::Cold(c) => self.cold.inner_product_estimate(c, query),
        }
    }

    /// Estimate the L2 distance `||original_vector - query||` from a `TieredCode`.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
    pub fn l2_distance_estimate(&self, code: &TieredCode, query: &[f32]) -> Result<f32> {
        match code {
            TieredCode::Hot(c) => self.hot.l2_distance_estimate(c, query),
            TieredCode::Warm(c) => self.warm.l2_distance_estimate(c, query),
            TieredCode::Cold(c) => self.cold.l2_distance_estimate(c, query),
        }
    }

    // -----------------------------------------------------------------------
    // Tier utilities
    // -----------------------------------------------------------------------

    /// Return the [`Tier`] of a `TieredCode`.
    pub fn tier(&self, code: &TieredCode) -> Tier {
        code.tier()
    }

    /// Approximate storage size of a `TieredCode` in bytes.
    pub fn code_size_bytes(&self, code: &TieredCode) -> usize {
        match code {
            TieredCode::Hot(c) => self.hot.code_size_bytes(c),
            TieredCode::Warm(c) => self.warm.code_size_bytes(c),
            TieredCode::Cold(c) => self.cold.code_size_bytes(c),
        }
    }

    /// Re-compress a `TieredCode` at a different tier.
    ///
    /// Decodes the code to an approximate f32 vector and re-encodes at
    /// `new_tier`. Note that the reconstructed vector loses the information
    /// discarded by the original quantization, so the result may be slightly
    /// less accurate than encoding the original vector directly.
    ///
    /// # Errors
    /// - `NonFiniteInput` (extremely unlikely in practice — decoded values are
    ///   always finite unless the original vector was adversarial)
    pub fn recompress(&self, code: &TieredCode, new_tier: Tier) -> Result<TieredCode> {
        let approx = self.decode(code);
        self.encode(&approx, new_tier)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 + 1.0) * 0.05).collect()
    }

    fn make_query(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * 0.03).sin()).collect()
    }

    #[test]
    fn test_new_valid() {
        assert!(TieredQuantization::new(64, 32, 42).is_ok());
    }

    #[test]
    fn test_new_zero_dim() {
        assert!(TieredQuantization::new(0, 32, 42).is_err());
    }

    #[test]
    fn test_new_odd_dim() {
        assert!(TieredQuantization::new(7, 32, 42).is_err());
    }

    #[test]
    fn test_dim() {
        let tq = TieredQuantization::new(64, 32, 42).unwrap();
        assert_eq!(tq.dim(), 64);
    }

    #[test]
    fn test_encode_hot() {
        let tq = TieredQuantization::new(16, 8, 1).unwrap();
        let v = make_vector(16);
        let code = tq.encode(&v, Tier::Hot).unwrap();
        assert!(matches!(code, TieredCode::Hot(_)));
    }

    #[test]
    fn test_encode_warm() {
        let tq = TieredQuantization::new(16, 8, 1).unwrap();
        let v = make_vector(16);
        let code = tq.encode(&v, Tier::Warm).unwrap();
        assert!(matches!(code, TieredCode::Warm(_)));
    }

    #[test]
    fn test_encode_cold() {
        let tq = TieredQuantization::new(16, 8, 1).unwrap();
        let v = make_vector(16);
        let code = tq.encode(&v, Tier::Cold).unwrap();
        assert!(matches!(code, TieredCode::Cold(_)));
    }

    #[test]
    fn test_decode_shape() {
        let tq = TieredQuantization::new(16, 8, 1).unwrap();
        let v = make_vector(16);
        for tier in [Tier::Hot, Tier::Warm, Tier::Cold] {
            let code = tq.encode(&v, tier).unwrap();
            let decoded = tq.decode(&code);
            assert_eq!(decoded.len(), 16, "tier {tier:?} decode len mismatch");
        }
    }

    #[test]
    fn test_inner_product_estimate() {
        let tq = TieredQuantization::new(32, 16, 42).unwrap();
        let v = make_vector(32);
        let q = make_query(32);
        for tier in [Tier::Hot, Tier::Warm, Tier::Cold] {
            let code = tq.encode(&v, tier).unwrap();
            let ip = tq.inner_product_estimate(&code, &q).unwrap();
            assert!(ip.is_finite(), "tier {tier:?} IP should be finite");
        }
    }

    #[test]
    fn test_l2_distance_estimate() {
        let tq = TieredQuantization::new(32, 16, 42).unwrap();
        let v = make_vector(32);
        let q = make_query(32);
        for tier in [Tier::Hot, Tier::Warm, Tier::Cold] {
            let code = tq.encode(&v, tier).unwrap();
            let l2 = tq.l2_distance_estimate(&code, &q).unwrap();
            assert!(l2 >= 0.0, "tier {tier:?} L2 should be non-negative");
            assert!(l2.is_finite(), "tier {tier:?} L2 should be finite");
        }
    }

    #[test]
    fn test_code_size_bytes_same_across_tiers() {
        // All tiers have the same byte footprint because angle indices are always
        // stored as u16 regardless of the bit width.  The bit width affects
        // quantization accuracy, not storage size.
        let tq = TieredQuantization::new(64, 32, 42).unwrap();
        let v = make_vector(64);
        let hot = tq.encode(&v, Tier::Hot).unwrap();
        let warm = tq.encode(&v, Tier::Warm).unwrap();
        let cold = tq.encode(&v, Tier::Cold).unwrap();
        assert_eq!(tq.code_size_bytes(&hot), tq.code_size_bytes(&warm));
        assert_eq!(tq.code_size_bytes(&warm), tq.code_size_bytes(&cold));
    }

    #[test]
    fn test_tier_returns_correct_variant() {
        let tq = TieredQuantization::new(16, 8, 1).unwrap();
        let v = make_vector(16);
        assert_eq!(tq.tier(&tq.encode(&v, Tier::Hot).unwrap()), Tier::Hot);
        assert_eq!(tq.tier(&tq.encode(&v, Tier::Warm).unwrap()), Tier::Warm);
        assert_eq!(tq.tier(&tq.encode(&v, Tier::Cold).unwrap()), Tier::Cold);
    }

    #[test]
    fn test_recompress_hot_to_cold() {
        let tq = TieredQuantization::new(32, 16, 42).unwrap();
        let v = make_vector(32);
        let hot = tq.encode(&v, Tier::Hot).unwrap();
        let cold = tq.recompress(&hot, Tier::Cold).unwrap();
        assert_eq!(tq.tier(&cold), Tier::Cold);
        assert_eq!(tq.decode(&cold).len(), 32);
    }

    #[test]
    fn test_recompress_cold_to_hot() {
        let tq = TieredQuantization::new(32, 16, 42).unwrap();
        let v = make_vector(32);
        let cold = tq.encode(&v, Tier::Cold).unwrap();
        let hot = tq.recompress(&cold, Tier::Hot).unwrap();
        assert_eq!(tq.tier(&hot), Tier::Hot);
    }

    #[test]
    fn test_encode_dimension_mismatch() {
        let tq = TieredQuantization::new(16, 8, 1).unwrap();
        let v = vec![0.0_f32; 8]; // wrong dim
        assert!(tq.encode(&v, Tier::Hot).is_err());
    }

    #[test]
    fn test_inner_product_self_positive() {
        // Self inner product of a positive vector should be positive.
        let tq = TieredQuantization::new(32, 16, 42).unwrap();
        let v: Vec<f32> = (1..=32).map(|i| i as f32 * 0.1).collect();
        for tier in [Tier::Hot, Tier::Warm, Tier::Cold] {
            let code = tq.encode(&v, tier).unwrap();
            let ip = tq.inner_product_estimate(&code, &v).unwrap();
            assert!(ip > 0.0, "tier {tier:?} self-IP should be positive, got {ip}");
        }
    }

    #[test]
    fn test_code_size_bytes_positive() {
        let tq = TieredQuantization::new(16, 8, 1).unwrap();
        let v = make_vector(16);
        for tier in [Tier::Hot, Tier::Warm, Tier::Cold] {
            let code = tq.encode(&v, tier).unwrap();
            assert!(tq.code_size_bytes(&code) > 0);
        }
    }

    #[test]
    fn test_tiered_code_clone() {
        // Verify TieredCode implements Clone correctly.
        let tq = TieredQuantization::new(16, 8, 1).unwrap();
        let v = make_vector(16);
        let hot = tq.encode(&v, Tier::Hot).unwrap();
        let cloned = hot.clone();
        assert_eq!(tq.tier(&cloned), Tier::Hot);
    }

    #[test]
    fn test_recompress_same_tier() {
        // Re-compressing at the same tier is valid and produces a same-tier code.
        let tq = TieredQuantization::new(32, 16, 42).unwrap();
        let v = make_vector(32);
        let warm = tq.encode(&v, Tier::Warm).unwrap();
        let rewarm = tq.recompress(&warm, Tier::Warm).unwrap();
        assert_eq!(tq.tier(&rewarm), Tier::Warm);
        assert_eq!(tq.decode(&rewarm).len(), 32);
    }

    #[test]
    fn test_recompress_warm_to_hot() {
        let tq = TieredQuantization::new(32, 16, 42).unwrap();
        let v = make_vector(32);
        let warm = tq.encode(&v, Tier::Warm).unwrap();
        let hot = tq.recompress(&warm, Tier::Hot).unwrap();
        assert_eq!(tq.tier(&hot), Tier::Hot);
    }

    #[test]
    fn test_l2_estimate_self_is_near_zero() {
        // Encoding and decoding a vector, then computing L2 to the same
        // decoded vector should give 0.
        let tq = TieredQuantization::new(32, 16, 42).unwrap();
        let v = make_vector(32);
        let hot = tq.encode(&v, Tier::Hot).unwrap();
        let decoded = tq.decode(&hot);
        // L2 distance from the decoded vector to itself must be 0.
        let l2 = tq
            .l2_distance_estimate(&tq.encode(&decoded, Tier::Hot).unwrap(), &decoded)
            .unwrap();
        assert!(l2 < 1e-3, "L2 self-distance should be ~0, got {l2}");
    }

    #[test]
    fn test_encode_non_finite_rejected() {
        let tq = TieredQuantization::new(16, 8, 1).unwrap();
        let mut v = make_vector(16);
        v[3] = f32::NAN;
        assert!(tq.encode(&v, Tier::Hot).is_err());
        assert!(tq.encode(&v, Tier::Warm).is_err());
        assert!(tq.encode(&v, Tier::Cold).is_err());
    }
}
