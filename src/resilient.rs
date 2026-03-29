//! Resilient quantization with automatic fallback.
//!
//! `ResilientQuantizer` wraps a primary [`TurboQuantizer`] and a fallback
//! [`PolarQuantizer`]. On encode, the primary is tried first; if it returns
//! an error the fallback is used instead, so callers always get a code.
//!
//! This is useful in production pipelines where partial failures (e.g., a
//! misconfigured quantizer for a particular domain) must not propagate as
//! hard errors.
//!
//! # Example
//!
//! ```rust
//! use bitpolar::resilient::ResilientQuantizer;
//!
//! let rq = ResilientQuantizer::new(64, 4, 32, 42, 4).unwrap();
//! let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
//!
//! let code = rq.encode(&v).unwrap();
//! assert!(!rq.is_fallback(&code));
//!
//! let query: Vec<f32> = vec![0.5_f32; 64];
//! let _score = rq.inner_product_estimate(&code, &query).unwrap();
//! ```

use crate::error::Result;
use crate::polar::{PolarCode, PolarQuantizer};
use crate::traits::VectorQuantizer;
use crate::turbo::{TurboCode, TurboQuantizer};

// ---------------------------------------------------------------------------
// ResilientCode
// ---------------------------------------------------------------------------

/// A code produced by [`ResilientQuantizer`], tagged by which encoder was used.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum ResilientCode {
    /// Encoded by the primary [`TurboQuantizer`].
    Primary(TurboCode),
    /// Encoded by the fallback [`PolarQuantizer`].
    Fallback(PolarCode),
}

// ---------------------------------------------------------------------------
// ResilientQuantizer
// ---------------------------------------------------------------------------

/// A quantizer that falls back to a simpler algorithm on primary failure.
///
/// Holds a [`TurboQuantizer`] as the primary encoder and a [`PolarQuantizer`]
/// as the fallback. `encode()` tries the primary first and uses the fallback
/// only when the primary returns an error.
///
/// # Errors
///
/// Construction fails if either quantizer fails to construct (e.g. zero or
/// odd dimension).
#[derive(Debug, Clone)]
pub struct ResilientQuantizer {
    primary: TurboQuantizer,
    fallback: PolarQuantizer,
}

impl ResilientQuantizer {
    /// Create a new `ResilientQuantizer`.
    ///
    /// # Arguments
    /// - `dim` — vector dimension (must be even and > 0)
    /// - `primary_bits` — bit width for the primary TurboQuantizer (1..=16)
    /// - `projections` — number of QJL projections for the primary (> 0)
    /// - `seed` — RNG seed for the primary's rotation and projection matrices
    /// - `fallback_bits` — bit width for the fallback PolarQuantizer (1..=16)
    ///
    /// # Errors
    /// - `ZeroDimension`, `OddDimension`, `InvalidBitWidth`, `ZeroProjections`
    pub fn new(
        dim: usize,
        primary_bits: u8,
        projections: usize,
        seed: u64,
        fallback_bits: u8,
    ) -> Result<Self> {
        let primary = TurboQuantizer::new(dim, primary_bits, projections, seed)?;
        let fallback = PolarQuantizer::new(dim, fallback_bits)?;
        Ok(Self { primary, fallback })
    }

    /// The vector dimension.
    pub fn dim(&self) -> usize {
        self.primary.dim()
    }

    // -----------------------------------------------------------------------
    // Encode / decode
    // -----------------------------------------------------------------------

    /// Encode a vector, using the primary encoder or falling back automatically.
    ///
    /// Returns `Ok(ResilientCode::Primary(...))` if the primary succeeds, or
    /// `Ok(ResilientCode::Fallback(...))` if the primary fails but the fallback
    /// succeeds. Returns `Err` only if both encoders fail.
    pub fn encode(&self, vector: &[f32]) -> Result<ResilientCode> {
        match self.primary.encode(vector) {
            Ok(code) => Ok(ResilientCode::Primary(code)),
            Err(_primary_err) => {
                // Try the fallback; propagate its error if it also fails.
                let code = self.fallback.encode(vector)?;
                Ok(ResilientCode::Fallback(code))
            }
        }
    }

    /// Decode a `ResilientCode` back to an approximate f32 vector.
    pub fn decode(&self, code: &ResilientCode) -> Vec<f32> {
        match code {
            ResilientCode::Primary(c) => self.primary.decode(c),
            ResilientCode::Fallback(c) => self.fallback.decode(c),
        }
    }

    // -----------------------------------------------------------------------
    // Estimation
    // -----------------------------------------------------------------------

    /// Estimate the inner product `<original_vector, query>` from a `ResilientCode`.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
    pub fn inner_product_estimate(&self, code: &ResilientCode, query: &[f32]) -> Result<f32> {
        match code {
            ResilientCode::Primary(c) => self.primary.inner_product_estimate(c, query),
            ResilientCode::Fallback(c) => self.fallback.inner_product_estimate(c, query),
        }
    }

    /// Estimate the L2 distance `||original_vector - query||` from a `ResilientCode`.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
    pub fn l2_distance_estimate(&self, code: &ResilientCode, query: &[f32]) -> Result<f32> {
        match code {
            ResilientCode::Primary(c) => self.primary.l2_distance_estimate(c, query),
            ResilientCode::Fallback(c) => self.fallback.l2_distance_estimate(c, query),
        }
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    /// Returns `true` if the code was produced by the fallback encoder.
    pub fn is_fallback(&self, code: &ResilientCode) -> bool {
        matches!(code, ResilientCode::Fallback(_))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rq(dim: usize) -> ResilientQuantizer {
        ResilientQuantizer::new(dim, 4, dim.max(4), 42, 4).unwrap()
    }

    fn make_vector(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 + 1.0) * 0.05).collect()
    }

    fn make_query(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * 0.03).sin()).collect()
    }

    #[test]
    fn test_new_valid() {
        assert!(ResilientQuantizer::new(32, 4, 16, 42, 4).is_ok());
    }

    #[test]
    fn test_new_zero_dim() {
        assert!(ResilientQuantizer::new(0, 4, 16, 42, 4).is_err());
    }

    #[test]
    fn test_new_odd_dim() {
        assert!(ResilientQuantizer::new(7, 4, 16, 42, 4).is_err());
    }

    #[test]
    fn test_dim() {
        assert_eq!(make_rq(32).dim(), 32);
    }

    #[test]
    fn test_encode_primary() {
        let rq = make_rq(32);
        let v = make_vector(32);
        let code = rq.encode(&v).unwrap();
        assert!(matches!(code, ResilientCode::Primary(_)));
        assert!(!rq.is_fallback(&code));
    }

    #[test]
    fn test_decode_shape() {
        let rq = make_rq(32);
        let v = make_vector(32);
        let code = rq.encode(&v).unwrap();
        let decoded = rq.decode(&code);
        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn test_inner_product_estimate_finite() {
        let rq = make_rq(32);
        let v = make_vector(32);
        let q = make_query(32);
        let code = rq.encode(&v).unwrap();
        let ip = rq.inner_product_estimate(&code, &q).unwrap();
        assert!(ip.is_finite());
    }

    #[test]
    fn test_l2_distance_estimate_non_negative() {
        let rq = make_rq(32);
        let v = make_vector(32);
        let q = make_query(32);
        let code = rq.encode(&v).unwrap();
        let l2 = rq.l2_distance_estimate(&code, &q).unwrap();
        assert!(l2 >= 0.0);
        assert!(l2.is_finite());
    }

    #[test]
    fn test_fallback_code_is_fallback() {
        let rq = make_rq(32);
        let v = make_vector(32);
        // Build a fallback code directly to test is_fallback().
        let fallback_code = rq.fallback.encode(&v).unwrap();
        let resilient = ResilientCode::Fallback(fallback_code);
        assert!(rq.is_fallback(&resilient));
    }

    #[test]
    fn test_fallback_decode_shape() {
        let rq = make_rq(32);
        let v = make_vector(32);
        let fallback_code = rq.fallback.encode(&v).unwrap();
        let resilient = ResilientCode::Fallback(fallback_code);
        let decoded = rq.decode(&resilient);
        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn test_fallback_inner_product_finite() {
        let rq = make_rq(32);
        let v = make_vector(32);
        let q = make_query(32);
        let fallback_code = rq.fallback.encode(&v).unwrap();
        let resilient = ResilientCode::Fallback(fallback_code);
        let ip = rq.inner_product_estimate(&resilient, &q).unwrap();
        assert!(ip.is_finite());
    }

    #[test]
    fn test_fallback_l2_non_negative() {
        let rq = make_rq(32);
        let v = make_vector(32);
        let q = make_query(32);
        let fallback_code = rq.fallback.encode(&v).unwrap();
        let resilient = ResilientCode::Fallback(fallback_code);
        let l2 = rq.l2_distance_estimate(&resilient, &q).unwrap();
        assert!(l2 >= 0.0);
    }

    #[test]
    fn test_dimension_mismatch_both_fail() {
        let rq = make_rq(32);
        let v = vec![0.0_f32; 8]; // wrong dim
                                  // Both primary and fallback have dim=32, so both fail.
        assert!(rq.encode(&v).is_err());
    }

    #[test]
    fn test_inner_product_self_positive() {
        let rq = make_rq(32);
        let v: Vec<f32> = (1..=32).map(|i| i as f32 * 0.1).collect();
        let code = rq.encode(&v).unwrap();
        let ip = rq.inner_product_estimate(&code, &v).unwrap();
        assert!(ip > 0.0, "self-IP should be positive, got {ip}");
    }

    #[test]
    fn test_resilient_code_clone() {
        let rq = make_rq(32);
        let v = make_vector(32);
        let code = rq.encode(&v).unwrap();
        let cloned = code.clone();
        // Both should give the same IP estimate.
        let q = make_query(32);
        let ip1 = rq.inner_product_estimate(&code, &q).unwrap();
        let ip2 = rq.inner_product_estimate(&cloned, &q).unwrap();
        assert!(
            (ip1 - ip2).abs() < 1e-6,
            "clone should produce identical estimates"
        );
    }

    #[test]
    fn test_non_finite_rejected() {
        let rq = make_rq(32);
        let mut v = make_vector(32);
        v[5] = f32::INFINITY;
        // Both primary and fallback should reject non-finite input.
        assert!(rq.encode(&v).is_err());
    }

    #[test]
    fn test_l2_non_negative_fallback() {
        let rq = make_rq(32);
        let v = make_vector(32);
        let q = make_query(32);
        let fallback_code = rq.fallback.encode(&v).unwrap();
        let resilient = ResilientCode::Fallback(fallback_code);
        let l2 = rq.l2_distance_estimate(&resilient, &q).unwrap();
        assert!(l2 >= 0.0);
        assert!(l2.is_finite());
    }
}
