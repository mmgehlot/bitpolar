//! TurboQuantizer: two-stage composition of PolarQuant + QJL residual.
//!
//! Stage 1: Rotate the vector (Haar QR), then polar-encode it.
//! Stage 2: Compute the reconstruction residual and QJL-sketch it.
//! At query time, combine both estimates for an unbiased inner product.

use crate::error::{validate_finite, Result, TurboQuantError};
use crate::polar::{PolarCode, PolarQuantizer};
use crate::qjl::{QjlQuantizer, QjlSketch};
use crate::rotation::StoredRotation;
use crate::stats::BatchStats;
use crate::traits::{SerializableCode, VectorQuantizer};

// ---------------------------------------------------------------------------
// TurboCode
// ---------------------------------------------------------------------------

/// Compressed representation produced by [`TurboQuantizer`].
///
/// Combines a polar code (Stage 1) and a QJL residual sketch (Stage 2).
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct TurboCode {
    /// Stage 1: polar coordinate code
    pub(crate) polar: PolarCode,
    /// Stage 2: QJL sketch of the residual
    pub(crate) residual_sketch: QjlSketch,
}

impl TurboCode {
    /// Approximate heap size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.polar.size_bytes() + self.residual_sketch.size_bytes()
    }
}

// ---------------------------------------------------------------------------
// TurboQuantizer
// ---------------------------------------------------------------------------

/// TurboQuantizer: two-stage vector quantizer.
///
/// Fully defined by four integers `(dim, bits, projections, seed)` —
/// no training data required.
///
/// # Example
///
/// ```rust
/// use bitpolar::TurboQuantizer;
/// use bitpolar::traits::VectorQuantizer;
///
/// let q = TurboQuantizer::new(128, 4, 32, 42).unwrap();
/// let vector = vec![0.1_f32; 128];
/// let code = q.encode(&vector).unwrap();
/// let query = vec![0.05_f32; 128];
/// let score = q.inner_product_estimate(&code, &query).unwrap();
/// let reconstructed = q.decode(&code);
/// assert_eq!(reconstructed.len(), 128);
/// ```
#[derive(Debug, Clone)]
pub struct TurboQuantizer {
    dim: usize,
    bits: u8,
    num_projections: usize,
    seed: u64,
    rotation: StoredRotation,
    polar: PolarQuantizer,
    qjl: QjlQuantizer,
}

impl TurboQuantizer {
    /// Create a new TurboQuantizer.
    ///
    /// # Arguments
    /// - `dim` — vector dimension (must be even and > 0)
    /// - `bits` — polar angle quantization bit width (1..=16)
    /// - `projections` — number of QJL residual projections (> 0)
    /// - `seed` — RNG seed for the rotation and projection matrices
    ///
    /// # Errors
    /// - `ZeroDimension`, `OddDimension`, `InvalidBitWidth`, `ZeroProjections`
    pub fn new(dim: usize, bits: u8, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(TurboQuantError::ZeroDimension);
        }
        if dim % 2 != 0 {
            return Err(TurboQuantError::OddDimension(dim));
        }
        let rotation = StoredRotation::new(dim, seed)?;
        let polar = PolarQuantizer::new(dim, bits)?;
        // QJL operates on the residual, which has the same dimension.
        let qjl = QjlQuantizer::new(dim, projections, seed.wrapping_add(1))?;
        Ok(Self {
            dim,
            bits,
            num_projections: projections,
            seed,
            rotation,
            polar,
            qjl,
        })
    }

    /// The vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The polar angle quantization bit width.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// The number of QJL residual projections.
    pub fn projections(&self) -> usize {
        self.num_projections
    }

    /// The RNG seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Encode a vector into a [`TurboCode`].
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::turbo::encode",
            skip(self, vector),
            fields(dim = self.dim, bits = self.bits, projections = self.num_projections)
        )
    )]
    pub fn encode(&self, vector: &[f32]) -> Result<TurboCode> {
        if vector.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        validate_finite(vector)?;

        // Stage 1: rotate, then polar-encode.
        let mut rotated = Vec::with_capacity(self.dim);
        self.rotation.apply_slice(vector, &mut rotated);
        let polar = self.polar.encode(&rotated)?;

        // Stage 2: compute residual and QJL-sketch it.
        let reconstructed = self.polar.decode(&polar);
        let residual: Vec<f32> = rotated
            .iter()
            .zip(reconstructed.iter())
            .map(|(orig, recon)| orig - recon)
            .collect();
        let residual_sketch = self.qjl.sketch(&residual)?;

        Ok(TurboCode {
            polar,
            residual_sketch,
        })
    }

    /// Decode a [`TurboCode`] back to an approximate f32 vector.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::turbo::decode",
            skip(self, code),
            fields(dim = self.dim, bits = self.bits)
        )
    )]
    pub fn decode(&self, code: &TurboCode) -> Vec<f32> {
        // Decode polar in rotated space.
        let rotated_approx = self.polar.decode(&code.polar);
        // Un-rotate back to original space.
        let mut out = Vec::with_capacity(self.dim);
        self.rotation.apply_inverse_slice(&rotated_approx, &mut out);
        out
    }

    /// Estimate the inner product `<original_vector, query>` from a TurboCode.
    ///
    /// Combines Stage 1 (polar) and Stage 2 (QJL residual) estimates.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::turbo::ip_estimate",
            skip(self, code, query),
            fields(dim = self.dim, bits = self.bits, projections = self.num_projections)
        )
    )]
    pub fn inner_product_estimate(&self, code: &TurboCode, query: &[f32]) -> Result<f32> {
        if query.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        validate_finite(query)?;

        // Rotate the query into the same space as the encoded vector.
        let mut rotated_query = Vec::with_capacity(self.dim);
        self.rotation.apply_slice(query, &mut rotated_query);

        // Stage 1: polar inner product estimate.
        let ip_polar = self
            .polar
            .inner_product_estimate(&code.polar, &rotated_query)?;

        // Stage 2: QJL residual inner product estimate.
        let ip_residual = self
            .qjl
            .inner_product_estimate(&code.residual_sketch, &rotated_query)?;

        Ok(ip_polar + ip_residual)
    }

    /// Estimate the L2 distance `||original_vector - query||` from a TurboCode.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::turbo::l2_estimate",
            skip(self, code, query),
            fields(dim = self.dim, bits = self.bits)
        )
    )]
    pub fn l2_distance_estimate(&self, code: &TurboCode, query: &[f32]) -> Result<f32> {
        if query.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        validate_finite(query)?;

        let decoded = self.decode(code);
        let sq: f32 = decoded
            .iter()
            .zip(query.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        Ok(crate::compat::math::sqrtf(sq))
    }

    /// Compute aggregate [`BatchStats`] for a collection of codes.
    pub fn batch_stats(&self, codes: &[TurboCode]) -> BatchStats {
        let count = codes.len();
        let original_bytes = count * self.dim * core::mem::size_of::<f32>();
        let compressed_bytes: usize = codes.iter().map(|c| c.size_bytes()).sum();
        let compression_ratio = if compressed_bytes == 0 {
            0.0
        } else {
            original_bytes as f64 / compressed_bytes as f64
        };
        let bits_per_value = if count == 0 || self.dim == 0 {
            0.0
        } else {
            (compressed_bytes as f64 * 8.0) / (count as f64 * self.dim as f64)
        };
        BatchStats {
            count,
            original_bytes,
            compressed_bytes,
            compression_ratio,
            bits_per_value,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchQuantizer impl (parallel feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
impl crate::traits::BatchQuantizer for TurboQuantizer {
    /// Encode multiple vectors in parallel using rayon.
    ///
    /// All vectors must have length `self.dim()`. Returns one [`TurboCode`]
    /// per input vector, in the same order. If any vector fails validation
    /// the first error encountered is returned.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::turbo::batch_encode",
            skip(self, vectors),
            fields(dim = self.dim, bits = self.bits, batch_size = vectors.len())
        )
    )]
    fn batch_encode(&self, vectors: &[&[f32]]) -> crate::error::Result<Vec<Self::Code>> {
        use rayon::prelude::*;
        vectors
            .par_iter()
            .map(|v| self.encode(v))
            .collect::<crate::error::Result<Vec<_>>>()
    }

    /// Estimate inner products between multiple codes and a single query in parallel.
    ///
    /// Returns one score per code, in the same order. The query must have
    /// length `self.dim()`. If the query is invalid the first error is returned.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::turbo::batch_inner_product",
            skip(self, codes, query),
            fields(dim = self.dim, bits = self.bits, batch_size = codes.len())
        )
    )]
    fn batch_inner_product(
        &self,
        codes: &[Self::Code],
        query: &[f32],
    ) -> crate::error::Result<Vec<f32>> {
        use rayon::prelude::*;
        codes
            .par_iter()
            .map(|c| self.inner_product_estimate(c, query))
            .collect::<crate::error::Result<Vec<_>>>()
    }

    /// Decode multiple codes in parallel.
    ///
    /// Returns one reconstructed f32 vector per code, in the same order.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::turbo::batch_decode",
            skip(self, codes),
            fields(dim = self.dim, bits = self.bits, batch_size = codes.len())
        )
    )]
    fn batch_decode(&self, codes: &[Self::Code]) -> Vec<Vec<f32>> {
        use rayon::prelude::*;
        codes.par_iter().map(|c| self.decode(c)).collect()
    }
}

// ---------------------------------------------------------------------------
// VectorQuantizer impl
// ---------------------------------------------------------------------------

impl VectorQuantizer for TurboQuantizer {
    type Code = TurboCode;

    fn encode(&self, vector: &[f32]) -> Result<Self::Code> {
        TurboQuantizer::encode(self, vector)
    }

    fn decode(&self, code: &Self::Code) -> Vec<f32> {
        TurboQuantizer::decode(self, code)
    }

    fn inner_product_estimate(&self, code: &Self::Code, query: &[f32]) -> Result<f32> {
        TurboQuantizer::inner_product_estimate(self, code, query)
    }

    fn l2_distance_estimate(&self, code: &Self::Code, query: &[f32]) -> Result<f32> {
        TurboQuantizer::l2_distance_estimate(self, code, query)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn code_size_bytes(&self, code: &Self::Code) -> usize {
        code.size_bytes()
    }
}

// ---------------------------------------------------------------------------
// Compact binary serialization for TurboCode
// ---------------------------------------------------------------------------

impl TurboCode {
    /// Serialize to compact binary.
    ///
    /// Format:
    /// ```text
    /// [version: u8][polar_len: u32 LE][polar_compact_bytes...][qjl_compact_bytes...]
    /// ```
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let polar_bytes = self.polar.to_compact_bytes();
        let qjl_bytes = self.residual_sketch.to_compact_bytes();

        let mut out = Vec::with_capacity(1 + 4 + polar_bytes.len() + qjl_bytes.len());
        out.push(crate::COMPACT_FORMAT_VERSION);
        let polar_len: u32 = polar_bytes
            .len()
            .try_into()
            .expect("polar payload exceeds u32::MAX; dimension too large for compact format");
        out.extend_from_slice(&polar_len.to_le_bytes());
        out.extend_from_slice(&polar_bytes);
        out.extend_from_slice(&qjl_bytes);
        out
    }

    /// Deserialize from compact binary bytes produced by [`to_compact_bytes`](Self::to_compact_bytes).
    ///
    /// # Errors
    /// - `DeserializationError` if the buffer is too short, has a wrong version,
    ///   or the inner polar/qjl payloads are invalid.
    pub fn from_compact_bytes(bytes: &[u8]) -> Result<Self> {
        let err = |reason: &str| TurboQuantError::DeserializationError {
            reason: reason.to_string(),
        };

        if bytes.is_empty() {
            return Err(err("buffer is empty"));
        }
        if bytes[0] != crate::COMPACT_FORMAT_VERSION {
            return Err(err(&format!(
                "unsupported version 0x{:02X}, expected 0x{:02X}",
                bytes[0],
                crate::COMPACT_FORMAT_VERSION
            )));
        }
        // Need at least version(1) + polar_len(4)
        if bytes.len() < 5 {
            return Err(err("buffer too short for polar length prefix"));
        }
        let polar_len = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
        let polar_start = 5;
        let polar_end = polar_start + polar_len;
        if bytes.len() < polar_end {
            return Err(err("buffer too short for polar payload"));
        }
        let polar = PolarCode::from_compact_bytes(&bytes[polar_start..polar_end])?;
        let qjl_bytes = &bytes[polar_end..];
        if qjl_bytes.is_empty() {
            return Err(err("missing qjl payload"));
        }
        let residual_sketch = QjlSketch::from_compact_bytes(qjl_bytes)?;
        Ok(Self {
            polar,
            residual_sketch,
        })
    }
}

impl SerializableCode for TurboCode {
    #[inline]
    fn to_compact_bytes(&self) -> Vec<u8> {
        TurboCode::to_compact_bytes(self)
    }

    #[inline]
    fn from_compact_bytes(bytes: &[u8]) -> Result<Self> {
        TurboCode::from_compact_bytes(bytes)
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::VectorQuantizer;

    #[test]
    fn test_zero_dimension_error() {
        assert!(matches!(
            TurboQuantizer::new(0, 4, 32, 42),
            Err(TurboQuantError::ZeroDimension)
        ));
    }

    #[test]
    fn test_odd_dimension_error() {
        assert!(matches!(
            TurboQuantizer::new(3, 4, 32, 42),
            Err(TurboQuantError::OddDimension(3))
        ));
    }

    #[test]
    fn test_encode_decode_shape() {
        let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
        let v: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let code = q.encode(&v).unwrap();
        let decoded = q.decode(&code);
        assert_eq!(decoded.len(), 8);
    }

    #[test]
    fn test_dimension_mismatch() {
        let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
        let v = vec![0.0_f32; 4];
        assert!(matches!(
            q.encode(&v),
            Err(TurboQuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_batch_stats() {
        let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
        let v: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let codes: Vec<TurboCode> = (0..10).map(|_| q.encode(&v).unwrap()).collect();
        let stats = q.batch_stats(&codes);
        assert_eq!(stats.count, 10);
        assert!(stats.compression_ratio > 0.0);
    }

    #[test]
    fn test_inner_product_positive() {
        let q = TurboQuantizer::new(16, 4, 32, 42).unwrap();
        let v: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let code = q.encode(&v).unwrap();
        let est = q.inner_product_estimate(&code, &v).unwrap();
        let exact: f32 = v.iter().map(|x| x * x).sum();
        // Should be in the right direction.
        assert!(est > 0.0, "estimate={est} exact={exact}");
    }

    #[test]
    fn test_trait_object_compiles() {
        let q: Box<dyn VectorQuantizer<Code = TurboCode>> =
            Box::new(TurboQuantizer::new(8, 4, 16, 42).unwrap());
        assert_eq!(q.dim(), 8);
    }

    // -----------------------------------------------------------------------
    // Compact serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_turbo_code_roundtrip() {
        let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
        let v: Vec<f32> = (0..8).map(|i| i as f32 * 0.3 - 1.0).collect();
        let code = q.encode(&v).unwrap();
        let bytes = code.to_compact_bytes();
        let decoded = TurboCode::from_compact_bytes(&bytes).unwrap();
        assert_eq!(decoded.polar.radii, code.polar.radii);
        assert_eq!(decoded.polar.angle_indices, code.polar.angle_indices);
        assert_eq!(decoded.polar.bits, code.polar.bits);
        assert_eq!(decoded.residual_sketch.signs, code.residual_sketch.signs);
        assert_eq!(
            decoded.residual_sketch.num_projections,
            code.residual_sketch.num_projections
        );
        assert_eq!(decoded.residual_sketch.norm, code.residual_sketch.norm);
    }

    #[test]
    fn test_turbo_code_wrong_version() {
        let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
        let v = vec![0.1_f32; 8];
        let code = q.encode(&v).unwrap();
        let mut bytes = code.to_compact_bytes();
        bytes[0] = 0xFF;
        assert!(matches!(
            TurboCode::from_compact_bytes(&bytes),
            Err(TurboQuantError::DeserializationError { .. })
        ));
    }

    #[test]
    fn test_turbo_code_truncated() {
        let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
        let v = vec![0.1_f32; 8];
        let code = q.encode(&v).unwrap();
        let bytes = code.to_compact_bytes();
        // Try various truncations
        for len in [0usize, 1, 3, 5, 8] {
            let truncated = &bytes[..len.min(bytes.len() - 1)];
            assert!(
                TurboCode::from_compact_bytes(truncated).is_err(),
                "expected error for len={len}"
            );
        }
    }

    #[test]
    fn test_turbo_code_empty_buffer() {
        assert!(matches!(
            TurboCode::from_compact_bytes(&[]),
            Err(TurboQuantError::DeserializationError { .. })
        ));
    }

    #[test]
    fn test_serializable_code_trait() {
        use crate::traits::SerializableCode;
        let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
        let v = vec![0.5_f32; 8];
        let code = q.encode(&v).unwrap();
        let bytes = <TurboCode as SerializableCode>::to_compact_bytes(&code);
        let decoded = <TurboCode as SerializableCode>::from_compact_bytes(&bytes).unwrap();
        assert_eq!(decoded.polar.bits, code.polar.bits);
    }

    // -----------------------------------------------------------------------
    // Batch operation tests (compiled always, executed under parallel feature)
    // -----------------------------------------------------------------------

    #[cfg(feature = "parallel")]
    mod batch_tests {
        use super::*;
        use crate::traits::BatchQuantizer;

        fn make_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
            (0..n)
                .map(|i| {
                    (0..dim)
                        .map(|j| ((i * dim + j) as f32 * 0.1).sin())
                        .collect()
                })
                .collect()
        }

        #[test]
        fn test_batch_encode_matches_sequential() {
            let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
            let vecs = make_vectors(10, 8);
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

            let batch_codes = q.batch_encode(&refs).unwrap();
            for (i, code) in batch_codes.iter().enumerate() {
                let seq_code = q.encode(&vecs[i]).unwrap();
                assert_eq!(code.polar.radii, seq_code.polar.radii);
                assert_eq!(code.polar.angle_indices, seq_code.polar.angle_indices);
                assert_eq!(code.residual_sketch.signs, seq_code.residual_sketch.signs);
            }
        }

        #[test]
        fn test_batch_inner_product_matches_sequential() {
            let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
            let vecs = make_vectors(10, 8);
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            let codes = q.batch_encode(&refs).unwrap();
            let query = vec![0.1_f32; 8];

            let batch_scores = q.batch_inner_product(&codes, &query).unwrap();
            for (i, &score) in batch_scores.iter().enumerate() {
                let seq_score = q.inner_product_estimate(&codes[i], &query).unwrap();
                assert!(
                    (score - seq_score).abs() < 1e-6,
                    "score mismatch at index {i}"
                );
            }
        }

        #[test]
        fn test_batch_decode_matches_sequential() {
            let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
            let vecs = make_vectors(5, 8);
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            let codes = q.batch_encode(&refs).unwrap();

            let batch_decoded = q.batch_decode(&codes);
            for (i, decoded) in batch_decoded.iter().enumerate() {
                let seq_decoded = q.decode(&codes[i]);
                for (a, b) in decoded.iter().zip(seq_decoded.iter()) {
                    assert!((a - b).abs() < 1e-6);
                }
            }
        }

        #[test]
        fn test_batch_encode_empty() {
            let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
            let codes = q.batch_encode(&[]).unwrap();
            assert!(codes.is_empty());
        }

        #[test]
        fn test_batch_encode_single() {
            let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
            let v = vec![0.1_f32; 8];
            let refs: &[&[f32]] = &[&v];
            let codes = q.batch_encode(refs).unwrap();
            assert_eq!(codes.len(), 1);
            let seq = q.encode(&v).unwrap();
            assert_eq!(codes[0].polar.radii, seq.polar.radii);
        }

        #[test]
        fn test_batch_encode_error_propagates() {
            let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
            let v_ok = vec![0.1_f32; 8];
            let v_bad = vec![0.0_f32; 4]; // wrong dim
            let refs: &[&[f32]] = &[&v_ok, &v_bad];
            assert!(q.batch_encode(refs).is_err());
        }

        #[test]
        fn test_batch_inner_product_empty() {
            let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
            let query = vec![0.1_f32; 8];
            let scores = q.batch_inner_product(&[], &query).unwrap();
            assert!(scores.is_empty());
        }

        #[test]
        fn test_batch_decode_empty() {
            let q = TurboQuantizer::new(8, 4, 16, 42).unwrap();
            let result = q.batch_decode(&[]);
            assert!(result.is_empty());
        }
    }
}
