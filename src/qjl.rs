//! Quantized Johnson-Lindenstrauss (QJL) 1-bit sketching.
//!
//! Projects vectors through a random Gaussian matrix and stores only sign bits.
//! Provides **provably unbiased** inner product estimation with variance
//! bounded by `π / (2 * projections) * ||y||²`.
//!
//! Used standalone for ultra-lightweight similarity or as TurboQuant's
//! residual correction stage (Stage 2).
//!
//! # Algorithm
//!
//! Given a random Gaussian matrix `S ∈ ℝ^{m×d}` (m = projections, d = dimension):
//!
//! **Sketch:** `sketch(x) = { sign(S·x), ||x||₂ }`
//!
//! **Inner product estimate:**
//! ```text
//! ⟨x, q⟩ ≈ ||x|| · √(π/2) / m · Σᵢ sign(gᵢ·x) · (gᵢ·q)
//! ```
//!
//! This estimator is **unbiased**: `E[estimate] = ⟨x, q⟩`.
//!
//! # References
//!
//! - QJL paper (AAAI 2025): <https://arxiv.org/abs/2406.03482>

use crate::error::{validate_finite, Result, TurboQuantError};
use crate::traits::SerializableCode;

/// A 1-bit QJL sketch of a vector.
///
/// Stores only the sign bits from random projections (packed into bytes)
/// and the original vector's L2 norm. The projection matrix is NOT stored
/// in the sketch — it lives in the [`QjlQuantizer`] and is shared across
/// all sketches.
///
/// # Memory
///
/// Total size: `ceil(projections / 8) + 4` bytes (signs + norm).
/// For projections=384: 52 bytes regardless of vector dimension.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct QjlSketch {
    /// Packed sign bits: bit i = 1 if projection i had positive dot product.
    /// Bit order within each byte: bit 0 = LSB (projection p is at byte p/8, bit p%8).
    pub(crate) signs: Vec<u8>,
    /// Number of projections (not all bits in last byte may be used)
    pub(crate) num_projections: usize,
    /// L2 norm of the original vector (needed for IP scale recovery)
    pub(crate) norm: f32,
}

impl QjlSketch {
    /// Approximate size of this sketch in bytes.
    ///
    /// This is the storage cost per vector — independent of the dimension.
    pub fn size_bytes(&self) -> usize {
        // signs bytes + num_projections (usize stored as u16 in compact) + norm (f32)
        self.signs.len() + 2 + 4
    }

    /// Number of projections in this sketch.
    pub fn num_projections(&self) -> usize {
        self.num_projections
    }

    /// L2 norm of the original vector.
    pub fn norm(&self) -> f32 {
        self.norm
    }

    /// Get the sign bit for projection `p` (true = positive, false = negative).
    #[inline]
    fn get_sign(&self, p: usize) -> f32 {
        let byte_idx = p / 8;
        let bit_idx = p % 8;
        if (self.signs[byte_idx] >> bit_idx) & 1 == 1 {
            1.0
        } else {
            -1.0
        }
    }
}

// ---------------------------------------------------------------------------
// Compact binary serialization for QjlSketch
// ---------------------------------------------------------------------------

impl QjlSketch {
    /// Serialize to compact binary.
    ///
    /// Format:
    /// ```text
    /// [version: u8 = 0x01][num_projections: u16 LE][norm: f32 LE]
    /// [signs: ceil(num_projections/8) bytes]
    /// ```
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let n_sign_bytes = self.num_projections.div_ceil(8);
        let mut out = Vec::with_capacity(1 + 2 + 4 + n_sign_bytes);
        out.push(crate::COMPACT_FORMAT_VERSION);
        let np: u16 = self.num_projections.try_into().expect(
            "QjlSketch num_projections exceeds u16::MAX; too many projections for compact format",
        );
        out.extend_from_slice(&np.to_le_bytes());
        out.extend_from_slice(&self.norm.to_le_bytes());
        out.extend_from_slice(&self.signs);
        out
    }

    /// Deserialize from compact binary bytes produced by
    /// [`to_compact_bytes`](Self::to_compact_bytes).
    ///
    /// # Errors
    /// - `DeserializationError` if the buffer is too short, version is wrong,
    ///   or the sign bytes count is inconsistent with `num_projections`.
    pub fn from_compact_bytes(bytes: &[u8]) -> Result<Self> {
        let err = |reason: &str| TurboQuantError::DeserializationError {
            reason: reason.to_string(),
        };

        // Header: version(1) + num_projections(2) + norm(4) = 7 bytes minimum
        if bytes.len() < 7 {
            return Err(err("buffer too short: need at least 7 bytes"));
        }
        if bytes[0] != crate::COMPACT_FORMAT_VERSION {
            return Err(err(&format!(
                "unsupported version 0x{:02X}, expected 0x{:02X}",
                bytes[0],
                crate::COMPACT_FORMAT_VERSION
            )));
        }
        let num_projections = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;
        if num_projections == 0 {
            return Err(err("num_projections must be > 0"));
        }
        let norm = f32::from_le_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]);
        if norm.is_nan() {
            return Err(err("norm is NaN"));
        }
        if norm.is_infinite() {
            return Err(err("norm is infinite"));
        }
        if norm < 0.0 {
            return Err(err("norm is negative"));
        }

        let n_sign_bytes = num_projections.div_ceil(8);
        let expected_len = 7 + n_sign_bytes;
        if bytes.len() < expected_len {
            return Err(err(&format!(
                "buffer too short: need {expected_len}, got {}",
                bytes.len()
            )));
        }

        let signs = bytes[7..7 + n_sign_bytes].to_vec();
        Ok(Self { signs, num_projections, norm })
    }
}

impl SerializableCode for QjlSketch {
    #[inline]
    fn to_compact_bytes(&self) -> Vec<u8> {
        QjlSketch::to_compact_bytes(self)
    }

    #[inline]
    fn from_compact_bytes(bytes: &[u8]) -> Result<Self> {
        QjlSketch::from_compact_bytes(bytes)
    }
}

// ---------------------------------------------------------------------------
// QjlQuantizer
// ---------------------------------------------------------------------------

/// QJL (Quantized Johnson-Lindenstrauss) 1-bit sketcher.
///
/// Builds a random Gaussian projection matrix once from `(dim, projections, seed)`
/// and reuses it for all sketch and estimation operations.
///
/// # Thread Safety
///
/// The quantizer is immutable after construction — `Send + Sync` by default.
/// Safe to share across threads via `Arc<QjlQuantizer>`.
///
/// # Memory
///
/// The projection matrix occupies `projections × dim × 4` bytes.
/// For dim=1536, projections=384: ~2.4 MB.
///
/// # Example
///
/// ```rust
/// use bitpolar::QjlQuantizer;
///
/// let q = QjlQuantizer::new(64, 32, 42).unwrap();
/// let v = vec![0.3_f32; 64];
/// let sketch = q.sketch(&v).unwrap();
///
/// let query = vec![0.2_f32; 64];
/// let est = q.inner_product_estimate(&sketch, &query).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct QjlQuantizer {
    /// Vector dimension
    dim: usize,
    /// Number of random projections (= number of sign bits per sketch)
    num_projections: usize,
    /// RNG seed for reproducibility
    seed: u64,
    /// Random Gaussian projection matrix: num_projections × dim, row-major.
    /// Each row g_i ~ N(0, I_d) (standard normal, NOT scaled by 1/√d).
    projection_matrix: Vec<f32>,
}

impl QjlQuantizer {
    /// Create a new QJL quantizer.
    ///
    /// # Arguments
    /// - `dim` — vector dimension (must be > 0)
    /// - `projections` — number of random projections (must be > 0; dim/4 recommended)
    /// - `seed` — RNG seed for deterministic projection matrix
    ///
    /// # Errors
    /// - `ZeroDimension` if `dim == 0`
    /// - `ZeroProjections` if `projections == 0`
    pub fn new(dim: usize, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(TurboQuantError::ZeroDimension);
        }
        if projections == 0 {
            return Err(TurboQuantError::ZeroProjections);
        }
        // num_projections is stored as u16 in compact format; reject oversized values.
        if projections > u16::MAX as usize {
            return Err(TurboQuantError::DimensionTooLarge(projections, u16::MAX as usize));
        }

        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::Distribution;
        use rand_distr::StandardNormal;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Generate standard normal projection matrix.
        // Each entry ~ N(0, 1). We do NOT scale by 1/√d here —
        // the scaling is handled in the estimation formula.
        let projection_matrix: Vec<f32> = (0..projections * dim)
            .map(|_| {
                <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng) as f32
            })
            .collect();

        Ok(Self {
            dim,
            num_projections: projections,
            seed,
            projection_matrix,
        })
    }

    /// The vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of 1-bit projections.
    pub fn projections(&self) -> usize {
        self.num_projections
    }

    /// The RNG seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Memory footprint of the projection matrix in bytes.
    pub fn matrix_size_bytes(&self) -> usize {
        self.projection_matrix.len() * core::mem::size_of::<f32>()
    }

    /// Compute the 1-bit sketch of a vector.
    ///
    /// For each projection direction g_i, stores `sign(g_i · v)` as a single bit.
    /// Also stores `||v||₂` for scale recovery during IP estimation.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `vector.len() != self.dim()`
    /// - `NonFiniteInput` if any element is NaN or Inf
    #[inline]
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::qjl::sketch",
            skip(self, vector),
            fields(dim = self.dim, projections = self.num_projections)
        )
    )]
    pub fn sketch(&self, vector: &[f32]) -> Result<QjlSketch> {
        if vector.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        validate_finite(vector)?;

        // Compute L2 norm
        let norm_sq: f32 = vector.iter().map(|x| x * x).sum::<f32>();
        let norm = crate::compat::math::sqrtf(norm_sq);

        // Compute sign bits for each projection: sign(g_i · v)
        let n_bytes = self.num_projections.div_ceil(8);
        let mut signs = vec![0u8; n_bytes];

        for p in 0..self.num_projections {
            let row_start = p * self.dim;
            let dot: f32 = self.projection_matrix[row_start..row_start + self.dim]
                .iter()
                .zip(vector.iter())
                .map(|(m, v)| m * v)
                .sum();
            // Store positive as 1, negative as 0
            if dot >= 0.0 {
                let byte_idx = p / 8;
                let bit_idx = p % 8;
                signs[byte_idx] |= 1 << bit_idx;
            }
        }

        Ok(QjlSketch {
            signs,
            num_projections: self.num_projections,
            norm,
        })
    }

    /// Estimate inner product between a sketch and a raw query vector.
    ///
    /// Uses the paper's unbiased estimator:
    /// ```text
    /// ⟨v, q⟩ ≈ ||v|| · √(π/2) / m · Σᵢ sign(gᵢ·v) · (gᵢ·q)
    /// ```
    ///
    /// This estimator is **provably unbiased**: `E[estimate] = ⟨v, q⟩`.
    ///
    /// # Arguments
    /// - `sketch` — 1-bit sketch of the stored vector
    /// - `query` — full-precision query vector
    ///
    /// # Errors
    /// - `DimensionMismatch` if `query.len() != self.dim()`
    /// - `NonFiniteInput` if any element is NaN or Inf
    #[inline]
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::qjl::ip_estimate",
            skip(self, sketch, query),
            fields(dim = self.dim, projections = self.num_projections)
        )
    )]
    pub fn inner_product_estimate(
        &self,
        sketch: &QjlSketch,
        query: &[f32],
    ) -> Result<f32> {
        if query.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        validate_finite(query)?;

        // If either vector is zero, IP is zero
        if sketch.norm < 1e-30 {
            return Ok(0.0);
        }

        // Scale factor: ||v|| * √(π/2) / m
        let scale =
            sketch.norm * crate::compat::math::sqrtf(core::f32::consts::FRAC_PI_2) / self.num_projections as f32;

        // Sum over projections: sign(gᵢ·v) * (gᵢ·q)
        let mut sum = 0.0f32;
        for p in 0..self.num_projections {
            // Get the sign bit for this projection
            let sign = sketch.get_sign(p);

            // Compute g_p · query
            let row_start = p * self.dim;
            let dot: f32 = self.projection_matrix[row_start..row_start + self.dim]
                .iter()
                .zip(query.iter())
                .map(|(m, q)| m * q)
                .sum();

            sum += sign * dot;
        }

        Ok(scale * sum)
    }

    /// Estimate inner product using a pre-provided norm (for residual correction).
    ///
    /// Same as [`inner_product_estimate`](Self::inner_product_estimate) but allows
    /// overriding the norm value. Used by TurboQuantizer when the sketch's norm
    /// is the residual norm.
    pub fn inner_product_estimate_with_norm(
        &self,
        sketch: &QjlSketch,
        query: &[f32],
        norm: f32,
    ) -> Result<f32> {
        if query.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        validate_finite(query)?;

        // Treat non-finite or negative norm as a zero-magnitude vector.
        if !norm.is_finite() || norm < 0.0 {
            return Ok(0.0);
        }

        if norm < 1e-30 {
            return Ok(0.0);
        }

        let scale = norm * crate::compat::math::sqrtf(core::f32::consts::FRAC_PI_2) / self.num_projections as f32;

        let mut sum = 0.0f32;
        for p in 0..self.num_projections {
            let sign = sketch.get_sign(p);
            let row_start = p * self.dim;
            let dot: f32 = self.projection_matrix[row_start..row_start + self.dim]
                .iter()
                .zip(query.iter())
                .map(|(m, q)| m * q)
                .sum();
            sum += sign * dot;
        }

        Ok(scale * sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_dimension_error() {
        assert!(matches!(
            QjlQuantizer::new(0, 16, 42),
            Err(TurboQuantError::ZeroDimension)
        ));
    }

    #[test]
    fn test_zero_projections_error() {
        assert!(matches!(
            QjlQuantizer::new(8, 0, 42),
            Err(TurboQuantError::ZeroProjections)
        ));
    }

    #[test]
    fn test_sketch_shape() {
        let q = QjlQuantizer::new(8, 16, 42).unwrap();
        let v = vec![0.5_f32; 8];
        let sketch = q.sketch(&v).unwrap();
        assert_eq!(sketch.signs.len(), 2); // ceil(16/8) = 2 bytes
        assert_eq!(sketch.num_projections(), 16);
    }

    #[test]
    fn test_sketch_odd_projections() {
        // 13 projections → 2 bytes (ceil(13/8) = 2)
        let q = QjlQuantizer::new(8, 13, 42).unwrap();
        let v = vec![0.5_f32; 8];
        let sketch = q.sketch(&v).unwrap();
        assert_eq!(sketch.signs.len(), 2);
    }

    #[test]
    fn test_dimension_mismatch() {
        let q = QjlQuantizer::new(8, 16, 42).unwrap();
        let v = vec![0.0_f32; 4];
        assert!(matches!(
            q.sketch(&v),
            Err(TurboQuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_non_finite_error() {
        let q = QjlQuantizer::new(4, 8, 42).unwrap();
        let v = vec![1.0_f32, f32::INFINITY, 0.0, 0.0];
        assert!(matches!(
            q.sketch(&v),
            Err(TurboQuantError::NonFiniteInput { .. })
        ));
    }

    #[test]
    fn test_inner_product_self_positive() {
        // Inner product of a vector with itself should be positive
        let q = QjlQuantizer::new(64, 128, 42).unwrap();
        let v: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let exact: f32 = v.iter().map(|x| x * x).sum();
        let sketch = q.sketch(&v).unwrap();
        let est = q.inner_product_estimate(&sketch, &v).unwrap();
        assert!(
            est > 0.0,
            "Self IP should be positive: estimate={}, exact={}",
            est,
            exact
        );
    }

    #[test]
    fn test_inner_product_unbiased() {
        // Over many random trials, the mean estimation error should be ~0
        let dim = 64;
        let q = QjlQuantizer::new(dim, 128, 42).unwrap();

        let mut total_error = 0.0f64;
        let trials = 200;

        for i in 0..trials {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.7).sin())
                .collect();
            let query: Vec<f32> = (0..dim)
                .map(|j| ((i * dim + j) as f32 * 1.3).cos())
                .collect();

            let exact: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            let sketch = q.sketch(&v).unwrap();
            let estimated = q.inner_product_estimate(&sketch, &query).unwrap();

            total_error += (estimated - exact) as f64;
        }

        let mean_error = total_error / trials as f64;
        assert!(
            mean_error.abs() < 2.0,
            "QJL should be approximately unbiased, mean error = {:.4}",
            mean_error
        );
    }

    #[test]
    fn test_zero_vector() {
        let q = QjlQuantizer::new(8, 16, 42).unwrap();
        let v = vec![0.0_f32; 8];
        let sketch = q.sketch(&v).unwrap();
        assert!(sketch.norm < 1e-10);

        let query = vec![1.0_f32; 8];
        let est = q.inner_product_estimate(&sketch, &query).unwrap();
        assert!(est.abs() < 1e-10, "Zero vector IP should be ~0, got {}", est);
    }

    #[test]
    fn test_deterministic() {
        let q = QjlQuantizer::new(16, 32, 99).unwrap();
        let v = vec![1.0_f32; 16];
        let s1 = q.sketch(&v).unwrap();
        let s2 = q.sketch(&v).unwrap();
        assert_eq!(s1.signs, s2.signs);
        assert_eq!(s1.norm, s2.norm);
    }

    #[test]
    fn test_different_seeds() {
        let q1 = QjlQuantizer::new(16, 32, 1).unwrap();
        let q2 = QjlQuantizer::new(16, 32, 2).unwrap();
        let v = vec![1.0_f32; 16];
        let s1 = q1.sketch(&v).unwrap();
        let s2 = q2.sketch(&v).unwrap();
        // Different seeds should produce different sketches (with high probability)
        assert_ne!(s1.signs, s2.signs);
    }

    #[test]
    fn test_sketch_size_bytes() {
        let q = QjlQuantizer::new(1536, 384, 42).unwrap();
        let v = vec![0.1_f32; 1536];
        let sketch = q.sketch(&v).unwrap();
        // 384 projections → 48 bytes signs + 2 bytes projections + 4 bytes norm = 54
        assert_eq!(sketch.size_bytes(), 54);
    }

    #[test]
    fn test_matrix_size_bytes() {
        let q = QjlQuantizer::new(1536, 384, 42).unwrap();
        // 384 * 1536 * 4 = 2,359,296 bytes ≈ 2.4 MB
        assert_eq!(q.matrix_size_bytes(), 384 * 1536 * 4);
    }

    // -----------------------------------------------------------------------
    // Compact serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_qjl_sketch_roundtrip() {
        let q = QjlQuantizer::new(16, 32, 42).unwrap();
        let v: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).sin()).collect();
        let sketch = q.sketch(&v).unwrap();
        let bytes = sketch.to_compact_bytes();
        let decoded = QjlSketch::from_compact_bytes(&bytes).unwrap();
        assert_eq!(decoded.signs, sketch.signs);
        assert_eq!(decoded.num_projections, sketch.num_projections);
        assert_eq!(decoded.norm, sketch.norm);
    }

    #[test]
    fn test_qjl_sketch_odd_projections_roundtrip() {
        // 13 projections → 2 sign bytes
        let q = QjlQuantizer::new(8, 13, 99).unwrap();
        let v = vec![1.0_f32; 8];
        let sketch = q.sketch(&v).unwrap();
        let bytes = sketch.to_compact_bytes();
        let back = QjlSketch::from_compact_bytes(&bytes).unwrap();
        assert_eq!(back.num_projections, 13);
        assert_eq!(back.signs.len(), 2); // ceil(13/8) = 2
    }

    #[test]
    fn test_qjl_sketch_wrong_version() {
        let q = QjlQuantizer::new(8, 16, 42).unwrap();
        let v = vec![0.5_f32; 8];
        let sketch = q.sketch(&v).unwrap();
        let mut bytes = sketch.to_compact_bytes();
        bytes[0] = 0xBB;
        assert!(matches!(
            QjlSketch::from_compact_bytes(&bytes),
            Err(TurboQuantError::DeserializationError { .. })
        ));
    }

    #[test]
    fn test_qjl_sketch_truncated() {
        let q = QjlQuantizer::new(8, 16, 42).unwrap();
        let v = vec![0.5_f32; 8];
        let sketch = q.sketch(&v).unwrap();
        let bytes = sketch.to_compact_bytes();
        // 7 bytes is header-only but sign bytes missing
        let truncated = &bytes[..7];
        assert!(matches!(
            QjlSketch::from_compact_bytes(truncated),
            Err(TurboQuantError::DeserializationError { .. })
        ));
    }

    #[test]
    fn test_qjl_sketch_empty_buffer() {
        assert!(matches!(
            QjlSketch::from_compact_bytes(&[]),
            Err(TurboQuantError::DeserializationError { .. })
        ));
    }

    #[test]
    fn test_serializable_code_trait_qjl() {
        use crate::traits::SerializableCode;
        let q = QjlQuantizer::new(16, 32, 42).unwrap();
        let v = vec![0.4_f32; 16];
        let sketch = q.sketch(&v).unwrap();
        let bytes = <QjlSketch as SerializableCode>::to_compact_bytes(&sketch);
        let back = <QjlSketch as SerializableCode>::from_compact_bytes(&bytes).unwrap();
        assert_eq!(back.norm, sketch.norm);
    }
}
