//! PolarQuant: polar coordinate vector encoding (Stage 1 of TurboQuant).
//!
//! Groups the `d`-dimensional rotated vector into `d/2` pairs, converts each
//! pair `(x, y)` to polar coordinates `(r, θ)`, stores radii losslessly as
//! f32, and quantizes angles to `bits` bits using a Lloyd-Max codebook.

use crate::codebook::LloydMaxCodebook;
use crate::error::{validate_finite, Result, TurboQuantError};
use crate::traits::{SerializableCode, VectorQuantizer};

// ---------------------------------------------------------------------------
// PolarCode
// ---------------------------------------------------------------------------

/// Compressed representation produced by [`PolarQuantizer`].
///
/// Contains lossless f32 radii and quantized angle indices.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(serde::Serialize, serde::Deserialize))]
pub struct PolarCode {
    /// Lossless radii for each coordinate pair (length = dim/2)
    pub(crate) radii: Vec<f32>,
    /// Quantized angle indices (length = dim/2)
    pub(crate) angle_indices: Vec<u16>,
    /// Bit width used to quantize angles
    pub(crate) bits: u8,
}

impl PolarCode {
    /// Approximate heap size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.radii.len() * core::mem::size_of::<f32>()
            + self.angle_indices.len() * core::mem::size_of::<u16>()
            + 1 // bits
    }
}

// ---------------------------------------------------------------------------
// PolarQuantizer
// ---------------------------------------------------------------------------

/// PolarQuant vector quantizer.
///
/// Encodes a `dim`-dimensional f32 vector into polar coordinates:
/// - Radii stored losslessly (exact f32)
/// - Angles quantized to `bits` bits
///
/// # Example
///
/// ```rust
/// use bitpolar::polar::PolarQuantizer;
/// use bitpolar::traits::VectorQuantizer;
///
/// let q = PolarQuantizer::new(8, 4).unwrap();
/// let v = vec![0.1_f32; 8];
/// let code = q.encode(&v).unwrap();
/// let approx = q.decode(&code);
/// assert_eq!(approx.len(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct PolarQuantizer {
    dim: usize,
    bits: u8,
    codebook: LloydMaxCodebook,
}

impl PolarQuantizer {
    /// Create a new PolarQuantizer.
    ///
    /// # Arguments
    /// - `dim` — must be even and > 0
    /// - `bits` — angle quantization bit width (1..=16)
    ///
    /// # Errors
    /// - `ZeroDimension`, `OddDimension`, `InvalidBitWidth`
    pub fn new(dim: usize, bits: u8) -> Result<Self> {
        if dim == 0 {
            return Err(TurboQuantError::ZeroDimension);
        }
        if dim % 2 != 0 {
            return Err(TurboQuantError::OddDimension(dim));
        }
        // num_pairs = dim/2 is stored as u16 in compact format; reject oversized dims.
        if dim / 2 > u16::MAX as usize {
            return Err(TurboQuantError::DimensionTooLarge(dim, u16::MAX as usize * 2));
        }
        let codebook = LloydMaxCodebook::compute(bits)?;
        Ok(Self { dim, bits, codebook })
    }

    /// The dimension this quantizer was created for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The angle quantization bit width.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    // -----------------------------------------------------------------------
    // Core encode / decode helpers
    // -----------------------------------------------------------------------

    #[inline]
    fn encode_inner(&self, vector: &[f32]) -> PolarCode {
        let pairs = self.dim / 2;
        let mut radii = Vec::with_capacity(pairs);
        let mut angle_indices = Vec::with_capacity(pairs);

        for i in 0..pairs {
            let x = vector[2 * i];
            let y = vector[2 * i + 1];
            let r = crate::compat::math::sqrtf(x * x + y * y);
            let theta = crate::compat::math::atan2f(y, x); // in [-π, π]
            radii.push(r);
            // Map θ from [-π, π] to a Gaussian-like value for the codebook.
            // We store θ directly and use the boundary-finding quantize step.
            // The codebook is calibrated for N(0,1); angle is ~ Uniform[-π,π].
            // We normalize to approximate Gaussian scale: divide by π/√3.
            let normalised = theta / core::f32::consts::PI; // in [-1, 1]
            angle_indices.push(self.codebook.quantize(normalised));
        }

        PolarCode { radii, angle_indices, bits: self.bits }
    }

    #[inline]
    fn decode_inner(&self, code: &PolarCode) -> Vec<f32> {
        let pairs = self.dim / 2;
        let mut out = vec![0.0_f32; self.dim];

        for i in 0..pairs.min(code.radii.len()).min(code.angle_indices.len()) {
            let r = code.radii[i];
            let normalised = self.codebook.dequantize(code.angle_indices[i]);
            let theta = normalised * core::f32::consts::PI;
            out[2 * i] = r * crate::compat::math::cosf(theta);
            out[2 * i + 1] = r * crate::compat::math::sinf(theta);
        }

        out
    }

    // -----------------------------------------------------------------------
    // Estimation helpers
    // -----------------------------------------------------------------------

    /// Estimate inner product `<original_vector, query>` from a PolarCode.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::polar::ip_estimate",
            skip(self, code, query),
            fields(dim = self.dim, bits = self.bits)
        )
    )]
    pub fn inner_product_estimate(&self, code: &PolarCode, query: &[f32]) -> Result<f32> {
        if query.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        validate_finite(query)?;
        let decoded = self.decode_inner(code);
        let ip = decoded.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
        Ok(ip)
    }

    /// Estimate L2 distance `||original_vector - query||` from a PolarCode.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::polar::l2_estimate",
            skip(self, code, query),
            fields(dim = self.dim, bits = self.bits)
        )
    )]
    pub fn l2_distance_estimate(&self, code: &PolarCode, query: &[f32]) -> Result<f32> {
        if query.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        validate_finite(query)?;
        let decoded = self.decode_inner(code);
        let sq: f32 = decoded.iter().zip(query.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        Ok(crate::compat::math::sqrtf(sq))
    }
}

// ---------------------------------------------------------------------------
// Compact binary serialization for PolarCode
// ---------------------------------------------------------------------------

impl PolarCode {
    /// Serialize to compact binary.
    ///
    /// Format:
    /// ```text
    /// [version: u8 = 0x01][bits: u8][num_pairs: u16 LE]
    /// [radii: num_pairs × f32 LE][angle_indices: num_pairs × u16 LE]
    /// ```
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let num_pairs: u16 = self.radii.len().try_into().expect(
            "PolarCode num_pairs exceeds u16::MAX; dimension too large for compact format",
        );
        let mut out = Vec::with_capacity(
            1 + 1 + 2 + self.radii.len() * 4 + self.angle_indices.len() * 2,
        );
        out.push(crate::COMPACT_FORMAT_VERSION);
        out.push(self.bits);
        out.extend_from_slice(&num_pairs.to_le_bytes());
        for &r in &self.radii {
            out.extend_from_slice(&r.to_le_bytes());
        }
        for &a in &self.angle_indices {
            out.extend_from_slice(&a.to_le_bytes());
        }
        out
    }

    /// Deserialize from compact binary bytes produced by [`to_compact_bytes`](Self::to_compact_bytes).
    ///
    /// # Errors
    /// - `DeserializationError` if the buffer is too short, version is wrong,
    ///   or lengths are inconsistent.
    pub fn from_compact_bytes(bytes: &[u8]) -> Result<Self> {
        let err = |reason: &str| TurboQuantError::DeserializationError {
            reason: reason.to_string(),
        };

        if bytes.len() < 4 {
            return Err(err("buffer too short: need at least 4 bytes"));
        }
        if bytes[0] != crate::COMPACT_FORMAT_VERSION {
            return Err(err(&format!(
                "unsupported version 0x{:02X}, expected 0x{:02X}",
                bytes[0],
                crate::COMPACT_FORMAT_VERSION
            )));
        }
        let bits = bytes[1];
        if bits == 0 || bits > 16 {
            return Err(err(&format!(
                "invalid bit width {bits}: must be 1..=16"
            )));
        }
        let num_pairs = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;

        // 4 bytes header + num_pairs * f32 (4 bytes) + num_pairs * u16 (2 bytes)
        let payload_size = num_pairs
            .checked_mul(4)
            .and_then(|r| r.checked_add(num_pairs.checked_mul(2)?))
            .ok_or_else(|| err("num_pairs causes size overflow"))?;
        let expected_len = 4 + payload_size;
        if bytes.len() < expected_len {
            return Err(err(&format!(
                "buffer too short: need {expected_len}, got {}",
                bytes.len()
            )));
        }

        let mut radii = Vec::with_capacity(num_pairs);
        let radii_start = 4;
        for i in 0..num_pairs {
            let off = radii_start + i * 4;
            let r = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            radii.push(r);
        }

        let angles_start = radii_start + num_pairs * 4;
        let mut angle_indices = Vec::with_capacity(num_pairs);
        for i in 0..num_pairs {
            let off = angles_start + i * 2;
            let a = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            angle_indices.push(a);
        }

        Ok(Self { radii, angle_indices, bits })
    }
}

impl SerializableCode for PolarCode {
    #[inline]
    fn to_compact_bytes(&self) -> Vec<u8> {
        PolarCode::to_compact_bytes(self)
    }

    #[inline]
    fn from_compact_bytes(bytes: &[u8]) -> Result<Self> {
        PolarCode::from_compact_bytes(bytes)
    }
}

// ---------------------------------------------------------------------------
// BatchQuantizer impl (parallel feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
impl crate::traits::BatchQuantizer for PolarQuantizer {
    /// Encode multiple vectors in parallel using rayon.
    ///
    /// All vectors must have length `self.dim()`. If any vector fails
    /// validation the first error encountered is returned.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::polar::batch_encode",
            skip(self, vectors),
            fields(dim = self.dim, bits = self.bits, batch_size = vectors.len())
        )
    )]
    fn batch_encode(&self, vectors: &[&[f32]]) -> Result<Vec<Self::Code>> {
        use rayon::prelude::*;
        vectors
            .par_iter()
            .map(|v| self.encode(v))
            .collect::<Result<Vec<_>>>()
    }

    /// Estimate inner products between multiple codes and a single query in parallel.
    ///
    /// Returns one score per code in the same order.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::polar::batch_inner_product",
            skip(self, codes, query),
            fields(dim = self.dim, bits = self.bits, batch_size = codes.len())
        )
    )]
    fn batch_inner_product(&self, codes: &[Self::Code], query: &[f32]) -> Result<Vec<f32>> {
        use rayon::prelude::*;
        codes
            .par_iter()
            .map(|c| self.inner_product_estimate(c, query))
            .collect::<Result<Vec<_>>>()
    }

    /// Decode multiple codes in parallel.
    ///
    /// Returns one reconstructed f32 vector per code in the same order.
    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::polar::batch_decode",
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

impl VectorQuantizer for PolarQuantizer {
    type Code = PolarCode;

    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::polar::encode",
            skip(self, vector),
            fields(dim = self.dim, bits = self.bits)
        )
    )]
    fn encode(&self, vector: &[f32]) -> Result<Self::Code> {
        if vector.len() != self.dim {
            return Err(TurboQuantError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        validate_finite(vector)?;
        Ok(self.encode_inner(vector))
    }

    #[cfg_attr(
        feature = "tracing-support",
        tracing::instrument(
            name = "bitpolar::polar::decode",
            skip(self, code),
            fields(dim = self.dim, bits = self.bits)
        )
    )]
    fn decode(&self, code: &Self::Code) -> Vec<f32> {
        self.decode_inner(code)
    }

    fn inner_product_estimate(&self, code: &Self::Code, query: &[f32]) -> Result<f32> {
        self.inner_product_estimate(code, query)
    }

    fn l2_distance_estimate(&self, code: &Self::Code, query: &[f32]) -> Result<f32> {
        self.l2_distance_estimate(code, query)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn code_size_bytes(&self, code: &Self::Code) -> usize {
        code.size_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_dimension_error() {
        assert!(matches!(PolarQuantizer::new(0, 4), Err(TurboQuantError::ZeroDimension)));
    }

    #[test]
    fn test_odd_dimension_error() {
        assert!(matches!(PolarQuantizer::new(3, 4), Err(TurboQuantError::OddDimension(3))));
    }

    #[test]
    fn test_encode_decode_shape() {
        let q = PolarQuantizer::new(8, 4).unwrap();
        let v: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let code = q.encode(&v).unwrap();
        assert_eq!(code.radii.len(), 4);
        assert_eq!(code.angle_indices.len(), 4);
        let decoded = q.decode(&code);
        assert_eq!(decoded.len(), 8);
    }

    #[test]
    fn test_dimension_mismatch() {
        let q = PolarQuantizer::new(8, 4).unwrap();
        let v = vec![0.0_f32; 4];
        assert!(matches!(q.encode(&v), Err(TurboQuantError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_non_finite_error() {
        let q = PolarQuantizer::new(4, 4).unwrap();
        let v = vec![1.0_f32, f32::NAN, 0.0, 0.0];
        assert!(matches!(q.encode(&v), Err(TurboQuantError::NonFiniteInput { .. })));
    }

    #[test]
    fn test_zero_vector_encode_decode() {
        let q = PolarQuantizer::new(4, 4).unwrap();
        let v = vec![0.0_f32; 4];
        let code = q.encode(&v).unwrap();
        let decoded = q.decode(&code);
        assert_eq!(decoded.len(), 4);
        // All radii are 0, so decoded values should be ~0.
        for x in decoded {
            assert!(x.abs() < 1e-5);
        }
    }

    // -----------------------------------------------------------------------
    // Compact serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_polar_code_roundtrip() {
        let q = PolarQuantizer::new(8, 4).unwrap();
        let v: Vec<f32> = (0..8).map(|i| i as f32 * 0.25 - 1.0).collect();
        let code = q.encode(&v).unwrap();
        let bytes = code.to_compact_bytes();
        let decoded = PolarCode::from_compact_bytes(&bytes).unwrap();
        assert_eq!(decoded.bits, code.bits);
        assert_eq!(decoded.radii, code.radii);
        assert_eq!(decoded.angle_indices, code.angle_indices);
    }

    #[test]
    fn test_polar_code_roundtrip_empty_pairs() {
        // 2-dim → 1 pair
        let q = PolarQuantizer::new(2, 1).unwrap();
        let v = vec![1.0_f32, 0.0];
        let code = q.encode(&v).unwrap();
        let bytes = code.to_compact_bytes();
        let back = PolarCode::from_compact_bytes(&bytes).unwrap();
        assert_eq!(back.radii.len(), 1);
    }

    #[test]
    fn test_polar_code_wrong_version() {
        let q = PolarQuantizer::new(4, 4).unwrap();
        let v = vec![0.5_f32; 4];
        let code = q.encode(&v).unwrap();
        let mut bytes = code.to_compact_bytes();
        bytes[0] = 0xAB;
        assert!(matches!(
            PolarCode::from_compact_bytes(&bytes),
            Err(TurboQuantError::DeserializationError { .. })
        ));
    }

    #[test]
    fn test_polar_code_truncated() {
        let q = PolarQuantizer::new(8, 4).unwrap();
        let v = vec![0.1_f32; 8];
        let code = q.encode(&v).unwrap();
        let bytes = code.to_compact_bytes();
        // Truncate to 5 bytes (header OK but payload missing)
        let truncated = &bytes[..5];
        assert!(matches!(
            PolarCode::from_compact_bytes(truncated),
            Err(TurboQuantError::DeserializationError { .. })
        ));
    }

    #[test]
    fn test_polar_code_empty_buffer() {
        assert!(matches!(
            PolarCode::from_compact_bytes(&[]),
            Err(TurboQuantError::DeserializationError { .. })
        ));
    }

    #[test]
    fn test_serializable_code_trait() {
        use crate::traits::SerializableCode;
        let q = PolarQuantizer::new(8, 4).unwrap();
        let v = vec![0.5_f32; 8];
        let code = q.encode(&v).unwrap();
        let bytes = <PolarCode as SerializableCode>::to_compact_bytes(&code);
        let decoded = <PolarCode as SerializableCode>::from_compact_bytes(&bytes).unwrap();
        assert_eq!(decoded.bits, code.bits);
    }

    // -----------------------------------------------------------------------
    // Batch operation tests
    // -----------------------------------------------------------------------

    #[cfg(feature = "parallel")]
    mod batch_tests {
        use super::*;
        use crate::traits::BatchQuantizer;

        fn make_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
            (0..n)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).cos()).collect())
                .collect()
        }

        #[test]
        fn test_batch_encode_matches_sequential() {
            let q = PolarQuantizer::new(8, 4).unwrap();
            let vecs = make_vectors(8, 8);
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            let batch = q.batch_encode(&refs).unwrap();
            for (i, code) in batch.iter().enumerate() {
                let seq = q.encode(&vecs[i]).unwrap();
                assert_eq!(code.radii, seq.radii);
                assert_eq!(code.angle_indices, seq.angle_indices);
            }
        }

        #[test]
        fn test_batch_inner_product_matches_sequential() {
            let q = PolarQuantizer::new(8, 4).unwrap();
            let vecs = make_vectors(8, 8);
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            let codes = q.batch_encode(&refs).unwrap();
            let query = vec![0.2_f32; 8];
            let batch_scores = q.batch_inner_product(&codes, &query).unwrap();
            for (i, &score) in batch_scores.iter().enumerate() {
                let seq = q.inner_product_estimate(&codes[i], &query).unwrap();
                assert!((score - seq).abs() < 1e-6);
            }
        }

        #[test]
        fn test_batch_decode_matches_sequential() {
            let q = PolarQuantizer::new(8, 4).unwrap();
            let vecs = make_vectors(5, 8);
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            let codes = q.batch_encode(&refs).unwrap();
            let batch_dec = q.batch_decode(&codes);
            for (i, dec) in batch_dec.iter().enumerate() {
                let seq = q.decode(&codes[i]);
                for (a, b) in dec.iter().zip(seq.iter()) {
                    assert!((a - b).abs() < 1e-6);
                }
            }
        }

        #[test]
        fn test_batch_empty() {
            let q = PolarQuantizer::new(8, 4).unwrap();
            assert!(q.batch_encode(&[]).unwrap().is_empty());
            assert!(q.batch_inner_product(&[], &[0.0_f32; 8]).unwrap().is_empty());
            assert!(q.batch_decode(&[]).is_empty());
        }

        #[test]
        fn test_batch_single() {
            let q = PolarQuantizer::new(8, 4).unwrap();
            let v = vec![0.3_f32; 8];
            let codes = q.batch_encode(&[&v]).unwrap();
            assert_eq!(codes.len(), 1);
        }

        #[test]
        fn test_batch_encode_error_propagates() {
            let q = PolarQuantizer::new(8, 4).unwrap();
            let v_bad = vec![0.0_f32; 3]; // wrong dim
            assert!(q.batch_encode(&[&v_bad]).is_err());
        }
    }
}
