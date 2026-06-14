//! PolarQuant: polar coordinate vector encoding (Stage 1 of TurboQuant).
//!
//! Groups the `d`-dimensional rotated vector into `d/2` pairs, converts each
//! pair `(x, y)` to polar coordinates `(r, θ)`, and quantizes angles to `bits`
//! bits using a Lloyd-Max codebook. In the compact serialization (v0x02) radii
//! are quantized to a per-vector scale + `bits` bits and angle indices are
//! bit-packed to `bits` bits (the in-memory `PolarCode` keeps f32 radii).

use crate::codebook::LloydMaxCodebook;
use crate::error::{validate_finite, Result, TurboQuantError};
use crate::traits::{SerializableCode, VectorQuantizer};

// ---------------------------------------------------------------------------
// PolarCode
// ---------------------------------------------------------------------------

/// Compressed representation produced by [`PolarQuantizer`].
///
/// Holds f32 radii and quantized angle indices in memory. NOTE: the compact
/// serialization (v0x02) quantizes radii lossily (per-vector scale + `bits`
/// bits); only the in-memory form here keeps full-precision radii.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct PolarCode {
    /// Radii for each coordinate pair (length = dim/2). Full-precision in memory;
    /// quantized in the compact serialization.
    pub(crate) radii: Vec<f32>,
    /// Quantized angle indices (length = dim/2)
    pub(crate) angle_indices: Vec<u16>,
    /// Bit width used to quantize angles
    pub(crate) bits: u8,
    /// Bit width used to quantize radii in the compact format (defaults to `bits`).
    pub(crate) radii_bits: u8,
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
    /// Bit width for radii (magnitude) quantization in the compact format.
    /// Independent of angle `bits`: magnitude and direction precision can be
    /// tuned separately. Defaults to `bits`.
    radii_bits: u8,
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
            return Err(TurboQuantError::DimensionTooLarge(
                dim,
                u16::MAX as usize * 2,
            ));
        }
        let codebook = LloydMaxCodebook::compute(bits)?;
        Ok(Self {
            dim,
            bits,
            radii_bits: bits,
            codebook,
        })
    }

    /// Set the radii (magnitude) quantization bit width independently of the
    /// angle `bits`. Magnitude precision drives reconstruction magnitude error;
    /// direction (angle) precision drives cosine. Tuning them separately lets a
    /// caller trade size for quality more finely (e.g. fewer angle bits but more
    /// radii bits, or vice versa). `n` must be in 1..=16.
    pub fn with_radii_bits(mut self, n: u8) -> Result<Self> {
        if n == 0 || n > 16 {
            return Err(TurboQuantError::InvalidBitWidth(n));
        }
        self.radii_bits = n;
        Ok(self)
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
            // x*x + y*y can overflow to +inf for large-but-finite components
            // (|x| > ~1.8e19), which passed validate_finite. sqrt(inf)=inf would
            // then propagate NaN through radii scaling/decode. Clamp to the max
            // finite radius so reconstruction stays finite (monotonic, lossy).
            let r = crate::compat::math::sqrtf(x * x + y * y);
            let r = if r.is_finite() { r } else { f32::MAX };
            let theta = crate::compat::math::atan2f(y, x); // in [-π, π]
            radii.push(r);
            // Map θ from [-π, π] to a Gaussian-like value for the codebook.
            // We store θ directly and use the boundary-finding quantize step.
            // The codebook is calibrated for N(0,1); angle is ~ Uniform[-π,π].
            // We normalize to approximate Gaussian scale: divide by π/√3.
            let normalised = theta / core::f32::consts::PI; // in [-1, 1]
            angle_indices.push(self.codebook.quantize(normalised));
        }

        PolarCode {
            radii,
            angle_indices,
            bits: self.bits,
            radii_bits: self.radii_bits,
        }
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
        let sq: f32 = decoded
            .iter()
            .zip(query.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        Ok(crate::compat::math::sqrtf(sq))
    }
}

// ---------------------------------------------------------------------------
// Compact binary serialization for PolarCode
// ---------------------------------------------------------------------------

/// Pack `bits`-bit values (LSB-first) into a byte buffer. Values are masked to
/// `bits` bits. `bits` must be in 1..=16.
fn pack_bits(values: &[u16], bits: u8) -> Vec<u8> {
    let bits = bits as usize;
    let mut out = vec![0u8; (values.len() * bits).div_ceil(8)];
    let mask: u32 = (1u32 << bits) - 1;
    let mut bitpos = 0usize;
    for &v in values {
        let v = v as u32 & mask;
        for b in 0..bits {
            if (v >> b) & 1 == 1 {
                out[bitpos / 8] |= 1 << (bitpos % 8);
            }
            bitpos += 1;
        }
    }
    out
}

/// Inverse of [`pack_bits`]: read `count` values of `bits` bits each (LSB-first).
fn unpack_bits(bytes: &[u8], count: usize, bits: u8) -> Vec<u16> {
    let bits = bits as usize;
    let mut out = Vec::with_capacity(count);
    let mut bitpos = 0usize;
    for _ in 0..count {
        let mut v = 0u32;
        for b in 0..bits {
            let byte = bytes.get(bitpos / 8).copied().unwrap_or(0);
            v |= (((byte >> (bitpos % 8)) & 1) as u32) << b;
            bitpos += 1;
        }
        out.push(v as u16);
    }
    out
}

impl PolarCode {
    /// Serialize to compact binary.
    ///
    /// Format (v0x03):
    /// ```text
    /// [version: u8 = 0x03][angle_bits: u8][radii_bits: u8][num_pairs: u16 LE]
    /// [radii_scale: f32 LE]
    /// [radii_q: num_pairs × `radii_bits` bits][angle_indices: num_pairs × `angle_bits` bits]
    /// ```
    /// Radii are quantized to `radii_bits` bits relative to the per-vector max
    /// radius (`radii_scale`); angle indices are bit-packed to `angle_bits` bits
    /// (already in `[0, 2^angle_bits)` from the codebook). Magnitude and direction
    /// precision are tuned independently. (v0x01 stored radii as lossless f32 and
    /// angles as u16 — size invariant to bits, defeating compression; v0x02 tied
    /// radii and angle widths together.)
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let num_pairs: u16 =
            self.radii.len().try_into().expect(
                "PolarCode num_pairs exceeds u16::MAX; dimension too large for compact format",
            );
        let angle_bits = self.bits;
        let radii_bits = self.radii_bits;
        let max_q = ((1u32 << radii_bits) - 1) as f32;
        // Radii are magnitudes (>= 0); scale by the per-vector max.
        let scale = self.radii.iter().copied().fold(0.0f32, f32::max);
        let radii_q: Vec<u16> = self
            .radii
            .iter()
            .map(|&r| {
                if scale > 0.0 {
                    (r / scale * max_q).round().clamp(0.0, max_q) as u16
                } else {
                    0
                }
            })
            .collect();

        let radii_packed = pack_bits(&radii_q, radii_bits);
        let angles_packed = pack_bits(&self.angle_indices, angle_bits);

        let mut out = Vec::with_capacity(9 + radii_packed.len() + angles_packed.len());
        out.push(crate::COMPACT_FORMAT_VERSION);
        out.push(angle_bits);
        out.push(radii_bits);
        out.extend_from_slice(&num_pairs.to_le_bytes());
        out.extend_from_slice(&scale.to_le_bytes());
        out.extend_from_slice(&radii_packed);
        out.extend_from_slice(&angles_packed);
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

        // header: version(1) + angle_bits(1) + radii_bits(1) + num_pairs(2) + scale(4) = 9
        if bytes.len() < 9 {
            return Err(err("buffer too short: need at least 9 bytes"));
        }
        if bytes[0] != crate::COMPACT_FORMAT_VERSION {
            return Err(err(&format!(
                "unsupported version 0x{:02X}, expected 0x{:02X}",
                bytes[0],
                crate::COMPACT_FORMAT_VERSION
            )));
        }
        let angle_bits = bytes[1];
        let radii_bits = bytes[2];
        if angle_bits == 0 || angle_bits > 16 {
            return Err(err(&format!(
                "invalid angle bit width {angle_bits}: must be 1..=16"
            )));
        }
        if radii_bits == 0 || radii_bits > 16 {
            return Err(err(&format!(
                "invalid radii bit width {radii_bits}: must be 1..=16"
            )));
        }
        let num_pairs = u16::from_le_bytes([bytes[3], bytes[4]]) as usize;
        let scale = f32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
        // Reject a hostile/corrupt scale: a non-finite or negative scale would
        // produce NaN/inf radii on dequantization and silently corrupt decode.
        if !scale.is_finite() || scale < 0.0 {
            return Err(err("radii_scale is not finite or is negative"));
        }

        let radii_packed_len = (num_pairs * radii_bits as usize).div_ceil(8);
        let angles_packed_len = (num_pairs * angle_bits as usize).div_ceil(8);
        let expected_len = 9 + radii_packed_len + angles_packed_len;
        if bytes.len() < expected_len {
            return Err(err(&format!(
                "buffer too short: need {expected_len}, got {}",
                bytes.len()
            )));
        }

        let radii_packed = &bytes[9..9 + radii_packed_len];
        let angles_packed = &bytes[9 + radii_packed_len..9 + radii_packed_len + angles_packed_len];

        let max_q = ((1u32 << radii_bits) - 1) as f32;
        let radii: Vec<f32> = unpack_bits(radii_packed, num_pairs, radii_bits)
            .into_iter()
            .map(|q| {
                if scale > 0.0 {
                    q as f32 / max_q * scale
                } else {
                    0.0
                }
            })
            .collect();
        let angle_indices = unpack_bits(angles_packed, num_pairs, angle_bits);

        Ok(Self {
            radii,
            angle_indices,
            bits: angle_bits,
            radii_bits,
        })
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
        assert!(matches!(
            PolarQuantizer::new(0, 4),
            Err(TurboQuantError::ZeroDimension)
        ));
    }

    #[test]
    fn test_odd_dimension_error() {
        assert!(matches!(
            PolarQuantizer::new(3, 4),
            Err(TurboQuantError::OddDimension(3))
        ));
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
        assert!(matches!(
            q.encode(&v),
            Err(TurboQuantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_non_finite_error() {
        let q = PolarQuantizer::new(4, 4).unwrap();
        let v = vec![1.0_f32, f32::NAN, 0.0, 0.0];
        assert!(matches!(
            q.encode(&v),
            Err(TurboQuantError::NonFiniteInput { .. })
        ));
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
        let bits = 4u8;
        let q = PolarQuantizer::new(8, bits).unwrap();
        let v: Vec<f32> = (0..8).map(|i| i as f32 * 0.25 - 1.0).collect();
        let code = q.encode(&v).unwrap();
        let bytes = code.to_compact_bytes();
        let decoded = PolarCode::from_compact_bytes(&bytes).unwrap();
        assert_eq!(decoded.bits, code.bits);
        // Angle indices are bit-packed losslessly (already in [0, 2^bits)).
        assert_eq!(decoded.angle_indices, code.angle_indices);
        // Radii are quantized to `bits` bits relative to the per-vector max, so
        // round-trip is lossy within one quantization step.
        let scale = code.radii.iter().copied().fold(0.0f32, f32::max);
        let step = scale / ((1u32 << bits) - 1) as f32;
        assert_eq!(decoded.radii.len(), code.radii.len());
        for (d, o) in decoded.radii.iter().zip(code.radii.iter()) {
            assert!(
                (d - o).abs() <= step + 1e-6,
                "radius {d} vs {o} exceeds quantization step {step}"
            );
        }
    }

    #[test]
    fn test_encode_huge_finite_components_no_nan() {
        // Components are finite but x*x overflows f32 -> r would be inf -> NaN on
        // decode. Encoder must clamp so decode stays finite.
        let q = PolarQuantizer::new(4, 4).unwrap();
        let v = vec![3.0e38_f32, 3.0e38, 0.0, 0.0];
        let code = q.encode(&v).unwrap();
        for r in &code.radii {
            assert!(r.is_finite(), "radius must be finite, got {r}");
        }
        let bytes = code.to_compact_bytes();
        let back = PolarCode::from_compact_bytes(&bytes).unwrap();
        for x in q.decode(&back) {
            assert!(x.is_finite(), "decoded value must be finite");
        }
    }

    #[test]
    fn test_from_compact_bytes_rejects_nonfinite_scale() {
        let q = PolarQuantizer::new(4, 4).unwrap();
        let code = q.encode(&[0.5_f32, -0.3, 0.1, 0.2]).unwrap();
        let mut bytes = code.to_compact_bytes();
        // Overwrite radii_scale with +inf. Header (v0x03): version(1) angle_bits(1)
        // radii_bits(1) num_pairs(2) -> scale at offset 5..9.
        bytes[5..9].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(PolarCode::from_compact_bytes(&bytes).is_err());
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
                .map(|i| {
                    (0..dim)
                        .map(|j| ((i * dim + j) as f32 * 0.1).cos())
                        .collect()
                })
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
            assert!(q
                .batch_inner_product(&[], &[0.0_f32; 8])
                .unwrap()
                .is_empty());
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
