//! Error types for bitpolar.
//!
//! All public API methods return `Result<T, TurboQuantError>`. No panics
//! in the public API — all errors are recoverable by the caller.

use thiserror::Error;

/// Errors that can occur during TurboQuant operations.
///
/// Every variant includes enough context for the caller to diagnose
/// the issue without inspecting internal state.
#[derive(Debug, Clone, Error)]
pub enum TurboQuantError {
    /// Dimension must be even (required for polar coordinate pairing).
    ///
    /// TurboQuant groups coordinates into pairs for polar encoding.
    /// An odd dimension would leave one coordinate unpaired.
    #[error("dimension must be even, got {0}")]
    OddDimension(usize),

    /// Dimension is zero — cannot create a quantizer for zero-dimensional vectors.
    #[error("dimension must be > 0")]
    ZeroDimension,

    /// Bit width out of supported range.
    ///
    /// Supported range is 1..=16. At 1 bit, each angle gets 2 levels.
    /// At 16 bits, each angle gets 65536 levels (effectively lossless angles).
    #[error("bit width must be 1..=16, got {0}")]
    InvalidBitWidth(u8),

    /// Projection count must be positive for QJL sketching.
    #[error("projections must be > 0")]
    ZeroProjections,

    /// Dimension exceeds the maximum supported by the compact binary format.
    ///
    /// The compact serialization uses u16 for pair/projection counts,
    /// limiting dimension to 131070 (65535 pairs) and projections to 65535.
    #[error("dimension {0} too large for compact format (max {1})")]
    DimensionTooLarge(usize, usize),

    /// Input vector dimension does not match the quantizer's expected dimension.
    #[error("dimension mismatch: quantizer expects {expected}, got {actual}")]
    DimensionMismatch {
        /// The dimension the quantizer was created for
        expected: usize,
        /// The dimension of the input vector
        actual: usize,
    },

    /// Attempted operation on empty data.
    #[error("empty input: {0}")]
    EmptyInput(&'static str),

    /// Input vector contains NaN or infinity values.
    ///
    /// Quantization of non-finite values produces garbage output.
    /// The caller should sanitize input before encoding.
    #[error("input contains non-finite values (NaN or Inf) at index {index}")]
    NonFiniteInput {
        /// Index of the first non-finite value found
        index: usize,
    },

    /// Deserialization of compact binary data failed.
    ///
    /// The byte buffer is too short, has an unrecognized version,
    /// or contains internally inconsistent lengths.
    #[error("deserialization failed: {reason}")]
    DeserializationError {
        /// Human-readable description of what went wrong
        reason: String,
    },

    /// Index out of bounds (e.g., requesting a KV cache position that doesn't exist).
    #[error("index {index} out of bounds (length {length})")]
    IndexOutOfBounds {
        /// The requested index
        index: usize,
        /// The actual length of the collection
        length: usize,
    },
}

/// Result type alias using [`TurboQuantError`].
pub type Result<T> = core::result::Result<T, TurboQuantError>;

/// Validate that a vector contains only finite f32 values.
///
/// Returns `Ok(())` if all values are finite, or `Err(NonFiniteInput)`
/// with the index of the first non-finite value.
#[inline]
pub fn validate_finite(vector: &[f32]) -> Result<()> {
    for (i, &v) in vector.iter().enumerate() {
        if !v.is_finite() {
            return Err(TurboQuantError::NonFiniteInput { index: i });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_finite_ok() {
        let v = vec![1.0, -2.5, 0.0, 1e30, -1e-30];
        assert!(validate_finite(&v).is_ok());
    }

    #[test]
    fn test_validate_finite_nan() {
        let v = vec![1.0, 2.0, f32::NAN, 4.0];
        match validate_finite(&v) {
            Err(TurboQuantError::NonFiniteInput { index }) => assert_eq!(index, 2),
            other => panic!("expected NonFiniteInput, got {:?}", other),
        }
    }

    #[test]
    fn test_validate_finite_inf() {
        let v = vec![f32::INFINITY, 1.0];
        match validate_finite(&v) {
            Err(TurboQuantError::NonFiniteInput { index }) => assert_eq!(index, 0),
            other => panic!("expected NonFiniteInput, got {:?}", other),
        }
    }

    #[test]
    fn test_validate_finite_neg_inf() {
        let v = vec![1.0, f32::NEG_INFINITY];
        match validate_finite(&v) {
            Err(TurboQuantError::NonFiniteInput { index }) => assert_eq!(index, 1),
            other => panic!("expected NonFiniteInput, got {:?}", other),
        }
    }

    #[test]
    fn test_validate_finite_empty() {
        assert!(validate_finite(&[]).is_ok());
    }

    #[test]
    fn test_error_display() {
        let e = TurboQuantError::DimensionMismatch {
            expected: 128,
            actual: 256,
        };
        assert_eq!(
            e.to_string(),
            "dimension mismatch: quantizer expects 128, got 256"
        );
    }
}
