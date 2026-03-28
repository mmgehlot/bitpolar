//! Core traits for ecosystem integration.
//!
//! These traits define the public contract that all quantizers implement.
//! Consumers (vector databases, inference engines) program against these
//! traits rather than concrete types, enabling pluggable quantization.
//!
//! # Trait Hierarchy
//!
//! ```text
//! VectorQuantizer          — core encode/decode/estimate interface
//!   └── BatchQuantizer     — parallel batch operations (rayon feature)
//!
//! RotationStrategy         — pluggable rotation (QR, WHT, identity)
//! SerializableCode         — compact binary serialization (serde feature)
//! ```

use crate::error::Result;

/// Core trait for any vector quantizer that produces compressed codes.
///
/// All quantizers in this crate implement this trait, providing a uniform
/// interface for vector databases and inference engines to consume.
///
/// # Thread Safety
///
/// All implementations are `Send + Sync` — quantizers are immutable after
/// construction and safe to share across threads via `Arc<dyn VectorQuantizer>`.
///
/// # Example
///
/// ```rust
/// use bitpolar::traits::VectorQuantizer;
/// use bitpolar::TurboQuantizer;
///
/// fn search<Q: VectorQuantizer>(quantizer: &Q, codes: &[Q::Code], query: &[f32]) -> Vec<f32> {
///     codes.iter()
///         .map(|code| quantizer.inner_product_estimate(code, query).unwrap_or(f32::MIN))
///         .collect()
/// }
/// ```
pub trait VectorQuantizer: Send + Sync {
    /// The compressed representation type produced by this quantizer.
    type Code: Sized + Send + Sync + Clone;

    /// Encode a full-precision vector into a compressed code.
    ///
    /// # Arguments
    /// - `vector` — f32 slice of length `self.dim()`
    ///
    /// # Errors
    /// - `DimensionMismatch` if `vector.len() != self.dim()`
    /// - `NonFiniteInput` if any element is NaN or Inf
    fn encode(&self, vector: &[f32]) -> Result<Self::Code>;

    /// Decode a compressed code back to an approximate f32 vector.
    ///
    /// The reconstructed vector is an approximation — quantization is lossy.
    /// Reconstruction quality depends on the bit width and algorithm.
    fn decode(&self, code: &Self::Code) -> Vec<f32>;

    /// Estimate the inner product `<original_vector, query>` from a compressed code.
    ///
    /// This is the primary operation for similarity search. The estimate is
    /// **provably unbiased** for TurboQuant at 3+ bits:
    /// `E[estimate] = <original_vector, query>`.
    ///
    /// # Arguments
    /// - `code` — compressed representation of the stored vector
    /// - `query` — full-precision query vector of length `self.dim()`
    fn inner_product_estimate(&self, code: &Self::Code, query: &[f32]) -> Result<f32>;

    /// Estimate the L2 (Euclidean) distance between the original vector and a query.
    ///
    /// # Arguments
    /// - `code` — compressed representation of the stored vector
    /// - `query` — full-precision query vector of length `self.dim()`
    fn l2_distance_estimate(&self, code: &Self::Code, query: &[f32]) -> Result<f32>;

    /// The vector dimension this quantizer was created for.
    fn dim(&self) -> usize;

    /// Approximate size of a single compressed code in bytes.
    fn code_size_bytes(&self, code: &Self::Code) -> usize;
}

/// Trait for batch operations on quantized vectors.
///
/// Implementations use parallel processing (rayon) when the `parallel`
/// feature is enabled, falling back to sequential processing otherwise.
///
/// Batch operations amortize per-call overhead and are essential for
/// high-throughput Python/FFI bindings where per-vector FFI calls are expensive.
#[cfg(feature = "parallel")]
pub trait BatchQuantizer: VectorQuantizer {
    /// Encode multiple vectors in parallel.
    ///
    /// Returns one code per input vector. All vectors must have
    /// length `self.dim()`.
    fn batch_encode(&self, vectors: &[&[f32]]) -> Result<Vec<Self::Code>>;

    /// Estimate inner products between multiple codes and a single query.
    ///
    /// Returns one score per code. The query must have length `self.dim()`.
    fn batch_inner_product(&self, codes: &[Self::Code], query: &[f32]) -> Result<Vec<f32>>;

    /// Decode multiple codes in parallel.
    ///
    /// Returns one reconstructed vector per code.
    fn batch_decode(&self, codes: &[Self::Code]) -> Vec<Vec<f32>>;
}

/// Trait for pluggable rotation strategies.
///
/// The default implementation uses Haar-distributed QR rotation (O(d^2)).
/// Alternative strategies include:
/// - **Walsh-Hadamard Transform (WHT):** O(d log d), used by llama.cpp
/// - **Identity rotation:** O(1), for testing or pre-whitened data
///
/// # Example
///
/// ```rust,ignore
/// struct WalshHadamardRotation { dim: usize }
///
/// impl RotationStrategy for WalshHadamardRotation {
///     fn rotate(&self, vector: &[f32]) -> Vec<f32> {
///         // O(d log d) butterfly implementation
///         todo!()
///     }
///     fn rotate_inverse(&self, vector: &[f32]) -> Vec<f32> {
///         // WHT is its own inverse (up to scaling)
///         todo!()
///     }
///     fn dim(&self) -> usize { self.dim }
/// }
/// ```
pub trait RotationStrategy: Send + Sync {
    /// Apply the rotation to a vector, returning a new vector.
    fn rotate(&self, vector: &[f32]) -> Vec<f32>;

    /// Apply the inverse rotation (transpose for orthogonal matrices).
    fn rotate_inverse(&self, vector: &[f32]) -> Vec<f32>;

    /// The dimension this rotation operates on.
    fn dim(&self) -> usize;
}

/// Trait for compact binary serialization of compressed codes.
///
/// JSON serialization (via serde) is available for debugging and interchange,
/// but compact binary is 3-10x smaller and essential for database storage.
///
/// All compact binary formats include a 1-byte version header for forward
/// compatibility.
pub trait SerializableCode: Sized {
    /// Serialize to a compact binary representation.
    ///
    /// The format is:
    /// ```text
    /// [version: u8][type-specific payload...]
    /// ```
    fn to_compact_bytes(&self) -> Vec<u8>;

    /// Deserialize from compact binary bytes.
    ///
    /// # Errors
    /// - `DeserializationError` if the buffer is too short, has an
    ///   unrecognized version, or contains inconsistent lengths.
    fn from_compact_bytes(bytes: &[u8]) -> Result<Self>;
}
