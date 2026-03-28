//! # bitpolar
//!
//! Zero-overhead vector quantization for semantic search and KV cache compression.
//!
//! Implements three algorithms from Google Research:
//! - **TurboQuant** (ICLR 2026) — two-stage composition for near-optimal compression
//! - **PolarQuant** (AISTATS 2026) — polar coordinate encoding with lossless radii
//! - **QJL** (AAAI 2025) — 1-bit Johnson-Lindenstrauss sketching
//!
//! ## Key Properties
//!
//! - **Data-oblivious**: No training, no codebooks, no calibration data
//! - **Deterministic**: Fully defined by 4 integers: `(dimension, bits, projections, seed)`
//! - **Provably unbiased**: Inner product estimates are unbiased at 3+ bits
//! - **Near-optimal**: Within ~2.7x of Shannon rate-distortion limit
//! - **Instant indexing**: Vectors compress on arrival — no offline training
//!
//! ## Quick Start
//!
//! ```rust
//! use bitpolar::TurboQuantizer;
//! use bitpolar::traits::VectorQuantizer;
//!
//! // Create quantizer from 4 integers — no training needed
//! let q = TurboQuantizer::new(128, 4, 32, 42).unwrap();
//!
//! // Encode a vector
//! let vector = vec![0.1_f32; 128];
//! let code = q.encode(&vector).unwrap();
//!
//! // Estimate inner product without decompression
//! let query = vec![0.05_f32; 128];
//! let score = q.inner_product_estimate(&code, &query).unwrap();
//! println!("Estimated IP: {score}");
//!
//! // Decode back to approximate vector
//! let reconstructed = q.decode(&code);
//! assert_eq!(reconstructed.len(), 128);
//! ```
//!
//! ## Architecture
//!
//! ```text
//! Input f32 vector
//!     │
//!     ▼
//! ┌─────────────────┐
//! │ Random Rotation  │  Haar-distributed orthogonal matrix (QR of Gaussian)
//! │ (StoredRotation) │  Spreads energy uniformly across coordinates
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │   PolarQuant     │  Groups d dims into d/2 pairs → polar coords
//! │  (Stage 1)       │  Radii: lossless f32 | Angles: b-bit quantized
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │   QJL Residual   │  Sketches reconstruction error
//! │  (Stage 2)       │  1 sign bit per projection → unbiased correction
//! └────────┬────────┘
//!          │
//!          ▼
//! TurboCode { polar: PolarCode, residual_sketch: QjlSketch }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | Yes | Standard library (nalgebra QR decomposition) |
//! | `serde-support` | Yes | Serde serialization for all types |
//! | `simd` | No | Hand-tuned NEON/AVX2 kernels |
//! | `parallel` | No | Parallel batch operations via rayon |
//! | `tracing-support` | No | OpenTelemetry-compatible instrumentation |
//! | `ffi` | No | C FFI exports |

#![warn(clippy::all)]
#![warn(missing_docs)]
// Forbid unsafe in the default code path — only allowed behind simd or ffi features
#![cfg_attr(not(any(feature = "simd", feature = "ffi")), forbid(unsafe_code))]
// Enable no_std when the `std` feature is not active.
// The `alloc` feature provides Vec/String without full std.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

// ============================================================================
// Compatibility layer (std/alloc/no_std)
// ============================================================================

/// Compatibility module for std/alloc/no_std switching.
/// All modules import Vec, math functions, etc. from here.
pub(crate) mod compat;

// ============================================================================
// Core modules (always available)
// ============================================================================

/// Error types — all public APIs return `Result<T, TurboQuantError>`
pub mod error;

/// Core traits for ecosystem integration: VectorQuantizer, BatchQuantizer, etc.
pub mod traits;

/// Compression statistics and quality metrics
pub mod stats;

/// Haar-distributed orthogonal rotation matrix (O(d²) memory)
pub mod rotation;

/// Walsh-Hadamard Transform rotation (O(d) memory, O(d log d) time)
pub mod wht;

/// Lloyd-Max optimal scalar quantizer for N(0,1) distribution
pub(crate) mod codebook;

/// PolarQuant: polar coordinate vector encoding (Stage 1)
pub mod polar;

/// Quantized Johnson-Lindenstrauss 1-bit sketching (Stage 2)
pub mod qjl;

/// TurboQuantizer: two-stage composition (Polar + QJL)
pub mod turbo;

/// KV cache compressor for transformer attention
pub mod kv_cache;

/// Online distortion tracking for quality monitoring
pub mod distortion;

/// Tiered quantization: hot, warm, and cold storage tiers
pub mod tiered;

/// Resilient quantization with automatic primary→fallback strategy
pub mod resilient;

/// Oversampled approximate nearest-neighbor search with exact re-ranking
pub mod search;

/// Adaptive per-vector bit-width selection with promote/demote
pub mod adaptive;

/// Prometheus-compatible metrics export for monitoring
pub mod metrics;

// ============================================================================
// Optional modules (behind feature flags)
// ============================================================================

/// SIMD-accelerated kernels (NEON on aarch64, AVX2 on x86_64)
#[cfg(feature = "simd")]
pub mod simd;

/// C FFI exports for cross-language bindings.
///
/// Enable with `features = ["ffi"]` in your `Cargo.toml`.
/// Use `cbindgen` to generate the corresponding C header.
#[cfg(feature = "ffi")]
pub mod ffi;

// ============================================================================
// Public re-exports — the primary API surface
// ============================================================================

// Error types
pub use error::{Result, TurboQuantError};

// Core quantizers
pub use polar::{PolarCode, PolarQuantizer};
pub use qjl::{QjlQuantizer, QjlSketch};
pub use turbo::{TurboCode, TurboQuantizer};

// KV cache
pub use kv_cache::{KvCacheCompressor, KvCacheConfig, MultiHeadKvCache};

// Statistics and monitoring
pub use distortion::DistortionTracker;
pub use stats::{BatchStats, DistortionMetrics};

// Rotation
pub use rotation::StoredRotation;
pub use wht::WhtRotation;

// Tiered, resilient, and search modules
pub use tiered::{Tier, TieredCode, TieredQuantization};
pub use resilient::{ResilientCode, ResilientQuantizer};
pub use search::OversampledSearch;

/// Compact binary format version for all serialized codes.
///
/// Incremented when the wire format changes in a backward-incompatible way.
/// Readers must reject versions they don't understand.
pub const COMPACT_FORMAT_VERSION: u8 = 0x01;
