# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.4] - 2026-06-14

### Fixed
- **Compact serialization was bit-width-invariant** (`PolarCode`): radii were
  stored as lossless f32 and angle indices as u16 regardless of `bits`, so the
  serialized size — and thus compression ratio — never changed with the
  configured bit-width. On real LLM KV caches this capped compression at ~1.2x
  vs f32 (expansion vs f16). Now bit-packs angles and quantizes radii (format
  v0x03), so size scales with bit-width as intended.
- `polar::encode`: clamp radius to finite (large-but-finite inputs could overflow
  `x²+y²` to inf → NaN reconstruction).
- `polar::from_compact_bytes`: reject non-finite/negative `radii_scale`.
- `qjl::inner_product_estimate{,_with_norm}`: return an error (was a panic) on a
  sketch whose projection count differs from the quantizer's.
- `turbo::from_compact_bytes`: `checked_add` on the polar length (32-bit overflow).
- `kv_cache::evict_outside_window`: trim keys and values by the same offset.

### Added
- **Independent radii vs angle bit budgets** (`with_radii_bits`): magnitude and
  direction precision tuned separately. For cosine-optimized workloads (KV
  caches), radii compress hard at ~2 bits for ~50% more ratio at near-identical
  quality.

### Changed
- `tiered::Tier` is now `#[repr(u8)]` with stable discriminants (serialized as
  `u8` by `AdaptiveCode::to_bytes`).
- `COMPACT_FORMAT_VERSION` 0x01 → 0x03 (breaking wire format; readers reject old).

## [0.2.0] - 2026-03-27

### Added
- Walsh-Hadamard Transform rotation: O(d log d) time, O(d) memory (577x less than Haar QR)
- `WhtRotation` implementing `RotationStrategy` trait for drop-in replacement
- Python bindings via PyO3 + maturin (`pip install bitpolar`)
  - `TurboQuantizer`: encode/decode/inner_product with numpy arrays
  - `VectorIndex`: add/search with top-k results
  - Zero-copy numpy input via `PyReadonlyArray`
- WASM bindings via wasm-bindgen (`npm install bitpolar-wasm`)
  - `WasmQuantizer`: encode/decode/inner_product in browser
  - `WasmVectorIndex`: add/search for browser-side vector search
  - `search_with_scores()` returning interleaved IDs + score bits
- `no_std` support with `alloc` feature flag
  - `compat` module with math function wrappers (sqrtf, sinf, cosf, atan2f, expf)
  - `libm` fallback for embedded/edge targets
  - `alloc` feature for Vec/String without full std
- Workspace structure: root crate + bitpolar-wasm + bitpolar-python

### Changed
- Bumped version to 0.2.0
- `fwht_in_place` and `fwht_normalized` are now `pub(crate)` (not public API)
- All math calls use `compat::math` wrappers for no_std compatibility
- `std::result::Result` → `core::result::Result` in error.rs
- `std::iter::once` → `core::iter::once` in codebook.rs
- `std::mem::size_of` → `core::mem::size_of` throughout

## [0.1.0] - 2026-03-26

### Added
- TurboQuantizer: two-stage vector quantization (PolarQuant + QJL)
- PolarQuantizer: single-stage polar coordinate encoding
- QjlQuantizer: 1-bit Johnson-Lindenstrauss sketching
- KvCacheCompressor and MultiHeadKvCache for transformer attention
- DistortionTracker for online quality monitoring
- VectorQuantizer, BatchQuantizer, RotationStrategy, SerializableCode traits
- Compact binary serialization with version headers
- SIMD kernels for NEON (aarch64) and AVX2 (x86_64)
- Parallel batch operations via rayon
- C FFI layer with opaque handle API
- OpenTelemetry-compatible tracing instrumentation
- Criterion benchmarks for encode/decode/IP/L2
- 170+ tests including conformance, integration, property-based
- Dual MIT/Apache-2.0 license
