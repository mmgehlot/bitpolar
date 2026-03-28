# Architecture

## Module Dependency Graph

```
lib.rs  (public re-exports, feature gates)
 ├── error.rs            (TurboQuantError, Result, validate_finite)
 ├── traits.rs           (VectorQuantizer, BatchQuantizer, RotationStrategy, SerializableCode)
 ├── stats.rs            (BatchStats, DistortionMetrics)
 ├── distortion.rs       ──► stats.rs
 ├── codebook.rs         ──► error.rs
 ├── rotation.rs         ──► error.rs, traits.rs
 ├── polar.rs            ──► codebook.rs, error.rs, traits.rs
 ├── qjl.rs              ──► error.rs, traits.rs
 ├── turbo.rs            ──► polar.rs, qjl.rs, rotation.rs, stats.rs, traits.rs
 ├── kv_cache.rs         ──► error.rs, turbo.rs
 ├── tiered.rs           ──► error.rs, turbo.rs, traits.rs
 ├── resilient.rs        ──► error.rs, polar.rs, turbo.rs, traits.rs
 ├── search.rs           ──► error.rs, traits.rs
 ├── simd/mod.rs         (feature = "simd") — no internal deps
 └── ffi.rs              (feature = "ffi") ──► error.rs, turbo.rs
```

## Data Flow: Vector → Rotation → Polar → QJL → TurboCode

```
Input f32 slice (dim elements)
         │
         │  validate_finite()
         ▼
┌─────────────────────────────┐
│  StoredRotation.apply_slice │  R ∈ ℝ^{d×d} Haar-distributed orthogonal matrix
│  rotated = R × vector       │  Storage: d² × 4 bytes (e.g. d=1536 → 9.4 MB)
└────────────┬────────────────┘
             │  rotated: Vec<f32>
             ▼
┌─────────────────────────────┐
│  PolarQuantizer.encode      │  Groups into d/2 pairs (x, y)
│  For each pair:             │    r = √(x²+y²)       stored as f32 (lossless)
│    radii[i]   = r           │    θ = atan2(y,x)      quantized to `bits` bits
│    angles[i]  = Q(θ/π)     │  via LloydMaxCodebook (binary-search boundaries)
└────────────┬────────────────┘
             │  PolarCode { radii: Vec<f32>, angle_indices: Vec<u16>, bits }
             │
             │  (TurboQuantizer also computes residual = rotated - decoded_polar)
             ▼
┌─────────────────────────────┐
│  QjlQuantizer.sketch        │  S ∈ ℝ^{m×d} Gaussian projection matrix
│  For each row g_p of S:     │  Storage: m × d × 4 bytes (e.g. m=384, d=1536 → 2.4 MB)
│    sign_bit[p] = sign(g_p·r)│  r = residual vector; stored packed 8 bits/byte
│  sketch.norm = ‖residual‖   │
└────────────┬────────────────┘
             │
             ▼
TurboCode { polar: PolarCode, residual_sketch: QjlSketch }
  compact size ≈ (d/2)×6 + ceil(m/8) + 4 bytes
```

## Trait Hierarchy

```
VectorQuantizer (traits.rs)
  ├── PolarQuantizer  (polar.rs)    implements VectorQuantizer<Code=PolarCode>
  ├── TurboQuantizer (turbo.rs)     implements VectorQuantizer<Code=TurboCode>
  └── BatchQuantizer (traits.rs)    supertrait of VectorQuantizer, requires parallel feature
        ├── PolarQuantizer  (parallel feature)
        └── TurboQuantizer  (parallel feature)

RotationStrategy (traits.rs)
  └── StoredRotation (rotation.rs)

SerializableCode (traits.rs)
  ├── PolarCode   (polar.rs)
  ├── QjlSketch   (qjl.rs)
  └── TurboCode   (turbo.rs)
```

Key design choice: `TieredQuantization`, `ResilientQuantizer`, and `OversampledSearch` do NOT
implement `VectorQuantizer` — they are higher-level wrappers that own concrete quantizers and
expose richer APIs (tier selection, fallback, oversampled search).

## Feature Flag Architecture

| Feature          | Default | Gates                                              |
|------------------|---------|----------------------------------------------------|
| `std`            | Yes     | `nalgebra` dep → true Haar QR rotation; without it uses a random-sign diagonal fallback |
| `serde-support`  | Yes     | `serde::Serialize/Deserialize` derives on all code/config types |
| `simd`           | No      | `src/simd/mod.rs` module; permits `unsafe` in that module only |
| `parallel`       | No      | `rayon` dep; `BatchQuantizer` impls on `PolarQuantizer` and `TurboQuantizer` |
| `tracing-support`| No      | `tracing` dep; `#[tracing::instrument]` on all hot public methods |
| `ffi`            | No      | `src/ffi.rs` module; permits `unsafe` in that module only |

The crate-level `#![cfg_attr(not(any(feature = "simd", feature = "ffi")), forbid(unsafe_code))]`
ensures that the default build is entirely safe Rust.

## Thread Safety Model

All quantizers are immutable after construction. Rust's ownership system enforces this:

| Type                  | Send | Sync | Notes                                         |
|-----------------------|------|------|-----------------------------------------------|
| `StoredRotation`      | Yes  | Yes  | Immutable `Vec<f32>`; no interior mutability  |
| `PolarQuantizer`      | Yes  | Yes  | Immutable after `new()`                       |
| `QjlQuantizer`        | Yes  | Yes  | Immutable after `new()`                       |
| `TurboQuantizer`      | Yes  | Yes  | Composes the above; same guarantees           |
| `TieredQuantization`  | Yes  | Yes  | Three immutable `TurboQuantizer` fields       |
| `ResilientQuantizer`  | Yes  | Yes  | Immutable primary + fallback                  |
| `OversampledSearch`   | Yes  | No   | Mutable (`add` takes `&mut self`)             |
| `DistortionTracker`   | Yes  | No   | Mutable (`observe` takes `&mut self`)         |
| `KvCacheCompressor`   | Yes  | No   | Mutable (`push` takes `&mut self`)            |
| `MultiHeadKvCache`    | Yes  | No   | Mutable (`push_token` takes `&mut self`)      |

All immutable types are safe to share via `Arc<Q>` across threads. Mutable types
should be wrapped in `Mutex<T>` or `RwLock<T>` for concurrent access.

## Memory Layout

### Rotation matrix — O(d²)

```
StoredRotation.data: Vec<f32>
  length = d × d
  bytes  = d² × 4

Examples:
  d =  128 →   65 536 bytes    (64 KB)
  d =  768 →  2 359 296 bytes  (2.3 MB)
  d = 1536 →  9 437 184 bytes  (9.4 MB)
```

Stored row-major: element (i,j) at `data[i*d + j]`.

### QJL projection matrix — O(m·d)

```
QjlQuantizer.projection_matrix: Vec<f32>
  length = m × d  (m = num_projections, d = dim)
  bytes  = m × d × 4

Recommended: m = d/4
  d =  128, m = 32  →   16 384 bytes  (16 KB)
  d =  768, m = 192 →  589 824 bytes  (576 KB)
  d = 1536, m = 384 → 2 359 296 bytes (2.4 MB)
```

### Compressed code sizes

```
PolarCode:
  radii:         (d/2) × 4 bytes   (f32 per pair, lossless)
  angle_indices: (d/2) × 2 bytes   (u16 per pair regardless of bits)
  total:         (d/2) × 6 bytes

QjlSketch:
  signs:  ceil(m/8) bytes          (1 bit per projection)
  norm:   4 bytes
  total:  ceil(m/8) + 4 bytes

TurboCode:
  polar:            (d/2) × 6 bytes
  residual_sketch:  ceil(m/8) + 4 bytes
  total ≈           d×3 + ceil(m/8) + 4 bytes

Example (d=1536, m=384, bits=4):
  PolarCode:    4608 bytes
  QjlSketch:    52   bytes
  TurboCode:    4660 bytes
  Original f32: 6144 bytes → compression ratio ≈ 1.32×

For bits=3, m=128 (maximum compression):
  PolarCode:  4608 bytes (unchanged — radii dominate)
  QjlSketch:  20   bytes
  TurboCode:  4628 bytes → compression ratio ≈ 1.33×
  (radii dominate; remove radii with PolarQuant-only for 3× compression)
```
