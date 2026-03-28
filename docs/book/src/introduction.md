# BitPolar

Near-optimal vector quantization with zero training overhead.

BitPolar compresses high-dimensional embeddings to 3-8 bits with provably unbiased inner products and no calibration data. It implements three algorithms from Google Research:

- **TurboQuant** (ICLR 2026) — two-stage composition for near-optimal compression
- **PolarQuant** (AISTATS 2026) — polar coordinate encoding with lossless radii
- **QJL** (AAAI 2025) — 1-bit Johnson-Lindenstrauss sketching

## Key Properties

| Property | Description |
|---|---|
| **Data-oblivious** | No training, no codebooks, no calibration data |
| **Deterministic** | Fully defined by 4 integers: `(dimension, bits, projections, seed)` |
| **Provably unbiased** | `E[estimate] = exact` at 3+ bits (theorem-backed) |
| **Near-optimal** | Within ~2.7x of Shannon rate-distortion limit |
| **Instant** | Vectors compress on arrival — 600x faster than Product Quantization |

## Available Everywhere

| Platform | Package | Install |
|---|---|---|
| Rust | `bitpolar` | `cargo add bitpolar` |
| Python | `bitpolar` | `pip install bitpolar` |
| JavaScript | `bitpolar-wasm` | `npm install bitpolar-wasm` |
| Node.js | `bitpolar` | `npm install bitpolar` |
| PostgreSQL | `bitpolar-pg` | `cargo pgrx install` |
| gRPC | `bitpolar-server` | Docker / cargo install |
| Go | `bitpolar-go` | CGO wrapper |
| Java | `com.bitpolar` | JNI wrapper |
| CLI | `bitpolar` | `cargo install bitpolar-cli` |
| Browser | WASM demo | bitpolar.dev |
