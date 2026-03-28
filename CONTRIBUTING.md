# Contributing to BitPolar

Thank you for your interest in contributing. This guide covers everything you need to get started.

## Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/mmgehlot/bitpolar.git
cd bitpolar

# Run the full test suite (requires Rust stable)
cargo test --all-features

# Run tests without default features
cargo test --no-default-features

# Run benchmarks
cargo bench

# Check for lint warnings (must be zero before opening a PR)
cargo clippy --all-features -- -D warnings

# Format code (required before committing)
cargo fmt --all
```

## Coding Standards

- **Formatting**: All code must be formatted with `rustfmt`. Run `cargo fmt --all` before each commit. CI will reject unformatted code.
- **Clippy**: Zero warnings at `cargo clippy --all-features -- -D warnings`.
- **Unsafe code**: Unsafe is only permitted inside `#[cfg(feature = "simd")]` or `#[cfg(feature = "ffi")]` blocks. All other code is covered by `#![cfg_attr(not(any(feature = "simd", feature = "ffi")), forbid(unsafe_code))]`.
- **Documentation**: Every public item must have a doc comment (`///`). Include at least one sentence explaining what the item does and an `# Errors` section for fallible functions.
- **Tests**: Every new function needs at least one unit test. Add integration tests in `tests/` for cross-module behavior.
- **No panics in public API**: All fallible operations must return `Result<_, TurboQuantError>`. Internal panics are only acceptable for logic that is provably unreachable.

## Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

Types:

| Type    | When to use |
|---------|-------------|
| `feat`  | New feature or public API addition |
| `fix`   | Bug fix |
| `docs`  | Documentation-only changes |
| `test`  | Adding or updating tests |
| `bench` | Benchmark additions or performance improvements |
| `ci`    | Changes to CI/CD configuration |
| `refactor` | Internal refactoring with no behavior change |
| `chore` | Dependency bumps, tooling, non-code changes |

Examples:

```
feat(tiered): add TieredQuantization with hot/warm/cold tiers
fix(qjl): avoid division by zero when norm is 0.0
docs(readme): add parameter selection table
test(turbo): property-based round-trip tests via proptest
bench(search): add OversampledSearch recall benchmark
ci: add cargo-deny license check
```

## Pull Request Process

1. **Branch from `main`**: `git checkout -b feat/my-feature main`
2. **Keep PRs focused**: One logical change per PR. Split unrelated changes into separate PRs.
3. **All tests must pass**: `cargo test --all-features` and `cargo test --no-default-features`.
4. **Clippy must be clean**: `cargo clippy --all-features -- -D warnings` — zero warnings.
5. **Documentation builds clean**: `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`.
6. **Update CHANGELOG.md**: Add an entry under `## [Unreleased]` describing your change.
7. **Request review**: At least one maintainer review is required before merge.

## Adding a New Quantization Strategy

To add a new quantizer (e.g., `MyQuantizer`):

1. **Create `src/my_quantizer.rs`** with your implementation.
2. **Implement `VectorQuantizer`** (in `src/traits.rs`):
   ```rust
   impl VectorQuantizer for MyQuantizer {
       type Code = MyCode;
       fn encode(&self, vector: &[f32]) -> Result<Self::Code> { ... }
       fn decode(&self, code: &Self::Code) -> Vec<f32> { ... }
       fn inner_product_estimate(&self, code: &Self::Code, query: &[f32]) -> Result<f32> { ... }
       fn l2_distance_estimate(&self, code: &Self::Code, query: &[f32]) -> Result<f32> { ... }
       fn dim(&self) -> usize { ... }
       fn code_size_bytes(&self, code: &Self::Code) -> usize { ... }
   }
   ```
3. **Implement `SerializableCode`** for `MyCode` with a versioned compact binary format.
4. **Add unit tests** in the same file covering: correct output shape, dimension mismatch errors, non-finite input errors, and compact serialization round-trips.
5. **Add a benchmark** in `benches/quantization_bench.rs` for encode, decode, inner product, and L2.
6. **Declare the module** in `src/lib.rs` and add pub use re-exports.
7. **Update CHANGELOG.md**.

## Running Benchmarks and Interpreting Results

```bash
# Run all benchmarks and save an HTML report
cargo bench

# Run a specific benchmark group
cargo bench turbo_encode

# Compare against a baseline (requires criterion baseline support)
cargo bench -- --save-baseline before
# make your change
cargo bench -- --baseline before
```

Benchmark output shows median throughput and confidence intervals. A change is significant if the confidence interval does not overlap with the baseline. Aim for no performance regression on the core `turbo_encode`, `turbo_inner_product`, and `polar_encode` benchmarks.

## Running Fuzz Tests

Fuzz testing requires the `cargo-fuzz` tool and nightly Rust:

```bash
cargo install cargo-fuzz

# List available fuzz targets
cargo fuzz list

# Run a fuzz target (Ctrl+C to stop)
cargo +nightly fuzz run fuzz_turbo_encode

# Run the compact-deserialization fuzzer
cargo +nightly fuzz run fuzz_compact_deser

# Run the QJL sketcher fuzzer
cargo +nightly fuzz run fuzz_qjl_sketch
```

Fuzz targets live in `fuzz/fuzz_targets/`. All targets must not panic on any input — returning `Err` is acceptable.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to abide by its terms. Please report unacceptable behavior to the maintainers.
