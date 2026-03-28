# Parameter Reference

## Bit-Width Selection

The `bits` parameter controls polar angle quantization. Radii are always stored
losslessly as f32 regardless of bit width.

| Bits | Levels | Use case                           | Compression | Recall@10 (d=768) |
|------|--------|------------------------------------|-------------|-------------------|
| 3    | 8      | Cold storage, maximum compression  | ~3.3×       | ~0.85             |
| 4    | 16     | Balanced search / KV cache         | ~3.2×       | ~0.92             |
| 6    | 64     | High-recall retrieval              | ~3.0×       | ~0.97             |
| 8    | 256    | Hot tier / near-lossless           | ~2.8×       | ~0.99             |

Notes:
- Compression ratios above exclude the rotation and projection matrices (one-time overhead).
- Recall figures are approximate averages over standard embedding datasets (cosine similarity).
- At 3 bits the angle distortion becomes noticeable; combine with `OversampledSearch` to recover recall.
- At 8 bits the angle distortion is negligible for most applications; the dominant error source is the radii approximation from the QJL residual.

## Projection Count

The `projections` parameter controls the QJL residual sketch. More projections reduce
the residual estimation variance at the cost of a larger projection matrix.

| Use case               | Recommended formula | Example (d=768) | Matrix size  |
|------------------------|---------------------|-----------------|--------------|
| Standard search        | `dim / 4`           | 192 projections | 576 KB       |
| KV cache               | `dim / 8`           | 96 projections  | 288 KB       |
| Maximum compression    | `dim / 16`          | 48 projections  | 144 KB       |
| Research / quality     | `dim / 2`           | 384 projections | 1.15 MB      |

The QJL variance bound is `Var[estimate] ≤ π / (2·m) · ‖v‖² · ‖q‖²` where m is the
projection count. Doubling projections halves the variance.

For the smallest possible sketch (e.g., edge inference), use `projections = 16`
regardless of dimension. The residual correction will be noisy but still unbiased.

## Seed Management

The `seed` parameter is a `u64` that fully determines the rotation matrix and
projection matrix. The same `(dim, bits, projections, seed)` tuple always produces
the same quantizer.

**Deterministic from a fixed seed:**
```rust
let q = TurboQuantizer::new(768, 4, 192, 42).unwrap();
// Same quantizer every time — safe to store seed instead of matrices.
```

**Per-tenant seeds** — isolate tenants so one cannot infer another's vectors:
```rust
let tenant_seed: u64 = hash(tenant_id) ^ global_secret;
let q = TurboQuantizer::new(768, 4, 192, tenant_seed).unwrap();
```

**Per-collection seeds** — use different quantizers for different embedding models:
```rust
let openai_q    = TurboQuantizer::new(1536, 4, 384, 0xADA002).unwrap();
let cohere_q    = TurboQuantizer::new(1024, 4, 256, 0xC0HERE).unwrap();
let voyage_q    = TurboQuantizer::new(1024, 4, 256, 0xV0Y4GE).unwrap();
```

**MultiHeadKvCache** assigns per-head seeds automatically:
```
head h uses seed = config.seed.wrapping_add(h as u64)
```
This prevents correlated projections across attention heads.

## Memory Calculator

### One-time overhead (quantizer state)

```
rotation_matrix_bytes = dim² × 4
projection_matrix_bytes = projections × dim × 4

total_quantizer_bytes = rotation_matrix_bytes + projection_matrix_bytes
                      = dim² × 4 + projections × dim × 4
                      = 4 × dim × (dim + projections)

Example (dim=768, projections=192):
  rotation:   768² × 4   = 2 359 296 bytes  (2.3 MB)
  projection: 192 × 768 × 4 = 589 824 bytes (576 KB)
  total:                    = 2 949 120 bytes (2.9 MB)
```

### Per-vector storage (codes)

```
polar_code_bytes  = (dim / 2) × 6
qjl_sketch_bytes  = ceil(projections / 8) + 4
turbo_code_bytes  = polar_code_bytes + qjl_sketch_bytes

For N vectors:
  total_code_storage = N × turbo_code_bytes

Example (dim=768, projections=192, N=1_000_000):
  polar:    768/2 × 6     = 2304 bytes/vector
  qjl:      ceil(192/8)+4 = 28   bytes/vector
  total:                  = 2332 bytes/vector
  1M vectors: 2.33 GB
  (vs. original f32: 768 × 4 × 1M = 3.07 GB → 1.32× compression)
```

## Performance Expectations

### Encode latency (single vector, no SIMD)

Dominated by the O(d²) rotation matrix multiply:

| Dimension | Typical encode latency (single thread) |
|-----------|----------------------------------------|
| 128       | ~5 µs                                  |
| 512       | ~80 µs                                 |
| 768       | ~180 µs                                |
| 1536      | ~700 µs                                |

With the `simd` feature enabled (NEON/AVX2), rotation matrix multiply is
approximately 4–8× faster (auto-vectorised in scalar path, explicit intrinsics
in the simd path).

### Inner product estimation throughput

IP estimation is O(d + m) per code — much faster than encoding:

| Scenario                    | Throughput (single thread)     |
|-----------------------------|-------------------------------|
| d=768, m=192, 10k vectors   | ~50k estimates/sec             |
| d=1536, m=384, 10k vectors  | ~15k estimates/sec             |
| `parallel` feature, 8 cores | ~8× above figures             |

The `parallel` feature enables rayon-based `BatchQuantizer::batch_inner_product`
which parallelises over the code array with near-linear scaling up to core count.
