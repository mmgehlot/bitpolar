# Integration Guide

## 1. HNSW Vector Database

Store `TurboCode`s at HNSW graph nodes. During graph traversal, use the fast
IP estimate to score neighbors without decompression. Re-rank only the final
candidate set with exact arithmetic.

```rust
use std::sync::Arc;
use bitpolar::{TurboQuantizer, TurboCode};
use bitpolar::traits::VectorQuantizer;

/// A node in an HNSW-like graph.
struct Node {
    id: usize,
    code: TurboCode,
    // original f32 vector retained for final re-ranking only
    original: Vec<f32>,
}

struct HnswIndex {
    quantizer: Arc<TurboQuantizer>,
    nodes: Vec<Node>,
}

impl HnswIndex {
    fn new(dim: usize, bits: u8, projections: usize, seed: u64) -> Self {
        let quantizer = Arc::new(TurboQuantizer::new(dim, bits, projections, seed).unwrap());
        Self { quantizer, nodes: Vec::new() }
    }

    /// Index a vector: compress on arrival, store original for re-ranking.
    fn insert(&mut self, id: usize, vector: Vec<f32>) {
        let code = self.quantizer.encode(&vector).unwrap();
        self.nodes.push(Node { id, code, original: vector });
    }

    /// Score a candidate node during HNSW graph traversal.
    /// Fast — uses compressed arithmetic, no decompression.
    fn approx_score(&self, node: &Node, query: &[f32]) -> f32 {
        self.quantizer.inner_product_estimate(&node.code, query).unwrap_or(f32::NEG_INFINITY)
    }

    /// Final re-rank on the top-k candidates from graph traversal.
    /// Slower — uses exact f32 arithmetic on original vectors.
    fn exact_score(node: &Node, query: &[f32]) -> f32 {
        node.original.iter().zip(query.iter()).map(|(a, b)| a * b).sum()
    }

    /// Search: approximate traversal + exact re-rank on top candidates.
    fn search(&self, query: &[f32], k: usize, oversample: usize) -> Vec<(usize, f32)> {
        let candidates = k * oversample;

        // Phase 1: approximate scoring on all nodes (replace with HNSW traversal).
        let mut approx: Vec<(usize, f32)> = self.nodes.iter()
            .map(|n| (n.id, self.approx_score(n, query)))
            .collect();

        // Phase 2: keep top `candidates` by approximate score.
        approx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        approx.truncate(candidates);

        // Phase 3: exact re-rank on retained candidates.
        let mut exact: Vec<(usize, f32)> = approx.iter()
            .map(|&(id, _)| (id, Self::exact_score(&self.nodes[id], query)))
            .collect();

        exact.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        exact.truncate(k);
        exact
    }
}
```

**Recommended parameters for HNSW:**
- `bits = 4` — good recall vs. compression trade-off during traversal
- `projections = dim / 4` — reduces residual error without excessive memory
- `oversample = 4` — recover the recall lost to 4-bit quantization

Use `OversampledSearch` from `bitpolar::search` as a ready-made implementation of
the two-phase approximate + exact strategy described above.

---

## 2. LLM Inference: KV Cache

`KvCacheCompressor` compresses key and value vectors on arrival during generation,
enabling long-context inference with bounded GPU/CPU memory.

```rust
use bitpolar::{KvCacheCompressor, KvCacheConfig, MultiHeadKvCache};

// --- Single-head usage ---

let config = KvCacheConfig {
    head_dim: 128,      // dimension per head
    bits: 4,            // 4-bit polar angles
    projections: 32,    // QJL residual projections (head_dim / 4)
    seed: 42,
};
let mut cache = KvCacheCompressor::new(&config).unwrap();

// During token generation: push each (key, value) pair as it is produced.
for _token in 0..2048 {
    let key: Vec<f32>   = vec![0.1_f32; 128]; // from attention layer
    let value: Vec<f32> = vec![0.2_f32; 128];
    cache.push(&key, &value).unwrap();
}

// Compute attention for a query token.
let query: Vec<f32> = vec![0.15_f32; 128];
let scores = cache.attention_scores(&query).unwrap();
// scores[i] = approx(q · K[i]) / sqrt(dim)
println!("tokens cached: {}, compression: {:.2}x",
         cache.len(), cache.compression_ratio());

// --- Multi-head usage ---

let num_heads = 32;
let mut mh = MultiHeadKvCache::new(num_heads, &config).unwrap();

let keys:   Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.1_f32; 128]).collect();
let values: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.2_f32; 128]).collect();
let k_refs: Vec<&[f32]>   = keys.iter().map(|v| v.as_slice()).collect();
let v_refs: Vec<&[f32]>   = values.iter().map(|v| v.as_slice()).collect();
mh.push_token(&k_refs, &v_refs).unwrap();

let queries: Vec<Vec<f32>> = (0..num_heads).map(|_| vec![0.15_f32; 128]).collect();
let q_refs:  Vec<&[f32]>   = queries.iter().map(|v| v.as_slice()).collect();
let head_scores = mh.attention_scores(&q_refs).unwrap();
// head_scores[h] = Vec of scores for head h
```

**Recommended parameters for KV cache:**
- `bits = 4` — good quality for attention score estimation
- `projections = head_dim / 8` — lighter than search; attention is robust to residual error
- Each head gets a different seed automatically in `MultiHeadKvCache` to avoid correlated projections

---

## 3. Python ML Pipelines

Add bitpolar as a Python extension via the C FFI layer (requires `features = ["ffi"]`):

```toml
# Cargo.toml for your Python extension
[dependencies]
bitpolar = { version = "0.1", features = ["ffi", "parallel"] }
```

Then use `ctypes` or `cffi` from Python, or build a `pyo3` wrapper:

```python
import ctypes
import numpy as np

lib = ctypes.CDLL("./libbitpolar.so")  # or .dylib on macOS

# Set up function signatures
lib.bitpolar_new.restype = ctypes.c_void_p
lib.bitpolar_new.argtypes = [ctypes.c_uint32, ctypes.c_uint8,
                              ctypes.c_uint32, ctypes.c_uint64,
                              ctypes.POINTER(ctypes.c_int32)]
lib.bitpolar_encode.restype = ctypes.c_void_p
lib.bitpolar_inner_product.restype = ctypes.c_int32

# Create quantizer: dim=768, bits=4, projections=192, seed=42
err = ctypes.c_int32(0)
q = lib.bitpolar_new(768, 4, 192, 42, ctypes.byref(err))
assert err.value == 0

# Encode a batch of numpy vectors
vectors = np.random.randn(1000, 768).astype(np.float32)
codes = []
for v in vectors:
    code = lib.bitpolar_encode(q, v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               768, ctypes.byref(err))
    assert err.value == 0
    codes.append(code)

# Inner product estimation
query = np.random.randn(768).astype(np.float32)
score = ctypes.c_float(0.0)
lib.bitpolar_inner_product(q, codes[0],
    query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    768, ctypes.byref(score))
print(f"IP estimate: {score.value:.4f}")
```

---

## 4. OversampledSearch for Recall Recovery

When quantizing to low bit widths (3–4 bits), the approximate inner product
can misrank candidates. `OversampledSearch` recovers recall by fetching more
candidates than needed and re-ranking with exact arithmetic.

```rust
use bitpolar::TurboQuantizer;
use bitpolar::search::OversampledSearch;

let q = TurboQuantizer::new(768, 4, 192, 42).unwrap();
let mut index = OversampledSearch::new(q, 4);  // 4× oversampling

// Index vectors (e.g., sentence embeddings)
let vectors: Vec<Vec<f32>> = (0..10_000)
    .map(|i| (0..768).map(|j| ((i * 768 + j) as f32 * 0.001).sin()).collect())
    .collect();

for v in vectors {
    index.add(v).unwrap();
}

// Search: internally fetches top 4*k candidates, re-ranks with exact IP
let query: Vec<f32> = vec![0.1_f32; 768];
let results = index.search(&query, 10).unwrap();
// results: Vec<(index, exact_ip_score)> sorted by descending score
for (idx, score) in &results {
    println!("  index={idx}  score={score:.4}");
}
```

**Choosing `oversample_factor`:**
- `2` — minimal recall recovery, low memory overhead
- `4` — good balance (recommended default)
- `8` — near-perfect recall at 4-bit quantization, 2× more exact IP computations

---

## 5. TieredQuantization for Hot/Warm/Cold Storage

Use different bit widths for vectors at different access frequencies to balance
accuracy and storage cost.

```rust
use bitpolar::tiered::{TieredQuantization, Tier};

let tq = TieredQuantization::new(768, 192, 42).unwrap();
let v: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();

// Hot tier: 8-bit, highest accuracy, accessed frequently
let hot = tq.encode(&v, Tier::Hot).unwrap();

// Warm tier: 4-bit, balanced, moderate access
let warm = tq.encode(&v, Tier::Warm).unwrap();

// Cold tier: 3-bit, maximum compression, rarely accessed
let cold = tq.encode(&v, Tier::Cold).unwrap();

// Promote cold → hot when access pattern changes
let promoted = tq.recompress(&cold, Tier::Hot).unwrap();

// All tiers support identical estimation API
let query: Vec<f32> = vec![0.1_f32; 768];
let _score = tq.inner_product_estimate(&hot, &query).unwrap();
let _score = tq.inner_product_estimate(&warm, &query).unwrap();
let _score = tq.inner_product_estimate(&cold, &query).unwrap();

// Inspect which tier a code belongs to
println!("promoted code tier: {:?}", tq.tier(&promoted));
```

---

## 6. ResilientQuantizer for Graceful Degradation

In production pipelines where errors must not propagate as hard failures:

```rust
use bitpolar::resilient::ResilientQuantizer;

// Primary: TurboQuantizer at 4-bit with 32 projections
// Fallback: PolarQuantizer at 4-bit (simpler, always succeeds for valid input)
let rq = ResilientQuantizer::new(
    768,  // dim
    4,    // primary_bits
    192,  // projections
    42,   // seed
    4,    // fallback_bits
).unwrap();

let v: Vec<f32> = vec![0.1_f32; 768];
let code = rq.encode(&v).unwrap();

if rq.is_fallback(&code) {
    // Log: primary encoder failed, fell back to PolarQuantizer
    eprintln!("warning: using fallback quantizer for this vector");
}

let query: Vec<f32> = vec![0.05_f32; 768];
let score = rq.inner_product_estimate(&code, &query).unwrap();
println!("score: {score:.4}");
```

The fallback is triggered only when the primary returns an error (e.g., dimension
mismatch at construction, which should not happen in practice for valid input).
Both encoders share the same dimension, so for well-formed inputs the primary will
always succeed.
