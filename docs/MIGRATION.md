# Migration Guide: From Product Quantization to BitPolar

## Why Migrate?

| Property | Product Quantization (FAISS) | BitPolar |
|---|---|---|
| Training required | Yes (k-means, hours for large datasets) | No (instant, 4 integers) |
| Inner product bias | Biased estimates | Provably unbiased |
| Streaming ingestion | Blocked during training | Instant (encode on arrival) |
| Deterministic | No (k-means has random init) | Yes (same seed = same quantizer) |
| Compression | 8-32x | 3-10x |
| Recall@10 (4-bit) | ~0.90-0.95 | ~0.90-0.95 |

## Migration Steps

### Step 1: Install BitPolar

**Rust:**
```bash
cargo add bitpolar
```

**Python:**
```bash
pip install bitpolar
```

### Step 2: Replace PQ Encoding

**Before (FAISS PQ):**
```python
import faiss
import numpy as np

# Training required — blocks for minutes/hours
quantizer = faiss.IndexPQ(dim, n_subquantizers, n_bits)
quantizer.train(training_data)  # ← This is what BitPolar eliminates
quantizer.add(vectors)

# Search
distances, indices = quantizer.search(query, k=10)
```

**After (BitPolar):**
```python
import numpy as np
from bitpolar_embeddings import compress_embeddings

# No training — instant compression
compressed = compress_embeddings(vectors, bits=4)

# Search
indices, scores = compressed.search(query, top_k=10)
```

### Step 3: Re-encode Existing Vectors

If you have original vectors stored alongside PQ codes:

```python
# Load your original vectors
vectors = np.load("embeddings.npy")  # float32, shape (n, dim)

# Compress with BitPolar
compressed = compress_embeddings(vectors, bits=4, seed=42)

# Save new format
compressed.save("embeddings.bp")

# Verify quality
from_pq = faiss_index.reconstruct_n(0, n)
from_bp = compressed.decompress()
pq_error = np.linalg.norm(vectors - from_pq, axis=1).mean()
bp_error = np.linalg.norm(vectors - from_bp, axis=1).mean()
print(f"PQ reconstruction error: {pq_error:.4f}")
print(f"BitPolar reconstruction error: {bp_error:.4f}")
```

### Step 4: Update Your Search Pipeline

**HNSW + PQ → HNSW + BitPolar:**

The key change: replace the PQ scoring function with BitPolar's asymmetric
inner product estimation.

```rust
// Before: PQ scoring during HNSW traversal
let score = pq_index.asymmetric_distance(code, query);

// After: BitPolar scoring
let score = quantizer.inner_product_estimate(&code, query)?;
```

### What You Lose

- PQ can achieve higher compression ratios (32x) at lower quality
- PQ codebooks can be fine-tuned for specific data distributions
- FAISS has GPU-accelerated PQ (BitPolar is CPU-only currently)

### What You Gain

- Zero training time (microseconds vs hours)
- Streaming ingestion (no batch retraining needed)
- Provably unbiased inner product estimates
- Deterministic from 4 integers (reproducible across machines)
- Same recall at comparable bit widths
