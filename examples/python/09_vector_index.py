"""BitPolar VectorIndex — in-memory compressed search index.

Build an index, add vectors, and search — all using compressed codes.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/09_vector_index.py
"""

import numpy as np
import bitpolar
import time

print("=== BitPolar VectorIndex ===\n")

dim = 384
n = 10000

# Create index
idx = bitpolar.VectorIndex(dim=dim, bits=4, projections=96, seed=42)

# Add vectors
print(f"Adding {n} vectors (dim={dim})...")
start = time.time()
vectors = np.random.randn(n, dim).astype(np.float32)
for i in range(n):
    idx.add(i, vectors[i])
elapsed = time.time() - start
print(f"Indexed {n} vectors in {elapsed:.2f}s ({n/elapsed:.0f} vectors/sec)")
print(f"Index size: {len(idx)} vectors")

# Search
query = vectors[42]  # Search for vector 42
start = time.time()
ids, scores = idx.search(query, top_k=10)
search_ms = (time.time() - start) * 1000
print(f"\nSearch completed in {search_ms:.1f}ms")
print(f"Top 10 results: {ids.tolist()}")
print(f"Top 10 scores: {[f'{s:.4f}' for s in scores.tolist()]}")

# Verify: vector 42 should be the top result (or close)
if 42 in ids[:3]:
    print(f"✓ Vector 42 found in top 3!")
else:
    print(f"Note: Vector 42 not in top 3 (expected at 4-bit compression)")
