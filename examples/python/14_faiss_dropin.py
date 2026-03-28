"""BitPolar FAISS-compatible Index — drop-in replacement.

Provides IndexBitPolarIP, IndexBitPolarL2, and IndexBitPolarIDMap
with the same API as FAISS but using compressed storage.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/14_faiss_dropin.py
"""

import numpy as np
from bitpolar_faiss import IndexBitPolarIP, IndexBitPolarL2, IndexBitPolarIDMap

print("=== BitPolar FAISS Drop-in Replacement ===\n")

dim = 128
n = 5000
vectors = np.random.randn(n, dim).astype(np.float32)
queries = np.random.randn(5, dim).astype(np.float32)

# --- IndexBitPolarIP (inner product) ----------------------------------------
print("--- IndexBitPolarIP ---")
index_ip = IndexBitPolarIP(dim, bits=4, seed=42)
index_ip.add(vectors)
print(f"Index size: {index_ip.ntotal} vectors")

distances, ids = index_ip.search(queries, k=5)
print(f"Search for 5 queries, top 5 each:")
for i in range(min(3, len(queries))):
    print(f"  query {i}: ids={ids[i].tolist()}, scores={[f'{d:.3f}' for d in distances[i]]}")

ip_bytes = index_ip.memory_bytes()
faiss_bytes = n * dim * 4  # faiss.IndexFlatIP stores raw float32
print(f"Memory: {ip_bytes:,}B vs FAISS IndexFlatIP: {faiss_bytes:,}B ({faiss_bytes / ip_bytes:.1f}x savings)")

# --- IndexBitPolarL2 (Euclidean) --------------------------------------------
print("\n--- IndexBitPolarL2 ---")
index_l2 = IndexBitPolarL2(dim, bits=4, seed=42)
index_l2.add(vectors)

distances, ids = index_l2.search(queries[:1], k=5)
print(f"L2 search: ids={ids[0].tolist()}")
print(f"L2 distances: {[f'{d:.3f}' for d in distances[0]]}")

# --- IndexBitPolarIDMap (external IDs) --------------------------------------
print("\n--- IndexBitPolarIDMap ---")
index_idmap = IndexBitPolarIDMap(dim, bits=4, seed=42)

external_ids = np.arange(1000, 1000 + n, dtype=np.int64)
index_idmap.add_with_ids(vectors, external_ids)
print(f"Index size: {index_idmap.ntotal} vectors (external IDs: {external_ids[0]}..{external_ids[-1]})")

distances, ids = index_idmap.search(queries[:1], k=5)
print(f"Search result IDs (external): {ids[0].tolist()}")
print(f"All IDs are in [1000, {1000+n}): {all(1000 <= i < 1000+n for i in ids[0])}")
