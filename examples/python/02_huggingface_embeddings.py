"""BitPolar + HuggingFace sentence-transformers.

Compress any sentence-transformers model's embeddings with one function call.

Prerequisites:
    pip install bitpolar numpy sentence-transformers

Usage:
    python examples/python/02_huggingface_embeddings.py
"""

import numpy as np
from bitpolar_embeddings import compress_embeddings, BitPolarEncoder

# =============================================================================
# Method 1: compress_embeddings() — compress any numpy embedding matrix
# =============================================================================

print("=== Method 1: compress_embeddings() ===\n")

# Simulate embeddings (in production, these come from sentence-transformers)
embeddings = np.random.randn(1000, 384).astype(np.float32)

# One-line compression
compressed = compress_embeddings(embeddings, bits=4)
print(f"Compressed {compressed.n_vectors} vectors")
print(f"Compression ratio: {compressed.compression_ratio:.1f}x")
print(f"Memory: {compressed.memory_bytes:,} bytes (vs {1000*384*4:,} original)")

# Search
query = np.random.randn(384).astype(np.float32)
ids, scores = compressed.search(query, top_k=5)
print(f"\nTop 5 results: {ids.tolist()}")
print(f"Scores: {scores.tolist()}")

# Decompress specific vectors
decoded = compressed.decompress(ids)
print(f"Decompressed top results: shape={decoded.shape}")

# Save and load
compressed.save("/tmp/embeddings.bp")
loaded = type(compressed).load("/tmp/embeddings.bp")
print(f"Loaded from disk: {loaded.n_vectors} vectors")

# =============================================================================
# Method 2: BitPolarEncoder — wraps a sentence-transformers model
# =============================================================================

print("\n=== Method 2: BitPolarEncoder (wraps sentence-transformers) ===\n")

try:
    encoder = BitPolarEncoder("all-MiniLM-L6-v2", bits=4)
    compressed = encoder.encode(["Hello world", "Semantic search is powerful", "BitPolar compresses vectors"])
    print(f"Encoded {compressed.n_vectors} sentences → {compressed.compression_ratio:.1f}x compression")

    # Search by encoding a query
    query_compressed = encoder.encode(["How does search work?"])
    query_vec = query_compressed.decompress()[0]
    ids, scores = compressed.search(query_vec, top_k=2)
    print(f"Most similar to 'How does search work?': indices {ids.tolist()}")
except ImportError:
    print("sentence-transformers not installed — skipping BitPolarEncoder demo")
    print("Install with: pip install sentence-transformers")
