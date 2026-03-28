"""BitPolar + DSPy retriever module.

Implements a DSPy-compatible retrieval model that uses BitPolar
for compressed similarity search over a corpus.

Prerequisites:
    pip install bitpolar dspy numpy

Usage:
    python examples/python/20_dspy_retriever.py
"""

import numpy as np
from bitpolar_dspy import BitPolarRM

print("=== BitPolar + DSPy Retriever ===\n")

# Build a corpus with mock embeddings
corpus = [
    "Vector quantization maps continuous vectors to discrete codes",
    "Product quantization splits vectors into sub-vectors and quantizes each",
    "Locality-sensitive hashing uses random projections for approximate search",
    "BitPolar achieves training-free quantization via random sign projections",
    "Scalar quantization maps each dimension to a fixed number of bits",
    "HNSW builds a hierarchical navigable small world graph for search",
    "IVF-PQ combines inverted file indexing with product quantization",
    "Binary quantization uses 1 bit per dimension for extreme compression",
]
corpus_embeddings = np.random.randn(len(corpus), 256).astype(np.float32)

# Create retrieval model
rm = BitPolarRM(
    corpus=corpus,
    corpus_embeddings=corpus_embeddings,
    dim=256,
    bits=4,
    seed=42,
    k=3,
)
print(f"Indexed {rm.corpus_size()} passages ({rm.memory_bytes():,} bytes compressed)")

# Forward pass: retrieve passages for a query
query = "How does BitPolar compress vectors?"
query_embedding = np.random.randn(256).astype(np.float32)
results = rm.forward(query=query, query_embedding=query_embedding)
print(f"\nQuery: '{query}'")
print(f"Retrieved {len(results)} passages:")
for i, r in enumerate(results):
    print(f"  {i+1}. (score={r['score']:.4f}) {r['text'][:65]}...")

# Batch retrieval
print("\n--- Batch Retrieval ---")
batch_queries = ["quantization methods", "approximate nearest neighbor"]
batch_embeddings = np.random.randn(2, 256).astype(np.float32)
batch_results = rm.forward_batch(
    queries=batch_queries,
    query_embeddings=batch_embeddings,
)
for q, results in zip(batch_queries, batch_results):
    print(f"\nQuery: '{q}'")
    for r in results:
        print(f"  (score={r['score']:.4f}) {r['text'][:65]}...")

# Stats
stats = rm.stats()
print(f"\nRetriever stats: {stats['corpus_size']} passages, {stats['ratio']:.1f}x compression")
