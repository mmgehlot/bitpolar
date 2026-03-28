"""BitPolar + Haystack DocumentStore and Retriever.

Implements a Haystack-compatible document store with compressed
embeddings and a retriever component for pipeline integration.

Prerequisites:
    pip install bitpolar haystack-ai numpy

Usage:
    python examples/python/19_haystack_pipeline.py
"""

import numpy as np
from bitpolar_haystack import BitPolarDocumentStore, BitPolarRetriever

print("=== BitPolar + Haystack Pipeline ===\n")

# --- Document Store ----------------------------------------------------------
print("--- BitPolarDocumentStore ---")
store = BitPolarDocumentStore(dim=384, bits=4, seed=42)

# Write documents with mock embeddings
documents = [
    {"id": "d1", "content": "Machine learning models require large datasets", "embedding": np.random.randn(384).astype(np.float32)},
    {"id": "d2", "content": "Vector quantization reduces memory by 4-8x", "embedding": np.random.randn(384).astype(np.float32)},
    {"id": "d3", "content": "Transformers use multi-head self-attention", "embedding": np.random.randn(384).astype(np.float32)},
    {"id": "d4", "content": "Retrieval-augmented generation combines search with LLMs", "embedding": np.random.randn(384).astype(np.float32)},
    {"id": "d5", "content": "BitPolar is a training-free quantization method", "embedding": np.random.randn(384).astype(np.float32)},
    {"id": "d6", "content": "Haystack provides composable NLP pipelines", "embedding": np.random.randn(384).astype(np.float32)},
]

store.write_documents(documents)
print(f"Wrote {store.count_documents()} documents")

# --- Retriever ---------------------------------------------------------------
print("\n--- BitPolarRetriever ---")
retriever = BitPolarRetriever(document_store=store)

query_embedding = np.random.randn(384).astype(np.float32)
results = retriever.run(query_embedding=query_embedding, top_k=3)
print(f"Retrieved {len(results['documents'])} documents:")
for doc in results["documents"]:
    print(f"  [{doc['id']}] score={doc['score']:.4f}: {doc['content'][:55]}...")

# --- Filter support ----------------------------------------------------------
print("\n--- Filtered retrieval ---")
store.write_documents([
    {"id": "d7", "content": "Python 3.12 adds performance improvements", "embedding": np.random.randn(384).astype(np.float32), "meta": {"topic": "python"}},
    {"id": "d8", "content": "Rust 1.75 stabilizes async traits", "embedding": np.random.randn(384).astype(np.float32), "meta": {"topic": "rust"}},
])
results = retriever.run(query_embedding=query_embedding, top_k=3, filters={"topic": "python"})
print(f"Filtered results (topic=python): {len(results['documents'])} docs")
for doc in results["documents"]:
    print(f"  [{doc['id']}] {doc['content'][:55]}...")

# Stats
stats = store.stats()
print(f"\nStore: {stats['count']} docs, {stats['compressed_bytes']:,}B, {stats['ratio']:.1f}x compression")
