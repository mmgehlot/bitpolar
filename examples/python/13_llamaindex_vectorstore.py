"""BitPolar + LlamaIndex VectorStore.

Compressed in-memory vector store compatible with LlamaIndex's
VectorStoreIndex. No external database needed.

Prerequisites:
    pip install bitpolar llama-index-core numpy

Usage:
    python examples/python/13_llamaindex_vectorstore.py
"""

import numpy as np
from bitpolar_llamaindex import BitPolarVectorStore

print("=== BitPolar + LlamaIndex VectorStore ===\n")

# Create vector store
store = BitPolarVectorStore(dim=384, bits=4, seed=42)

# Simulate LlamaIndex TextNode objects with mock embeddings
nodes = [
    {"id": "node_0", "text": "BitPolar uses random projections for quantization", "embedding": np.random.randn(384).astype(np.float32)},
    {"id": "node_1", "text": "No training data needed — works out of the box", "embedding": np.random.randn(384).astype(np.float32)},
    {"id": "node_2", "text": "Supports 1-8 bit quantization levels", "embedding": np.random.randn(384).astype(np.float32)},
    {"id": "node_3", "text": "Inner product estimates are provably unbiased", "embedding": np.random.randn(384).astype(np.float32)},
    {"id": "node_4", "text": "Compatible with any embedding model", "embedding": np.random.randn(384).astype(np.float32)},
    {"id": "node_5", "text": "600x faster indexing than Product Quantization", "embedding": np.random.randn(384).astype(np.float32)},
]

# Add nodes
for node in nodes:
    store.add(node["id"], node["text"], node["embedding"])
print(f"Added {len(nodes)} nodes to BitPolarVectorStore")
print(f"Memory: {store.memory_bytes():,} bytes")

# Query
query_embedding = np.random.randn(384).astype(np.float32)
results = store.query(query_embedding, top_k=3)
print(f"\nQuery results (top 3):")
for r in results:
    print(f"  id={r['id']}, score={r['score']:.4f}: {r['text'][:60]}")

# Delete a node
store.delete("node_2")
print(f"\nAfter deleting node_2: {store.size()} nodes remain")

# Query again
results = store.query(query_embedding, top_k=3)
print(f"Query after deletion (top 3):")
for r in results:
    print(f"  id={r['id']}, score={r['score']:.4f}: {r['text'][:60]}")

# Stats
stats = store.stats()
print(f"\nStore stats:")
print(f"  Nodes: {stats['count']}")
print(f"  Compression: {stats['ratio']:.1f}x")
print(f"  Memory: {stats['memory_bytes']:,} bytes")
