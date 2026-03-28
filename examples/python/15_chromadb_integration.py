"""BitPolar + ChromaDB — compressed embedding store.

Wraps ChromaDB collections with BitPolar compression for
reduced memory usage and optional compressed re-ranking.

Prerequisites:
    pip install bitpolar chromadb numpy

Usage:
    python examples/python/15_chromadb_integration.py
"""

import numpy as np
from bitpolar_chromadb import BitPolarChromaStore

print("=== BitPolar + ChromaDB ===\n")

# Create store (uses ephemeral ChromaDB client internally)
store = BitPolarChromaStore(
    collection_name="demo",
    dim=384,
    bits=4,
    seed=42,
)

# Add documents with mock embeddings
texts = [
    "Quantum computing uses qubits for computation",
    "Neural networks learn patterns from data",
    "Rust provides memory safety without garbage collection",
    "Python is the most popular language for data science",
    "Vector databases enable similarity search at scale",
    "BitPolar compresses vectors with random projections",
    "Transformers use self-attention mechanisms",
    "Graph neural networks operate on structured data",
]
embeddings = np.random.randn(len(texts), 384).astype(np.float32)
ids = [f"doc_{i}" for i in range(len(texts))]
metadatas = [{"topic": t.split()[0].lower()} for t in texts]

store.add(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
print(f"Added {len(texts)} documents")

# Search with re-ranking
query_embedding = np.random.randn(384).astype(np.float32)
results = store.search(
    query_embedding=query_embedding,
    n_results=4,
    rerank=True,        # use BitPolar compressed scores to rerank
    rerank_factor=3,    # retrieve 3x candidates before reranking
)
print(f"\nSearch results (with BitPolar re-ranking, top 4):")
for r in results:
    print(f"  [{r['id']}] score={r['score']:.4f}: {r['text'][:50]}...")

# Search without re-ranking for comparison
results_raw = store.search(query_embedding=query_embedding, n_results=4, rerank=False)
print(f"\nSearch results (no re-ranking, top 4):")
for r in results_raw:
    print(f"  [{r['id']}] score={r['score']:.4f}: {r['text'][:50]}...")

# Stats
stats = store.stats()
print(f"\nStore stats:")
print(f"  Documents: {stats['count']}")
print(f"  Compressed memory: {stats['compressed_bytes']:,}B")
print(f"  Compression ratio: {stats['ratio']:.1f}x")
