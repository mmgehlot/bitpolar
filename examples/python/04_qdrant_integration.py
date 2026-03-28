"""BitPolar + Qdrant — two-phase search with compressed re-ranking.

Stores compressed vectors as Qdrant payloads alongside full vectors.
Uses HNSW for candidate retrieval, then BitPolar for precise re-ranking.

Prerequisites:
    pip install bitpolar numpy qdrant-client

Usage:
    python examples/python/04_qdrant_integration.py
"""

import numpy as np
from bitpolar_embeddings.qdrant import BitPolarQdrantIndex

print("=== BitPolar + Qdrant Two-Phase Search ===\n")

try:
    from qdrant_client import QdrantClient

    # Create in-memory Qdrant client (no server needed for demo)
    client = QdrantClient(":memory:")

    # Create BitPolar-enhanced index
    index = BitPolarQdrantIndex(
        client=client,
        collection_name="demo",
        dim=128,
        bits=4,
        oversampling_factor=3,  # Retrieve 3x candidates before re-ranking
    )
    index.create_collection()

    # Generate and add vectors
    n = 500
    vectors = [np.random.randn(128).astype(np.float32) for _ in range(n)]
    index.upsert_vectors(
        ids=list(range(n)),
        vectors=vectors,
        payloads=[{"label": f"vec_{i}"} for i in range(n)],
    )
    print(f"Indexed {n} vectors with BitPolar compression")

    # Search with two-phase re-ranking
    query = np.random.randn(128).astype(np.float32)
    results = index.search(query, top_k=5, use_bitpolar_rerank=True)
    print(f"\nTop 5 results (HNSW + BitPolar re-ranking):")
    for r in results:
        print(f"  ID={r['id']}, score={r['score']:.4f}, label={r['payload'].get('label', '?')}")

    # Compare: without re-ranking
    results_no_rerank = index.search(query, top_k=5, use_bitpolar_rerank=False)
    print(f"\nTop 5 results (HNSW only, no re-ranking):")
    for r in results_no_rerank:
        print(f"  ID={r['id']}, score={r['score']:.4f}")

except ImportError:
    print("qdrant-client not installed — skipping")
    print("Install with: pip install qdrant-client")
