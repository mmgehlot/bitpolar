"""Qdrant integration for BitPolar compressed vector storage.

Provides a helper that stores BitPolar-compressed vectors as Qdrant payloads
alongside full-precision vectors. Enables two-phase search: fast approximate
scoring on compressed payloads + exact re-ranking on full vectors.

This is the recommended integration path until BitPolar is merged into
Qdrant's native quantization engine.

Usage:
    >>> from bitpolar_embeddings.qdrant import BitPolarQdrantIndex
    >>> from qdrant_client import QdrantClient
    >>>
    >>> client = QdrantClient(":memory:")
    >>> index = BitPolarQdrantIndex(client, "my_collection", dim=384, bits=4)
    >>> index.upsert_vectors(ids=[1, 2], vectors=[vec1, vec2])
    >>> results = index.search(query_vec, top_k=10)
"""

from __future__ import annotations

import base64
from typing import Optional, Sequence

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarQdrantIndex:
    """Qdrant collection with BitPolar compressed scoring.

    Stores compressed vectors as payload bytes in Qdrant and provides
    a two-phase search: approximate scoring on compressed vectors,
    then exact re-ranking on the top candidates.

    This approach works with any Qdrant deployment (cloud or local)
    without modifications to Qdrant itself.

    Args:
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        dim: Vector dimension
        bits: BitPolar quantization precision (3-8)
        projections: QJL projections (default: dim/4)
        seed: Random seed
        oversampling_factor: How many candidates to retrieve before re-ranking
    """

    def __init__(
        self,
        client,
        collection_name: str,
        dim: int,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
        oversampling_factor: int = 3,
    ):
        self._client = client
        self._collection = collection_name
        self._dim = dim
        self._bits = bits
        self._seed = seed
        self._oversampling = oversampling_factor

        proj = projections or max(dim // 4, 1)
        self._quantizer = _bp.TurboQuantizer(
            dim=dim, bits=bits, projections=proj, seed=seed
        )

    def create_collection(self, on_disk: bool = False):
        """Create the Qdrant collection with vector configuration.

        Args:
            on_disk: Whether to store vectors on disk (for large datasets)
        """
        try:
            from qdrant_client.models import (
                Distance,
                VectorParams,
            )
        except ImportError:
            raise ImportError(
                "qdrant-client required. Install with: pip install qdrant-client"
            )

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=self._dim,
                distance=Distance.DOT,
                on_disk=on_disk,
            ),
        )

    def upsert_vectors(
        self,
        ids: Sequence[int],
        vectors: Sequence[np.ndarray],
        payloads: Optional[Sequence[dict]] = None,
    ):
        """Upsert vectors with BitPolar-compressed payloads.

        Stores both the original vector (for Qdrant's native HNSW search)
        and a compressed version in the payload (for BitPolar scoring).

        Args:
            ids: Point IDs
            vectors: float32 vectors of shape (dim,)
            payloads: Optional additional payload data per vector
        """
        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            raise ImportError("qdrant-client required")

        points = []
        for i, (pid, vec) in enumerate(zip(ids, vectors)):
            vec_np = np.asarray(vec, dtype=np.float32)
            code = self._quantizer.encode(vec_np)

            payload = payloads[i].copy() if payloads and i < len(payloads) else {}
            # Store compressed bytes as base64 in payload
            payload["_bitpolar_code"] = base64.b64encode(bytes(code)).decode("ascii")
            payload["_bitpolar_bits"] = self._bits
            payload["_bitpolar_seed"] = self._seed

            points.append(PointStruct(
                id=pid,
                vector=vec_np.tolist(),
                payload=payload,
            ))

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        use_bitpolar_rerank: bool = True,
    ) -> list[dict]:
        """Search the collection using Qdrant HNSW + optional BitPolar re-ranking.

        Two-phase search:
        1. Qdrant's native HNSW retrieves top_k * oversampling_factor candidates
        2. BitPolar re-ranks candidates using compressed inner product estimation
        3. Return top_k results

        Args:
            query: float32 query vector
            top_k: Number of final results
            use_bitpolar_rerank: Whether to use compressed re-ranking

        Returns:
            List of dicts with 'id', 'score', and 'payload' keys
        """
        query_np = np.asarray(query, dtype=np.float32)

        # Phase 1: Qdrant HNSW retrieval with oversampling
        oversample_k = top_k * self._oversampling if use_bitpolar_rerank else top_k
        qdrant_results = self._client.search(
            collection_name=self._collection,
            query_vector=query_np.tolist(),
            limit=oversample_k,
        )

        if not use_bitpolar_rerank:
            return [
                {"id": r.id, "score": r.score, "payload": r.payload}
                for r in qdrant_results
            ]

        # Phase 2: BitPolar re-ranking on compressed payloads
        if len(query_np) != self._dim:
            raise ValueError(f"Query dimension {len(query_np)} != index dimension {self._dim}")
        reranked = []
        for r in qdrant_results:
            code_b64 = r.payload.get("_bitpolar_code")
            if code_b64:
                code = np.frombuffer(
                    base64.b64decode(code_b64), dtype=np.uint8
                ).copy()
                bp_score = self._quantizer.inner_product(code, query_np)
                reranked.append({
                    "id": r.id,
                    "score": bp_score,
                    "qdrant_score": r.score,
                    "payload": {
                        k: v for k, v in r.payload.items()
                        if not k.startswith("_bitpolar_")
                    },
                })
            else:
                reranked.append({
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload,
                })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bits(self) -> int:
        return self._bits

    def __repr__(self) -> str:
        return (
            f"BitPolarQdrantIndex(collection='{self._collection}', "
            f"dim={self._dim}, bits={self._bits})"
        )
