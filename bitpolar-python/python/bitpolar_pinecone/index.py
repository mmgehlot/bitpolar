"""Pinecone integration with BitPolar client-side compression.

Stores vectors in Pinecone with compressed BitPolar codes in metadata
for efficient two-phase re-ranking after Pinecone ANN retrieval.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarPineconeIndex:
    """Pinecone index with BitPolar client-side compression and two-phase search.

    Phase 1: Pinecone ANN retrieval (oversampled)
    Phase 2: BitPolar compressed inner product re-ranking

    Args:
        index_name: Pinecone index name
        dim: Vector dimension
        bits: Quantization precision (3-8, default 4)
        oversampling: Oversampling factor for two-phase search (default 3)
        projections: Number of QJL projections (default: dim // 4)
        seed: Random seed
        api_key: Pinecone API key (reads PINECONE_API_KEY env var if None)
        namespace: Pinecone namespace (default "")
    """

    def __init__(
        self,
        index_name: str,
        dim: int,
        bits: int = 4,
        oversampling: int = 3,
        projections: Optional[int] = None,
        seed: int = 42,
        api_key: Optional[str] = None,
        namespace: str = "",
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("pinecone required. Install with: pip install pinecone")

        self._index_name = index_name
        self._dim = dim
        self._bits = bits
        self._oversampling = oversampling
        self._seed = seed
        self._namespace = namespace
        self._proj = projections or max(dim // 4, 1)
        self._quantizer: Optional[_bp.TurboQuantizer] = None

        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)

    def _ensure_quantizer(self) -> _bp.TurboQuantizer:
        """Lazily initialise the quantizer on first use."""
        if self._quantizer is None:
            self._quantizer = _bp.TurboQuantizer(
                dim=self._dim,
                bits=self._bits,
                projections=self._proj,
                seed=self._seed,
            )
        return self._quantizer

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Upsert vectors with BitPolar compression in metadata.

        Stores the original vector in Pinecone and a base64-encoded
        BitPolar code in the _bp_code metadata field for re-ranking.

        Args:
            ids: List of unique string IDs
            vectors: List of embedding vectors
            metadatas: Optional list of metadata dicts per vector
        """
        if len(ids) != len(vectors):
            raise ValueError(f"ids length {len(ids)} != vectors length {len(vectors)}")

        for vec in vectors:
            if not vec:
                raise ValueError("Empty vector not allowed")
            if len(vec) != self._dim:
                raise ValueError(f"Vector dimension {len(vec)} != expected {self._dim}")

        quantizer = self._ensure_quantizer()
        upsert_data: list[dict[str, Any]] = []

        for i, (vid, vec) in enumerate(zip(ids, vectors)):
            vec_np = np.array(vec, dtype=np.float32)
            code = quantizer.encode(vec_np)
            code_b64 = base64.b64encode(bytes(code)).decode("ascii")

            metadata: dict[str, Any] = {}
            if metadatas and i < len(metadatas) and metadatas[i]:
                metadata.update(metadatas[i])
            metadata["_bp_code"] = code_b64

            upsert_data.append({
                "id": vid,
                "values": vec,
                "metadata": metadata,
            })

        # Pinecone recommends batching in groups of 100
        batch_size = 100
        for start in range(0, len(upsert_data), batch_size):
            batch = upsert_data[start : start + batch_size]
            self._index.upsert(
                vectors=batch,
                namespace=self._namespace,
            )

    def search(
        self,
        query: List[float],
        top_k: int = 10,
        rerank: bool = True,
        filter: Optional[Dict[str, Any]] = None,
    ) -> list[dict]:
        """Two-phase search: Pinecone ANN retrieval + BitPolar re-ranking.

        Args:
            query: Query embedding vector
            top_k: Number of final results
            rerank: Whether to apply BitPolar re-ranking (default True)
            filter: Optional Pinecone metadata filter

        Returns:
            List of dicts with 'id', 'score', and 'metadata' keys
        """
        if len(query) != self._dim:
            raise ValueError(f"Vector dimension {len(query)} != expected {self._dim}")

        n_candidates = top_k * self._oversampling if rerank else top_k

        query_kwargs: dict[str, Any] = {
            "vector": query,
            "top_k": n_candidates,
            "include_metadata": True,
            "namespace": self._namespace,
        }
        if filter:
            query_kwargs["filter"] = filter

        results = self._index.query(**query_kwargs)

        if not results.get("matches"):
            return []

        candidates: list[dict] = []
        for match in results["matches"]:
            metadata = dict(match.get("metadata", {}))
            bp_code = metadata.pop("_bp_code", "")
            entry: dict[str, Any] = {
                "id": match["id"],
                "score": match.get("score", 0.0),
                "metadata": metadata,
                "_bp_code": bp_code,
            }
            candidates.append(entry)

        # Phase 2: Re-rank with BitPolar compressed inner product
        if rerank:
            quantizer = self._ensure_quantizer()
            q_vec = np.array(query, dtype=np.float32)
            for entry in candidates:
                code_b64 = entry.get("_bp_code", "")
                if code_b64:
                    try:
                        code = np.frombuffer(base64.b64decode(code_b64), dtype=np.uint8).copy()
                    except Exception as e:
                        continue  # Skip entries with corrupt codes
                    entry["score"] = float(quantizer.inner_product(code, q_vec))

            candidates.sort(key=lambda x: x["score"], reverse=True)

        for c in candidates:
            c.pop("_bp_code", None)

        return candidates[:top_k]

    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID.

        Args:
            ids: List of vector IDs to delete
        """
        if ids:
            self._index.delete(ids=ids, namespace=self._namespace)
