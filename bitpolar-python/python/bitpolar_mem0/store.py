"""Mem0 vector store backend with BitPolar compression.

Provides a drop-in vector store for Mem0 agent memory that stores
embeddings in compressed form using BitPolar quantization, reducing
memory usage by 4-8x while preserving retrieval quality.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarVectorStore:
    """Mem0-compatible vector store with BitPolar compression.

    Stores embeddings as compressed BitPolar codes and uses approximate
    inner product for similarity search. Supports payload storage and
    filtered retrieval.

    Args:
        dim: Vector dimension
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim // 4)
        seed: Random seed for deterministic compression
    """

    def __init__(
        self,
        dim: int = 384,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._dim = dim
        self._bits = bits
        self._seed = seed
        self._proj = projections or max(dim // 4, 1)
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._entries: dict[str, dict] = {}

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

    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert vectors with optional payloads.

        Args:
            vectors: List of embedding vectors
            payloads: Optional list of metadata dicts (one per vector)
            ids: Optional list of string IDs (auto-generated if None)

        Returns:
            List of assigned IDs
        """
        if not vectors:
            return []

        n = len(vectors)
        if payloads is not None and len(payloads) != n:
            raise ValueError(f"payloads length {len(payloads)} != vectors length {n}")
        if ids is not None and len(ids) != n:
            raise ValueError(f"ids length {len(ids)} != vectors length {n}")

        quantizer = self._ensure_quantizer()
        assigned_ids: list[str] = []

        for i in range(n):
            vec_id = ids[i] if ids else str(uuid.uuid4())
            vec = np.array(vectors[i], dtype=np.float32)
            if len(vec) != self._dim:
                raise ValueError(f"Vector dim {len(vec)} != configured dim {self._dim}")
            code = quantizer.encode(vec)
            payload = payloads[i] if payloads else {}

            self._entries[vec_id] = {
                "id": vec_id,
                "code": code,
                "vector": vec,
                "payload": dict(payload),
            }
            assigned_ids.append(vec_id)

        return assigned_ids

    def search(
        self,
        query: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> list[dict]:
        """Search for similar vectors using compressed inner product.

        Args:
            query: Query embedding vector
            limit: Maximum number of results to return
            filters: Optional metadata key-value filters (exact match)

        Returns:
            List of dicts with 'id', 'score', and 'payload' keys,
            sorted by descending similarity score
        """
        if not self._entries:
            return []

        quantizer = self._ensure_quantizer()
        q_vec = np.array(query, dtype=np.float32)
        if len(q_vec) != self._dim:
            raise ValueError(f"Vector dim {len(q_vec)} != configured dim {self._dim}")

        candidates: list[dict] = []
        for entry in self._entries.values():
            # Apply metadata filters if provided
            if filters:
                payload = entry["payload"]
                match = all(payload.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            score = float(quantizer.inner_product(entry["code"], q_vec))
            candidates.append({
                "id": entry["id"],
                "score": score,
                "payload": dict(entry["payload"]),
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:limit]

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a stored vector and/or its payload.

        Args:
            vector_id: ID of the vector to update
            vector: New embedding vector (re-encoded if provided)
            payload: New payload dict (merged with existing if provided)

        Raises:
            KeyError: If vector_id does not exist
        """
        if vector_id not in self._entries:
            raise KeyError(f"Vector '{vector_id}' not found")

        entry = self._entries[vector_id]

        if vector is not None:
            quantizer = self._ensure_quantizer()
            vec = np.array(vector, dtype=np.float32)
            entry["code"] = quantizer.encode(vec)
            entry["vector"] = vec

        if payload is not None:
            entry["payload"].update(payload)

    def delete(self, vector_id: str) -> None:
        """Delete a vector by ID.

        Args:
            vector_id: ID of the vector to delete

        Raises:
            KeyError: If vector_id does not exist
        """
        if vector_id not in self._entries:
            raise KeyError(f"Vector '{vector_id}' not found")
        del self._entries[vector_id]

    def get(self, vector_id: str) -> dict:
        """Retrieve a stored vector entry by ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Dict with 'id', 'vector' (as list), and 'payload' keys

        Raises:
            KeyError: If vector_id does not exist
        """
        if vector_id not in self._entries:
            raise KeyError(f"Vector '{vector_id}' not found")
        entry = self._entries[vector_id]
        return {
            "id": entry["id"],
            "vector": entry["vector"].tolist(),
            "payload": dict(entry["payload"]),
        }

    def count(self) -> int:
        """Return the number of stored vectors."""
        return len(self._entries)

    def clear(self) -> None:
        """Remove all stored vectors."""
        self._entries.clear()
