"""AutoGen / Semantic Kernel compatible memory store with BitPolar compression.

Provides a memory store interface that compresses embeddings using BitPolar
quantization, compatible with Microsoft AutoGen's memory interface.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarMemoryStore:
    """AutoGen / Semantic Kernel compatible memory store with BitPolar compression.

    Stores text and embeddings in compressed form using BitPolar quantization.
    Provides add/search/get/delete/clear operations compatible with
    Microsoft AutoGen's memory interface.

    Args:
        bits: Quantization precision (3-8). Lower values give more
            compression; 4 is the recommended default.
        seed: Random seed for reproducible quantization.

    Example:
        >>> store = BitPolarMemoryStore(bits=4)
        >>> store.add("id1", "hello world", embedding=[0.1]*384)
        >>> results = store.search(query_embedding=[0.1]*384, top_k=3)
        >>> entry = store.get("id1")
        >>> store.delete("id1")
        >>> store.clear()
    """

    def __init__(self, bits: int = 4, seed: int = 42) -> None:
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._bits = bits
        self._seed = seed
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._quantizer_dim: Optional[int] = None
        self._store: dict[str, Tuple[str, np.ndarray, int]] = {}

    def _ensure_quantizer(self, dim: int) -> _bp.TurboQuantizer:
        """Lazily initialize or reinitialize the quantizer for the given dimension.

        Args:
            dim: Vector dimension.

        Returns:
            The initialized TurboQuantizer instance.
        """
        if self._quantizer is None or self._quantizer_dim != dim:
            proj = max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )
            self._quantizer_dim = dim
        return self._quantizer

    def add(
        self,
        id: str,
        text: str,
        embedding: List[float],
    ) -> None:
        """Add a memory entry with text and compressed embedding.

        Args:
            id: Unique string identifier for the memory entry.
            text: The text content to store.
            embedding: Float32 embedding vector to compress and store.

        Raises:
            ValueError: If the embedding is empty.
        """
        if not embedding:
            raise ValueError("embedding must not be empty")

        vec = np.array(embedding, dtype=np.float32)
        dim = len(vec)
        quantizer = self._ensure_quantizer(dim)
        code = quantizer.encode(vec)

        self._store[id] = (text, code, dim)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar memories using compressed inner product.

        Scores all stored embeddings against the query using BitPolar's
        compressed inner product, returning the top-k most similar entries.

        Args:
            query_embedding: Float32 query vector.
            top_k: Number of top results to return.

        Returns:
            List of result dicts with 'id', 'text', 'score', sorted by
            descending score.
        """
        if not self._store:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        self._ensure_quantizer(len(query_vec))

        scored: list[dict[str, Any]] = []
        for entry_id, (text, code, dim) in self._store.items():
            score = float(self._quantizer.inner_product(code, query_vec))
            scored.append({
                "id": entry_id,
                "text": text,
                "score": round(score, 6),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory entry by ID.

        Args:
            id: The unique identifier of the memory entry.

        Returns:
            Dict with 'id', 'text', and 'dim', or None if not found.
        """
        entry = self._store.get(id)
        if entry is None:
            return None

        text, code, dim = entry
        return {
            "id": id,
            "text": text,
            "dim": dim,
        }

    def delete(self, id: str) -> bool:
        """Delete a memory entry by ID.

        Args:
            id: The unique identifier of the memory entry to delete.

        Returns:
            True if the entry was found and deleted, False otherwise.
        """
        if id in self._store:
            del self._store[id]
            return True
        return False

    def clear(self) -> None:
        """Clear all stored memory entries.

        Removes all entries from the store and resets the quantizer.
        """
        self._store.clear()
        self._quantizer = None
        self._quantizer_dim = None

    def count(self) -> int:
        """Return the number of entries in the store.

        Returns:
            Number of stored memory entries.
        """
        return len(self._store)

    def contains(self, id: str) -> bool:
        """Check if an entry exists in the store.

        Args:
            id: The unique identifier to check.

        Returns:
            True if the entry exists, False otherwise.
        """
        return id in self._store
