"""CrewAI memory backend with BitPolar embedding compression.

Provides a storage backend compatible with CrewAI's Memory(storage=backend)
pattern, compressing embeddings for memory-efficient agent memory.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarMemoryBackend:
    """CrewAI-compatible memory storage backend with BitPolar compression.

    Stores embeddings in compressed form using BitPolar quantization,
    reducing memory footprint while maintaining high-quality similarity
    search via compressed inner product scoring.

    Compatible with CrewAI's ``Memory(storage=backend)`` pattern.

    Args:
        bits: Quantization precision (3-8). Lower values give more
            compression; 4 is the recommended default.
        seed: Random seed for reproducible quantization.

    Example:
        >>> backend = BitPolarMemoryBackend(bits=4)
        >>> backend.save("doc1", "hello", {}, [0.1]*384)
        >>> results = backend.search([0.1]*384, top_k=3)
    """

    def __init__(self, bits: int = 4, seed: int = 42) -> None:
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._bits = bits
        self._seed = seed
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._quantizer_dim: Optional[int] = None
        self._dim: Optional[int] = None
        self._store: dict[str, dict[str, Any]] = {}

    def _ensure_quantizer(self, dim: int) -> _bp.TurboQuantizer:
        """Lazily initialize the quantizer for the given dimension.

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

    def save(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Save a memory entry with optional compressed embedding.

        Args:
            key: Unique identifier for the memory entry.
            value: The content to store (text, dict, etc.).
            metadata: Optional metadata dictionary.
            embedding: Optional float32 embedding vector to compress
                and store alongside the value.
        """
        entry: dict[str, Any] = {
            "value": value,
            "metadata": metadata or {},
        }

        if embedding is not None:
            vec = np.array(embedding, dtype=np.float32)
            if self._dim is None:
                self._dim = len(vec)
            elif len(vec) != self._dim:
                raise ValueError(f"Embedding dimension {len(vec)} != stored dimension {self._dim}")
            quantizer = self._ensure_quantizer(len(vec))
            code = quantizer.encode(vec)
            entry["code"] = code
            entry["dim"] = len(vec)
        else:
            entry["code"] = None
            entry["dim"] = None

        self._store[key] = entry

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar memories using compressed inner product.

        Scores all stored embeddings against the query using BitPolar's
        compressed inner product, then returns the top-k results.

        Args:
            query_embedding: Float32 query vector.
            top_k: Number of top results to return.
            filters: Optional metadata filters. Each key-value pair must
                match the entry's metadata for inclusion.

        Returns:
            List of result dicts with 'key', 'value', 'metadata', 'score',
            sorted by descending score.
        """
        query_vec = np.array(query_embedding, dtype=np.float32)
        self._ensure_quantizer(len(query_vec))

        if self._dim is not None and len(query_vec) != self._dim:
            raise ValueError(f"Query dim {len(query_vec)} != stored dim {self._dim}")

        scored: list[dict[str, Any]] = []
        for key, entry in self._store.items():
            if entry["code"] is None:
                continue

            # Apply metadata filters
            if filters:
                meta = entry.get("metadata", {})
                if not all(meta.get(fk) == fv for fk, fv in filters.items()):
                    continue

            score = float(self._quantizer.inner_product(entry["code"], query_vec))
            scored.append(
                {
                    "key": key,
                    "value": entry["value"],
                    "metadata": entry["metadata"],
                    "score": round(score, 6),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory entry by key.

        Args:
            key: The unique identifier of the memory entry.

        Returns:
            Dict with 'value' and 'metadata', or None if not found.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        return {
            "key": key,
            "value": entry["value"],
            "metadata": entry["metadata"],
        }

    def reset(self) -> None:
        """Clear all stored memory entries.

        Removes all entries from the store and resets the quantizer.
        """
        self._store.clear()
        self._quantizer = None
        self._quantizer_dim = None
