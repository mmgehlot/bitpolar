"""Zep-style compressed store with time-decay scoring.

Provides agent memory storage with BitPolar compression and recency-weighted
retrieval, matching the Zep agent memory pattern of combining semantic
similarity with temporal relevance.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarZepStore:
    """Zep-compatible compressed memory store with time-decay scoring.

    Combines BitPolar-compressed semantic similarity with exponential
    time-decay weighting to surface both relevant and recent memories.

    Args:
        dim: Vector dimension
        bits: Quantization precision (3-8, default 4)
        decay_rate: Exponential decay rate in per-second units (default 1e-6)
        projections: Number of QJL projections (default: dim // 4)
        seed: Random seed for deterministic compression
    """

    def __init__(
        self,
        dim: int = 384,
        bits: int = 4,
        decay_rate: float = 1e-6,
        projections: Optional[int] = None,
        seed: int = 42,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._dim = dim
        self._bits = bits
        self._decay_rate = decay_rate
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

    def _time_decay(self, timestamp: float) -> float:
        """Compute exponential time-decay weight for a given timestamp.

        Args:
            timestamp: Unix timestamp of the memory

        Returns:
            Decay weight in (0, 1]
        """
        age = time.time() - timestamp
        return math.exp(-self._decay_rate * max(age, 0.0))

    def add(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> str:
        """Add a memory entry with text, embedding, and optional metadata.

        Args:
            text: The text content of the memory
            embedding: Embedding vector for the text
            metadata: Optional metadata dict
            timestamp: Unix timestamp (defaults to current time)

        Returns:
            String ID of the stored memory
        """
        if not embedding:
            raise ValueError("embedding cannot be empty")
        quantizer = self._ensure_quantizer()
        entry_id = str(uuid.uuid4())
        vec = np.array(embedding, dtype=np.float32)
        code = quantizer.encode(vec)
        ts = timestamp if timestamp is not None else time.time()

        self._entries[entry_id] = {
            "id": entry_id,
            "text": text,
            "code": code,
            "vector": vec,
            "metadata": dict(metadata) if metadata else {},
            "timestamp": ts,
        }
        return entry_id

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        time_weight: bool = True,
    ) -> list[dict]:
        """Search memories with optional time-decay weighting.

        Computes a combined score: similarity * time_decay_weight when
        time_weight is True, otherwise ranks by similarity alone.

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of results
            time_weight: Whether to apply recency weighting (default True)

        Returns:
            List of dicts with 'id', 'text', 'score', 'similarity',
            'recency', 'metadata', 'timestamp' keys
        """
        if not self._entries:
            return []

        quantizer = self._ensure_quantizer()
        q_vec = np.array(query_embedding, dtype=np.float32)

        scored: list[dict] = []
        for entry in self._entries.values():
            similarity = float(quantizer.inner_product(entry["code"], q_vec))
            recency = self._time_decay(entry["timestamp"]) if time_weight else 1.0
            combined = similarity * recency

            scored.append({
                "id": entry["id"],
                "text": entry["text"],
                "score": combined,
                "similarity": similarity,
                "recency": recency,
                "metadata": dict(entry["metadata"]),
                "timestamp": entry["timestamp"],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete(self, id: str) -> None:
        """Delete a memory entry by ID.

        Args:
            id: The memory ID to delete

        Raises:
            KeyError: If the ID does not exist
        """
        if id not in self._entries:
            raise KeyError(f"Memory '{id}' not found")
        del self._entries[id]

    def clear(self) -> None:
        """Remove all memory entries."""
        self._entries.clear()

    def count(self) -> int:
        """Return the number of stored memories."""
        return len(self._entries)
