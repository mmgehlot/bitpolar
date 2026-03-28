"""Letta (MemGPT) archival memory tier with BitPolar compression.

Provides a compressed archival memory implementation that matches
Letta's archival memory interface, enabling agents to store and
retrieve long-term knowledge with reduced memory footprint.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarArchivalMemory:
    """Letta-compatible archival memory with BitPolar compression.

    Mimics Letta's archival memory tier for storing and retrieving
    long-term agent knowledge. Embeddings are stored in compressed
    form using BitPolar quantization for 4-8x memory reduction.

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
        self._insertion_order: list[str] = []

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

    def insert(self, text: str, embedding: List[float]) -> str:
        """Insert a text passage into archival memory.

        Args:
            text: The text content to archive
            embedding: Embedding vector for the text

        Returns:
            String ID of the archived entry
        """
        quantizer = self._ensure_quantizer()
        entry_id = str(uuid.uuid4())
        vec = np.array(embedding, dtype=np.float32)
        if len(vec) != self._dim:
            raise ValueError(f"Embedding dimension {len(vec)} != configured dim {self._dim}")
        code = quantizer.encode(vec)

        self._entries[entry_id] = {
            "id": entry_id,
            "text": text,
            "code": code,
            "vector": vec,
        }
        self._insertion_order.append(entry_id)
        return entry_id

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> list[dict]:
        """Search archival memory by semantic similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of results

        Returns:
            List of dicts with 'id', 'text', 'score' keys,
            sorted by descending similarity
        """
        if not self._entries:
            return []

        quantizer = self._ensure_quantizer()
        q_vec = np.array(query_embedding, dtype=np.float32)

        scored: list[dict] = []
        for entry in self._entries.values():
            score = float(quantizer.inner_product(entry["code"], q_vec))
            scored.append({
                "id": entry["id"],
                "text": entry["text"],
                "score": score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete(self, id: str) -> None:
        """Delete an archival entry by ID.

        Args:
            id: The entry ID to delete

        Raises:
            KeyError: If the ID does not exist
        """
        if id not in self._entries:
            raise KeyError(f"Archival entry '{id}' not found")
        del self._entries[id]
        self._insertion_order = [eid for eid in self._insertion_order if eid != id]

    def get_all(self) -> list[dict]:
        """Retrieve all archival entries in insertion order.

        Returns:
            List of dicts with 'id' and 'text' keys
        """
        results: list[dict] = []
        for eid in self._insertion_order:
            entry = self._entries.get(eid)
            if entry:
                results.append({
                    "id": entry["id"],
                    "text": entry["text"],
                })
        return results

    def size(self) -> int:
        """Return the number of entries in archival memory."""
        return len(self._entries)
