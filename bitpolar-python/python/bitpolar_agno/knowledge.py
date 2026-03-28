"""Agno (Phidata) knowledge base backed by BitPolar compressed embeddings.

Provides a knowledge base that stores documents with their embeddings in
compressed form using BitPolar quantization. Compatible with Agno's
knowledge base interface for integration into Agno agents and assistants.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarKnowledgeBase:
    """Agno-compatible knowledge base with BitPolar compressed embeddings.

    Stores documents alongside their compressed vector representations
    for memory-efficient similarity search. Each entry is stored as a
    tuple of (text, compressed_code, metadata) keyed by a unique string ID.

    Args:
        dim: Vector dimensionality. The quantizer is initialized eagerly
            with this dimension.
        bits: Quantization precision (3-8, default 4).
        seed: Random seed for deterministic compression.

    Example:
        >>> kb = BitPolarKnowledgeBase(dim=384, bits=4)
        >>> kb.add(
        ...     texts=["The cat sat on the mat"],
        ...     embeddings=[[0.1] * 384],
        ...     metadatas=[{"source": "test"}],
        ... )
        >>> results = kb.search(query_embedding=[0.1] * 384, top_k=1)
        >>> results[0]["text"]
        'The cat sat on the mat'
    """

    def __init__(self, dim: int, bits: int = 4, seed: int = 42):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._dim = dim
        self._bits = bits
        self._seed = seed

        proj = max(dim // 4, 1)
        self._quantizer = _bp.TurboQuantizer(
            dim=dim, bits=bits, projections=proj, seed=seed
        )

        # Internal storage: id -> (text, compressed_code, metadata)
        self._entries: Dict[str, Tuple[str, np.ndarray, Dict[str, Any]]] = {}

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents with embeddings to the knowledge base.

        Each document consists of text, its embedding, and optional metadata.
        Embeddings are compressed using BitPolar quantization for storage
        efficiency.

        Args:
            texts: List of document text strings.
            embeddings: List of float vectors, one per document.
            metadatas: Optional list of metadata dicts, one per document.
                Defaults to empty dicts if not provided.
            ids: Optional list of unique IDs. Auto-generated UUIDs if not
                provided.

        Returns:
            List of string IDs assigned to the added documents.

        Raises:
            ValueError: If texts and embeddings have different lengths, or
                if embedding dimension doesn't match the configured dim.
        """
        if not texts or not embeddings:
            raise ValueError("texts and embeddings cannot be empty")

        if len(texts) != len(embeddings):
            raise ValueError(
                f"texts ({len(texts)}) and embeddings ({len(embeddings)}) must have same length"
            )

        n = len(texts)
        if n == 0:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        if metadatas is None:
            metadatas = [{} for _ in range(n)]

        for i in range(n):
            vec = np.array(embeddings[i], dtype=np.float32)
            if len(vec) != self._dim:
                raise ValueError(
                    f"Dimension mismatch: expected {self._dim}, got {len(vec)}"
                )
            code = self._quantizer.encode(vec)
            self._entries[ids[i]] = (texts[i], code, metadatas[i])

        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for the most similar documents by compressed inner product.

        Computes the inner product between the query vector and all stored
        compressed codes, returning the top-k most similar documents.

        Args:
            query_embedding: Float query vector.
            top_k: Maximum number of results to return (default 10).

        Returns:
            List of result dicts sorted by descending score. Each dict has
            keys: 'id', 'text', 'score', 'metadata'.
        """
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        if not self._entries:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        if len(q) != self._dim:
            raise ValueError(
                f"Dimension mismatch: expected {self._dim}, got {len(q)}"
            )

        entry_ids = list(self._entries.keys())
        scores = np.empty(len(entry_ids), dtype=np.float32)
        for i, eid in enumerate(entry_ids):
            _text, code, _meta = self._entries[eid]
            scores[i] = self._quantizer.inner_product(code, q)

        k = min(top_k, len(entry_ids))
        if k < len(entry_ids):
            top_idx = np.argpartition(scores, -k)[-k:]
        else:
            top_idx = np.arange(len(entry_ids))
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results: List[Dict[str, Any]] = []
        for idx in top_idx[:k]:
            eid = entry_ids[idx]
            text, _code, metadata = self._entries[eid]
            results.append(
                {
                    "id": eid,
                    "text": text,
                    "score": float(round(scores[idx], 6)),
                    "metadata": metadata,
                }
            )
        return results

    def delete(self, ids: List[str]) -> None:
        """Delete documents by their IDs.

        Removes the specified documents from the knowledge base. IDs that
        are not found in the store are silently ignored.

        Args:
            ids: List of document IDs to delete.
        """
        for doc_id in ids:
            self._entries.pop(doc_id, None)

    def clear(self) -> None:
        """Remove all documents from the knowledge base.

        Resets the internal storage to an empty state. The quantizer is
        preserved so new documents must match the original dimension.
        """
        self._entries.clear()

    def count(self) -> int:
        """Return the number of documents in the knowledge base.

        Returns:
            Integer count of stored documents.
        """
        return len(self._entries)

    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single document by ID.

        Args:
            id: The document ID to look up.

        Returns:
            Dict with 'id', 'text', 'metadata' keys, or None if not found.
        """
        entry = self._entries.get(id)
        if entry is None:
            return None
        text, _code, metadata = entry
        return {
            "id": id,
            "text": text,
            "metadata": metadata,
        }
