"""ChromaDB integration with BitPolar compression.

Provides a BitPolar-compressed embedding function for ChromaDB collections
and a high-level store wrapper with two-phase search (ChromaDB HNSW + BitPolar re-rank).
"""

from __future__ import annotations

import base64
import uuid
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarEmbeddingFunction:
    """ChromaDB-compatible embedding function that compresses output.

    Wraps any embedding model and stores both original and compressed
    embeddings for two-phase retrieval.

    Args:
        embed_fn: Callable that takes list of texts and returns list of embeddings
        bits: Quantization precision (3-8)
        seed: Random seed
    """

    def __init__(self, embed_fn: Any, bits: int = 4, seed: int = 42):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._embed_fn = embed_fn
        self._bits = bits
        self._seed = seed
        self._quantizer: Optional[_bp.TurboQuantizer] = None

    def _ensure_quantizer(self, dim: int) -> None:
        if self._quantizer is None:
            proj = max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Compute embeddings (ChromaDB embedding function interface).

        Returns original embeddings — compression happens at storage layer.
        """
        if hasattr(self._embed_fn, "embed_documents"):
            return self._embed_fn.embed_documents(input)
        elif callable(self._embed_fn):
            return self._embed_fn(input)
        raise TypeError("embed_fn must be callable or have embed_documents method")

    def compress(self, embeddings: List[List[float]]) -> List[str]:
        """Compress embeddings to base64-encoded BitPolar codes."""
        if not embeddings:
            return []
        dim = len(embeddings[0])
        self._ensure_quantizer(dim)
        codes = []
        for emb in embeddings:
            vec = np.array(emb, dtype=np.float32)
            code = self._quantizer.encode(vec)
            codes.append(base64.b64encode(bytes(code)).decode("ascii"))
        return codes

    def score(self, code_b64: str, query: List[float]) -> float:
        """Score a compressed code against a full-precision query."""
        code = np.frombuffer(base64.b64decode(code_b64), dtype=np.uint8).copy()
        q = np.array(query, dtype=np.float32)
        return float(self._quantizer.inner_product(code, q))


class BitPolarChromaStore:
    """High-level ChromaDB store with BitPolar compression.

    Wraps a ChromaDB collection and adds BitPolar compression for
    memory-efficient storage with two-phase search.

    Args:
        collection_name: ChromaDB collection name
        bits: Quantization precision (3-8)
        seed: Random seed
        chroma_client: Optional ChromaDB client (creates in-memory if None)
        embedding_function: Optional embedding function
        oversampling: Oversampling factor for two-phase search (default 3)
    """

    def __init__(
        self,
        collection_name: str = "bitpolar_default",
        bits: int = 4,
        seed: int = 42,
        chroma_client: Optional[Any] = None,
        embedding_function: Optional[Any] = None,
        oversampling: int = 3,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._bits = bits
        self._seed = seed
        self._oversampling = oversampling
        self._quantizer: Optional[_bp.TurboQuantizer] = None

        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb required. Install with: pip install chromadb")

        self._client = chroma_client or chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

    def _ensure_quantizer(self, dim: int) -> None:
        if self._quantizer is None:
            proj = max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )

    def add(
        self,
        texts: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents with optional embeddings.

        Args:
            texts: Document texts (will be embedded if embedding_function is set)
            embeddings: Pre-computed embeddings
            metadatas: Metadata dicts per document
            ids: Document IDs (auto-generated if None)

        Returns:
            List of document IDs
        """
        n = len(texts) if texts else (len(embeddings) if embeddings else 0)
        if n == 0:
            return []

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]

        # Compress embeddings and store as metadata
        add_kwargs: dict[str, Any] = {"ids": ids}
        if texts:
            add_kwargs["documents"] = texts
        if metadatas is None:
            metadatas = [{} for _ in range(n)]

        if embeddings:
            for idx, emb in enumerate(embeddings):
                if not emb:
                    raise ValueError(f"embedding at index {idx} cannot be empty")
            self._ensure_quantizer(len(embeddings[0]))
            for i, emb in enumerate(embeddings):
                vec = np.array(emb, dtype=np.float32)
                code = self._quantizer.encode(vec)
                code_b64 = base64.b64encode(bytes(code)).decode("ascii")
                metadatas[i]["_bp_code"] = code_b64
            add_kwargs["embeddings"] = embeddings

        add_kwargs["metadatas"] = metadatas
        self._collection.add(**add_kwargs)
        return ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search with optional BitPolar re-ranking.

        Phase 1: ChromaDB HNSW retrieval (oversampled)
        Phase 2: BitPolar compressed inner product re-ranking

        Args:
            query_embedding: Query vector
            top_k: Number of results
            where: ChromaDB metadata filter
            rerank: Whether to apply BitPolar re-ranking (default True)

        Returns:
            List of result dicts with 'id', 'document', 'metadata', 'score'
        """
        # Phase 1: Retrieve candidates from ChromaDB
        n_candidates = top_k * self._oversampling if rerank else top_k
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_candidates, self._collection.count() or 1),
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        if not results["ids"] or not results["ids"][0]:
            return []

        candidates = []
        for i, doc_id in enumerate(results["ids"][0]):
            entry = {
                "id": doc_id,
                "document": (results["documents"][0][i] if results.get("documents") and results["documents"][0] else None),
                "metadata": results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"][0] else {},
                "score": results["distances"][0][i] if results.get("distances") and results["distances"][0] else 0.0,
            }
            candidates.append(entry)

        # Phase 2: Re-rank with BitPolar compressed inner product
        if rerank and self._quantizer is not None:
            q_vec = np.array(query_embedding, dtype=np.float32)
            for entry in candidates:
                code_b64 = entry["metadata"].get("_bp_code")
                if code_b64:
                    code = np.frombuffer(base64.b64decode(code_b64), dtype=np.uint8).copy()
                    entry["score"] = float(self._quantizer.inner_product(code, q_vec))

            candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:top_k]

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self._collection.delete(ids=ids)

    def count(self) -> int:
        """Return number of documents in the collection."""
        return self._collection.count()
