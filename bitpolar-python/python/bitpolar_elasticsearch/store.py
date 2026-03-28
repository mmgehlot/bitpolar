"""Elasticsearch integration with BitPolar client-side compression.

Provides two-phase search: Elasticsearch knn retrieval followed by
BitPolar compressed inner product re-ranking for improved accuracy.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarElasticsearchStore:
    """Elasticsearch store with BitPolar client-side compression.

    Phase 1: Elasticsearch knn retrieval (oversampled)
    Phase 2: BitPolar compressed inner product re-ranking

    Args:
        index_name: Elasticsearch index name
        dim: Vector dimension
        bits: Quantization precision (3-8, default 4)
        oversampling: Oversampling factor for two-phase search (default 3)
        projections: Number of QJL projections (default: dim // 4)
        seed: Random seed
        es_url: Elasticsearch URL (default "http://localhost:9200")
        es_client: Optional pre-configured Elasticsearch client
    """

    def __init__(
        self,
        index_name: str,
        dim: int,
        bits: int = 4,
        oversampling: int = 3,
        projections: Optional[int] = None,
        seed: int = 42,
        es_url: str = "http://localhost:9200",
        es_client: Optional[Any] = None,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError(
                "elasticsearch required. Install with: pip install elasticsearch"
            )

        self._index_name = index_name
        self._dim = dim
        self._bits = bits
        self._oversampling = oversampling
        self._seed = seed
        self._proj = projections or max(dim // 4, 1)
        self._quantizer: Optional[_bp.TurboQuantizer] = None

        self._es = es_client or Elasticsearch(es_url)

        # Create index with dense_vector mapping if it doesn't exist
        if not self._es.indices.exists(index=index_name):
            self._es.indices.create(
                index=index_name,
                body={
                    "mappings": {
                        "properties": {
                            "vector": {
                                "type": "dense_vector",
                                "dims": dim,
                                "index": True,
                                "similarity": "dot_product",
                            },
                            "bp_code": {"type": "keyword", "index": False},
                            "text": {"type": "text"},
                            "metadata": {"type": "object", "enabled": False},
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    },
                },
            )

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

    def add(
        self,
        id: str,
        vector: List[float],
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a vector with optional text and metadata.

        Args:
            id: Unique string ID
            vector: Embedding vector
            text: Optional text content for full-text search
            metadata: Optional metadata dict
        """
        if not vector:
            raise ValueError("Empty vector not allowed")
        if len(vector) != self._dim:
            raise ValueError(f"Vector dimension {len(vector)} != expected {self._dim}")

        quantizer = self._ensure_quantizer()
        vec_np = np.array(vector, dtype=np.float32)
        code = quantizer.encode(vec_np)
        code_b64 = base64.b64encode(bytes(code)).decode("ascii")

        doc: dict[str, Any] = {
            "vector": vector,
            "bp_code": code_b64,
        }
        if text is not None:
            doc["text"] = text
        if metadata:
            doc["metadata"] = metadata

        self._es.index(index=self._index_name, id=id, body=doc)

    def search(
        self,
        query: List[float],
        top_k: int = 10,
        rerank: bool = True,
    ) -> list[dict]:
        """Two-phase search: ES knn retrieval + BitPolar re-ranking.

        Args:
            query: Query embedding vector
            top_k: Number of final results
            rerank: Whether to apply BitPolar re-ranking (default True)

        Returns:
            List of dicts with 'id', 'score', 'text', 'metadata' keys
        """
        if len(query) != self._dim:
            raise ValueError(f"Vector dimension {len(query)} != expected {self._dim}")

        n_candidates = top_k * self._oversampling if rerank else top_k

        # Phase 1: Elasticsearch knn search
        response = self._es.search(
            index=self._index_name,
            body={
                "size": n_candidates,
                "knn": {
                    "field": "vector",
                    "query_vector": query,
                    "k": n_candidates,
                    "num_candidates": n_candidates * 2,
                },
                "_source": ["bp_code", "text", "metadata"],
            },
        )

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            return []

        candidates: list[dict] = []
        for hit in hits:
            source = hit.get("_source", {})
            entry: dict[str, Any] = {
                "id": hit["_id"],
                "score": hit.get("_score", 0.0),
                "text": source.get("text"),
                "metadata": source.get("metadata", {}),
                "_bp_code": source.get("bp_code", ""),
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

    def delete(self, id: str) -> None:
        """Delete a document by ID.

        Args:
            id: Document ID to delete
        """
        self._es.delete(index=self._index_name, id=id, ignore=[404])

    def count(self) -> int:
        """Return the number of documents in the index."""
        self._es.indices.refresh(index=self._index_name)
        result = self._es.count(index=self._index_name)
        return result.get("count", 0)
