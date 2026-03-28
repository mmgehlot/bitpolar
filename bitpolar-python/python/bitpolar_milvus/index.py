"""Milvus integration with BitPolar client-side compression.

Stores original vectors in Milvus for HNSW retrieval and compressed
BitPolar codes in metadata for efficient two-phase re-ranking.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarMilvusIndex:
    """Milvus index with BitPolar client-side compression and two-phase search.

    Phase 1: Milvus HNSW retrieval (oversampled)
    Phase 2: BitPolar compressed inner product re-ranking

    Args:
        collection_name: Milvus collection name
        dim: Vector dimension
        bits: Quantization precision (3-8, default 4)
        oversampling: Oversampling factor for two-phase search (default 3)
        projections: Number of QJL projections (default: dim // 4)
        seed: Random seed
        milvus_uri: Milvus connection URI (default "http://localhost:19530")
    """

    def __init__(
        self,
        collection_name: str,
        dim: int,
        bits: int = 4,
        oversampling: int = 3,
        projections: Optional[int] = None,
        seed: int = 42,
        milvus_uri: str = "http://localhost:19530",
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                MilvusClient,
                connections,
                utility,
            )
        except ImportError:
            raise ImportError("pymilvus required. Install with: pip install pymilvus")

        self._collection_name = collection_name
        self._dim = dim
        self._bits = bits
        self._oversampling = oversampling
        self._seed = seed
        self._proj = projections or max(dim // 4, 1)
        self._quantizer: Optional[_bp.TurboQuantizer] = None

        # Connect and set up collection
        self._client = MilvusClient(uri=milvus_uri)

        # Check if collection exists, create if not
        if not self._client.has_collection(collection_name):
            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field("bp_code", DataType.VARCHAR, max_length=65535)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="IP",
                params={"M": 16, "efConstruction": 200},
            )
            self._client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
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

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Upsert vectors with BitPolar compression metadata.

        Stores original vectors for Milvus HNSW indexing and base64-encoded
        BitPolar codes in the bp_code field for re-ranking.

        Args:
            ids: List of unique string IDs
            vectors: List of embedding vectors
            payloads: Optional list of metadata dicts per vector
        """
        if len(ids) != len(vectors):
            raise ValueError(f"ids length {len(ids)} != vectors length {len(vectors)}")

        for vec in vectors:
            if not vec:
                raise ValueError("Empty vector not allowed")
            if len(vec) != self._dim:
                raise ValueError(f"Vector dimension {len(vec)} != expected {self._dim}")

        quantizer = self._ensure_quantizer()
        data: list[dict[str, Any]] = []

        for i, (vid, vec) in enumerate(zip(ids, vectors)):
            vec_np = np.array(vec, dtype=np.float32)
            code = quantizer.encode(vec_np)
            code_b64 = base64.b64encode(bytes(code)).decode("ascii")

            row: dict[str, Any] = {
                "id": vid,
                "vector": vec,
                "bp_code": code_b64,
            }
            if payloads and i < len(payloads) and payloads[i]:
                row.update(payloads[i])
            data.append(row)

        self._client.upsert(
            collection_name=self._collection_name,
            data=data,
        )

    def search(
        self,
        query: List[float],
        top_k: int = 10,
        rerank: bool = True,
        filter_expr: Optional[str] = None,
    ) -> list[dict]:
        """Two-phase search: Milvus HNSW retrieval + BitPolar re-ranking.

        Phase 1: Retrieve oversampled candidates from Milvus HNSW index
        Phase 2: Re-rank candidates using BitPolar compressed inner product

        Args:
            query: Query embedding vector
            top_k: Number of final results
            rerank: Whether to apply BitPolar re-ranking (default True)
            filter_expr: Optional Milvus filter expression

        Returns:
            List of dicts with 'id', 'score', and 'distance' keys plus
            any dynamic fields stored in the collection
        """
        if len(query) != self._dim:
            raise ValueError(f"Vector dimension {len(query)} != expected {self._dim}")

        n_candidates = top_k * self._oversampling if rerank else top_k

        search_params = {"metric_type": "IP", "params": {"ef": max(n_candidates * 2, 64)}}
        search_kwargs: dict[str, Any] = {
            "collection_name": self._collection_name,
            "data": [query],
            "limit": n_candidates,
            "output_fields": ["bp_code", "*"],
            "search_params": search_params,
        }
        if filter_expr:
            search_kwargs["filter"] = filter_expr

        results = self._client.search(**search_kwargs)

        if not results or not results[0]:
            return []

        candidates: list[dict] = []
        for hit in results[0]:
            entry: dict[str, Any] = {
                "id": hit["id"],
                "distance": hit["distance"],
                "score": hit["distance"],
            }
            entity = hit.get("entity", {})
            entry["bp_code"] = entity.get("bp_code", "")
            for k, v in entity.items():
                if k not in ("bp_code", "vector"):
                    entry[k] = v
            candidates.append(entry)

        # Phase 2: Re-rank with BitPolar compressed inner product
        if rerank:
            quantizer = self._ensure_quantizer()
            q_vec = np.array(query, dtype=np.float32)
            for entry in candidates:
                code_b64 = entry.get("bp_code", "")
                if code_b64:
                    try:
                        code = np.frombuffer(base64.b64decode(code_b64), dtype=np.uint8).copy()
                    except Exception as e:
                        continue  # Skip entries with corrupt codes
                    entry["score"] = float(quantizer.inner_product(code, q_vec))

            candidates.sort(key=lambda x: x["score"], reverse=True)

        # Remove internal bp_code from results
        for c in candidates:
            c.pop("bp_code", None)

        return candidates[:top_k]

    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID.

        Args:
            ids: List of vector IDs to delete
        """
        if ids:
            # Sanitize IDs: escape quotes and build proper Milvus filter expression
            sanitized = []
            for i in ids:
                s = str(i).replace("\\", "\\\\").replace("'", "\\'")
                sanitized.append(f"'{s}'")
            filter_expr = f"id in [{', '.join(sanitized)}]"
            self._client.delete(
                collection_name=self._collection_name,
                filter=filter_expr,
            )
