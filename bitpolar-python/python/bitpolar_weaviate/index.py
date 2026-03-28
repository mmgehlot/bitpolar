"""Weaviate integration with BitPolar client-side compression.

Stores original vectors in Weaviate for HNSW retrieval and compressed
BitPolar codes in properties for efficient two-phase re-ranking.
"""

from __future__ import annotations

import base64
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarWeaviateIndex:
    """Weaviate index with BitPolar client-side compression and two-phase search.

    Phase 1: Weaviate HNSW retrieval (oversampled)
    Phase 2: BitPolar compressed inner product re-ranking

    Args:
        class_name: Weaviate class name
        dim: Vector dimension
        bits: Quantization precision (3-8, default 4)
        oversampling: Oversampling factor for two-phase search (default 3)
        projections: Number of QJL projections (default: dim // 4)
        seed: Random seed
        weaviate_url: Weaviate connection URL (default "http://localhost:8080")
        weaviate_api_key: Optional Weaviate API key
    """

    def __init__(
        self,
        class_name: str,
        dim: int,
        bits: int = 4,
        oversampling: int = 3,
        projections: Optional[int] = None,
        seed: int = 42,
        weaviate_url: str = "http://localhost:8080",
        weaviate_api_key: Optional[str] = None,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        try:
            import weaviate
            from weaviate.classes.config import Configure, DataType, Property
        except ImportError:
            raise ImportError("weaviate-client required. Install with: pip install weaviate-client")

        self._class_name = class_name
        self._dim = dim
        self._bits = bits
        self._oversampling = oversampling
        self._seed = seed
        self._proj = projections or max(dim // 4, 1)
        self._quantizer: Optional[_bp.TurboQuantizer] = None

        # Connect to Weaviate
        if weaviate_api_key:
            auth = weaviate.auth.AuthApiKey(api_key=weaviate_api_key)
            self._client = weaviate.connect_to_custom(
                http_host=weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                http_port=int(weaviate_url.split(":")[-1]) if ":" in weaviate_url.split("//")[-1] else 8080,
                http_secure=weaviate_url.startswith("https"),
                grpc_host=weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                grpc_port=50051,
                grpc_secure=weaviate_url.startswith("https"),
                auth_credentials=auth,
            )
        else:
            self._client = weaviate.connect_to_local()

        # Create collection if it doesn't exist
        if not self._client.collections.exists(class_name):
            self._client.collections.create(
                name=class_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="bp_code", data_type=DataType.TEXT),
                    Property(name="payload", data_type=DataType.TEXT),
                    Property(name="external_id", data_type=DataType.TEXT),
                ],
            )

        self._collection = self._client.collections.get(class_name)

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
        """Upsert vectors with BitPolar compression properties.

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

        import json

        quantizer = self._ensure_quantizer()

        with self._collection.batch.dynamic() as batch:
            for i, (vid, vec) in enumerate(zip(ids, vectors)):
                vec_np = np.array(vec, dtype=np.float32)
                code = quantizer.encode(vec_np)
                code_b64 = base64.b64encode(bytes(code)).decode("ascii")

                payload_json = json.dumps(payloads[i]) if payloads and i < len(payloads) and payloads[i] else "{}"

                # Use a deterministic UUID from the external ID for idempotent upserts
                weaviate_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, vid)

                batch.add_object(
                    properties={
                        "external_id": vid,
                        "bp_code": code_b64,
                        "payload": payload_json,
                    },
                    vector=vec,
                    uuid=weaviate_uuid,
                )

    def search(
        self,
        query: List[float],
        top_k: int = 10,
        rerank: bool = True,
    ) -> list[dict]:
        """Two-phase search: Weaviate HNSW retrieval + BitPolar re-ranking.

        Args:
            query: Query embedding vector
            top_k: Number of final results
            rerank: Whether to apply BitPolar re-ranking (default True)

        Returns:
            List of dicts with 'id', 'score', 'payload' keys
        """
        import json

        from weaviate.classes.query import MetadataQuery

        if len(query) != self._dim:
            raise ValueError(f"Vector dimension {len(query)} != expected {self._dim}")

        n_candidates = top_k * self._oversampling if rerank else top_k

        results = self._collection.query.near_vector(
            near_vector=query,
            limit=n_candidates,
            return_metadata=MetadataQuery(distance=True),
        )

        if not results.objects:
            return []

        candidates: list[dict] = []
        for obj in results.objects:
            props = obj.properties
            payload_str = props.get("payload", "{}")
            try:
                payload = json.loads(payload_str) if isinstance(payload_str, str) else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}

            entry: dict[str, Any] = {
                "id": props.get("external_id", str(obj.uuid)),
                "score": 1.0 - (obj.metadata.distance or 0.0),
                "bp_code": props.get("bp_code", ""),
                "payload": payload,
            }
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

        for c in candidates:
            c.pop("bp_code", None)

        return candidates[:top_k]

    def delete(self, ids: List[str]) -> None:
        """Delete vectors by external ID.

        Args:
            ids: List of external IDs to delete
        """
        for vid in ids:
            weaviate_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, vid)
            self._collection.data.delete_by_id(weaviate_uuid)

    def close(self) -> None:
        """Close the Weaviate client connection."""
        self._client.close()
