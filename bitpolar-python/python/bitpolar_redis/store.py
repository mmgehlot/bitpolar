"""Redis integration with BitPolar vector compression.

Stores compressed BitPolar codes as Redis byte strings for
memory-efficient vector search with optional metadata.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarRedisStore:
    """Redis vector store with BitPolar compression.

    Stores compressed codes as Redis byte strings and metadata as
    JSON hashes. Search is performed client-side using BitPolar
    inner product scoring.

    Args:
        dim: Vector dimension
        bits: Quantization precision (3-8, default 4)
        prefix: Redis key prefix (default "bp:")
        projections: Number of QJL projections (default: dim // 4)
        seed: Random seed
        redis_url: Redis connection URL (default "redis://localhost:6379")
        redis_client: Optional pre-configured Redis client
    """

    def __init__(
        self,
        dim: int = 384,
        bits: int = 4,
        prefix: str = "bp:",
        projections: Optional[int] = None,
        seed: int = 42,
        redis_url: str = "redis://localhost:6379",
        redis_client: Optional[Any] = None,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        try:
            import redis as redis_lib
        except ImportError:
            raise ImportError("redis required. Install with: pip install redis")

        self._dim = dim
        self._bits = bits
        self._prefix = prefix
        self._seed = seed
        self._proj = projections or max(dim // 4, 1)
        self._quantizer: Optional[_bp.TurboQuantizer] = None

        if redis_client is not None:
            self._redis = redis_client
        else:
            self._redis = redis_lib.from_url(redis_url, decode_responses=False)

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

    def _code_key(self, id: str) -> str:
        """Return the Redis key for a vector's compressed code."""
        return f"{self._prefix}code:{id}"

    def _meta_key(self, id: str) -> str:
        """Return the Redis key for a vector's metadata."""
        return f"{self._prefix}meta:{id}"

    def _index_key(self) -> str:
        """Return the Redis key for the ID index set."""
        return f"{self._prefix}ids"

    def add(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a vector with optional metadata.

        Args:
            id: Unique string ID for the vector
            vector: Embedding vector
            metadata: Optional metadata dict
        """
        if not vector:
            raise ValueError("Empty vector not allowed")
        if len(vector) != self._dim:
            raise ValueError(f"Vector dimension {len(vector)} != expected {self._dim}")

        quantizer = self._ensure_quantizer()
        vec_np = np.array(vector, dtype=np.float32)
        code = quantizer.encode(vec_np)
        code_bytes = bytes(code)

        pipe = self._redis.pipeline()
        pipe.set(self._code_key(id), code_bytes)
        if metadata:
            pipe.set(self._meta_key(id), json.dumps(metadata).encode("utf-8"))
        else:
            pipe.set(self._meta_key(id), b"{}")
        pipe.sadd(self._index_key(), id.encode("utf-8"))
        pipe.execute()

    def search(
        self,
        query: List[float],
        top_k: int = 10,
    ) -> list[dict]:
        """Search for similar vectors using BitPolar compressed scoring.

        Scans all stored codes and scores against the query vector.
        Suitable for collections up to ~100K vectors.

        Args:
            query: Query embedding vector
            top_k: Maximum number of results

        Returns:
            List of dicts with 'id', 'score', and 'metadata' keys
        """
        if len(query) != self._dim:
            raise ValueError(f"Vector dimension {len(query)} != expected {self._dim}")

        quantizer = self._ensure_quantizer()
        q_vec = np.array(query, dtype=np.float32)

        # Get all IDs from the index set
        id_bytes = self._redis.smembers(self._index_key())
        if not id_bytes:
            return []

        all_ids = [ib.decode("utf-8") if isinstance(ib, bytes) else ib for ib in id_bytes]

        # Fetch all codes in a single pipeline
        pipe = self._redis.pipeline()
        for vid in all_ids:
            pipe.get(self._code_key(vid))
        code_results = pipe.execute()

        scored: list[dict] = []
        for vid, code_bytes in zip(all_ids, code_results):
            if code_bytes is None:
                continue
            code = np.frombuffer(code_bytes, dtype=np.uint8).copy()
            score = float(quantizer.inner_product(code, q_vec))
            scored.append({"id": vid, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored[:top_k]

        # Fetch metadata for top results
        if top_results:
            pipe = self._redis.pipeline()
            for entry in top_results:
                pipe.get(self._meta_key(entry["id"]))
            meta_results = pipe.execute()

            for entry, meta_bytes in zip(top_results, meta_results):
                if meta_bytes:
                    raw = meta_bytes.decode("utf-8") if isinstance(meta_bytes, bytes) else meta_bytes
                    try:
                        entry["metadata"] = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        entry["metadata"] = {}
                else:
                    entry["metadata"] = {}

        return top_results

    def delete(self, id: str) -> None:
        """Delete a vector by ID.

        Args:
            id: The vector ID to delete
        """
        pipe = self._redis.pipeline()
        pipe.delete(self._code_key(id))
        pipe.delete(self._meta_key(id))
        pipe.srem(self._index_key(), id.encode("utf-8"))
        pipe.execute()

    def count(self) -> int:
        """Return the number of stored vectors."""
        return self._redis.scard(self._index_key()) or 0
