"""Supabase integration with BitPolar vector compression.

Compresses vectors client-side before storing in Supabase pgvector,
enabling two-phase retrieval with BitPolar re-ranking.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarSupabaseClient:
    """Supabase vector client with BitPolar compression.

    Stores original vectors in a Supabase pgvector column and
    base64-encoded BitPolar codes in a text column for client-side
    re-ranking after pgvector retrieval.

    Expects a table with columns:
        - id (text, primary key)
        - embedding (vector)
        - bp_code (text)
        - metadata (jsonb)

    Create the table with:
        CREATE TABLE vectors (
            id TEXT PRIMARY KEY,
            embedding VECTOR(384),
            bp_code TEXT,
            metadata JSONB DEFAULT '{}'::jsonb
        );

    And a similarity search function:
        CREATE OR REPLACE FUNCTION match_vectors(
            query_embedding VECTOR(384),
            match_count INT DEFAULT 10
        ) RETURNS TABLE (id TEXT, embedding VECTOR(384), bp_code TEXT, metadata JSONB, similarity FLOAT)
        LANGUAGE plpgsql AS $$
        BEGIN
            RETURN QUERY
            SELECT v.id, v.embedding, v.bp_code, v.metadata,
                   1 - (v.embedding <=> query_embedding) AS similarity
            FROM vectors v
            ORDER BY v.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;

    Args:
        url: Supabase project URL
        key: Supabase API key (anon or service role)
        table: Table name (default "vectors")
        dim: Vector dimension (default 384)
        bits: Quantization precision (3-8, default 4)
        oversampling: Oversampling factor for two-phase search (default 3)
        projections: Number of QJL projections (default: dim // 4)
        seed: Random seed
    """

    def __init__(
        self,
        url: str,
        key: str,
        table: str = "vectors",
        dim: int = 384,
        bits: int = 4,
        oversampling: int = 3,
        projections: Optional[int] = None,
        seed: int = 42,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        try:
            from supabase import create_client
        except ImportError:
            raise ImportError("supabase required. Install with: pip install supabase")

        self._table = table
        self._dim = dim
        self._bits = bits
        self._oversampling = oversampling
        self._seed = seed
        self._proj = projections or max(dim // 4, 1)
        self._quantizer: Optional[_bp.TurboQuantizer] = None

        self._client = create_client(url, key)

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a vector with optional metadata.

        Compresses the vector and stores both the original and
        compressed code in Supabase.

        Args:
            id: Unique string ID
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
        code_b64 = base64.b64encode(bytes(code)).decode("ascii")

        row = {
            "id": id,
            "embedding": vector,
            "bp_code": code_b64,
            "metadata": metadata or {},
        }

        self._client.table(self._table).upsert(row).execute()

    def search(
        self,
        query: List[float],
        top_k: int = 10,
        rerank: bool = True,
    ) -> list[dict]:
        """Two-phase search: Supabase pgvector retrieval + BitPolar re-ranking.

        Calls the match_vectors RPC function for initial retrieval,
        then optionally re-ranks using BitPolar compressed scores.

        Args:
            query: Query embedding vector
            top_k: Number of final results
            rerank: Whether to apply BitPolar re-ranking (default True)

        Returns:
            List of dicts with 'id', 'score', and 'metadata' keys
        """
        if len(query) != self._dim:
            raise ValueError(f"Vector dimension {len(query)} != expected {self._dim}")

        n_candidates = top_k * self._oversampling if rerank else top_k

        # Call the pgvector similarity search RPC
        result = self._client.rpc(
            "match_vectors",
            {
                "query_embedding": query,
                "match_count": n_candidates,
            },
        ).execute()

        rows = result.data if result.data else []
        if not rows:
            return []

        candidates: list[dict] = []
        for row in rows:
            entry: dict[str, Any] = {
                "id": row["id"],
                "score": row.get("similarity", 0.0),
                "metadata": row.get("metadata", {}),
                "_bp_code": row.get("bp_code", ""),
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
        """Delete a vector by ID.

        Args:
            id: The vector ID to delete
        """
        self._client.table(self._table).delete().eq("id", id).execute()
