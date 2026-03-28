"""Neon serverless Postgres integration with BitPolar vector compression.

Compresses vectors client-side before storing in Neon pgvector,
enabling efficient two-phase retrieval with BitPolar re-ranking.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, List, Optional

import numpy as np


def _sanitize_table_name(name: str) -> str:
    """Validate table name to prevent SQL injection."""
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        raise ValueError(f"Invalid table name: {name!r}. Only alphanumeric and underscore allowed.")
    return name

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarNeonClient:
    """Neon serverless Postgres client with BitPolar compression.

    Stores original vectors in a pgvector column and base64-encoded
    BitPolar codes in a text column for client-side re-ranking.

    Automatically creates the required table and pgvector extension
    on first use. Requires the pgvector extension enabled in Neon.

    Args:
        connection_string: Neon/PostgreSQL connection string
        table: Table name (default "vectors")
        dim: Vector dimension (default 384)
        bits: Quantization precision (3-8, default 4)
        oversampling: Oversampling factor for two-phase search (default 3)
        projections: Number of QJL projections (default: dim // 4)
        seed: Random seed
    """

    def __init__(
        self,
        connection_string: str,
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
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError("psycopg2 required. Install with: pip install psycopg2-binary")

        self._table = _sanitize_table_name(table)
        self._dim = dim
        self._bits = bits
        self._oversampling = oversampling
        self._seed = seed
        self._proj = projections or max(dim // 4, 1)
        self._quantizer: Optional[_bp.TurboQuantizer] = None

        self._conn = psycopg2.connect(connection_string)
        self._conn.autocommit = True

        # Ensure pgvector extension and table exist
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id TEXT PRIMARY KEY,
                    embedding vector({dim}),
                    bp_code TEXT,
                    metadata JSONB DEFAULT '{{}}'::jsonb
                )
            """)
            # Create an IVFFlat or HNSW index if it doesn't already exist
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx
                ON {self._table}
                USING hnsw (embedding vector_ip_ops)
                WITH (m = 16, ef_construction = 200)
            """)

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
        compressed code in Neon Postgres.

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

        vec_str = "[" + ",".join(str(float(v)) for v in vector) + "]"
        meta_json = json.dumps(metadata) if metadata else "{}"

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._table} (id, embedding, bp_code, metadata)
                VALUES (%s, %s::vector, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    bp_code = EXCLUDED.bp_code,
                    metadata = EXCLUDED.metadata
                """,
                (id, vec_str, code_b64, meta_json),
            )

    def search(
        self,
        query: List[float],
        top_k: int = 10,
        rerank: bool = True,
    ) -> list[dict]:
        """Two-phase search: pgvector HNSW retrieval + BitPolar re-ranking.

        Phase 1: Retrieve oversampled candidates using pgvector inner product
        Phase 2: Re-rank candidates using BitPolar compressed inner product

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
        query_str = "[" + ",".join(str(float(v)) for v in query) + "]"

        with self._conn.cursor() as cur:
            # Use inner product distance (<#>) — pgvector returns negative IP
            cur.execute(
                f"""
                SELECT id, bp_code, metadata,
                       (embedding <#> %s::vector) * -1 AS similarity
                FROM {self._table}
                ORDER BY embedding <#> %s::vector
                LIMIT %s
                """,
                (query_str, query_str, n_candidates),
            )
            rows = cur.fetchall()

        if not rows:
            return []

        candidates: list[dict] = []
        for row_id, bp_code, meta, similarity in rows:
            metadata = meta if isinstance(meta, dict) else {}
            entry: dict[str, Any] = {
                "id": row_id,
                "score": float(similarity) if similarity else 0.0,
                "metadata": metadata,
                "_bp_code": bp_code or "",
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
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE id = %s", (id,)
            )

    def count(self) -> int:
        """Return the number of stored vectors."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            result = cur.fetchone()
            return result[0] if result else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
