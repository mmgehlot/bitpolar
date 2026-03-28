"""SQLite-vec integration with BitPolar vector compression.

Stores compressed BitPolar codes as BLOBs in SQLite for
lightweight, dependency-free vector storage using the stdlib sqlite3 module.
"""

from __future__ import annotations

import json
import re
import sqlite3
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


class BitPolarSQLiteStore:
    """SQLite vector store with BitPolar compression.

    Stores compressed codes as BLOBs in SQLite alongside JSON metadata.
    Uses only the stdlib sqlite3 module with no additional dependencies.
    Search is performed client-side using BitPolar inner product scoring.

    Args:
        db_path: SQLite database path (default ":memory:")
        table: Table name (default "vectors")
        dim: Vector dimension (default 384)
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim // 4)
        seed: Random seed
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        table: str = "vectors",
        dim: int = 384,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        self._table = _sanitize_table_name(table)
        self._dim = dim
        self._bits = bits
        self._seed = seed
        self._proj = projections or max(dim // 4, 1)
        self._quantizer: Optional[_bp.TurboQuantizer] = None

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id TEXT PRIMARY KEY,
                code BLOB NOT NULL,
                metadata TEXT DEFAULT '{{}}'
            )
        """)
        self._conn.commit()

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
        code_bytes = bytes(code)
        meta_json = json.dumps(metadata) if metadata else "{}"

        self._conn.execute(
            f"""
            INSERT OR REPLACE INTO {self._table} (id, code, metadata)
            VALUES (?, ?, ?)
            """,
            (id, code_bytes, meta_json),
        )
        self._conn.commit()

    def search(
        self,
        query: List[float],
        top_k: int = 10,
    ) -> list[dict]:
        """Search for similar vectors using BitPolar compressed scoring.

        Fetches all stored codes from SQLite and scores them
        client-side. Suitable for collections up to ~100K vectors.

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

        cursor = self._conn.execute(
            f"SELECT id, code, metadata FROM {self._table}"
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        scored: list[dict] = []
        for row_id, code_blob, meta_json in rows:
            code = np.frombuffer(code_blob, dtype=np.uint8).copy()
            score = float(quantizer.inner_product(code, q_vec))
            try:
                metadata = json.loads(meta_json) if meta_json else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            scored.append({
                "id": row_id,
                "score": score,
                "metadata": metadata,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete(self, id: str) -> None:
        """Delete a vector by ID.

        Args:
            id: The vector ID to delete
        """
        self._conn.execute(
            f"DELETE FROM {self._table} WHERE id = ?", (id,)
        )
        self._conn.commit()

    def count(self) -> int:
        """Return the number of stored vectors."""
        cursor = self._conn.execute(
            f"SELECT COUNT(*) FROM {self._table}"
        )
        result = cursor.fetchone()
        return result[0] if result else 0

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
