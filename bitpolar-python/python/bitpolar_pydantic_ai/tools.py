"""PydanticAI typed tool functions for BitPolar vector operations.

Provides Pydantic input models and tool functions compatible with PydanticAI's
``@agent.tool`` decorator pattern. All functions operate on a module-level
``BitPolarToolServer`` instance that is lazily initialized on first use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("pydantic required. Install with: pip install pydantic")


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class CompressInput(BaseModel):
    """Input schema for vector compression.

    Attributes:
        vector: List of float values representing the vector to compress.
        bits: Quantization precision between 3 and 8 (default 4).
    """

    vector: list[float] = Field(..., description="Float vector to compress")
    bits: int = Field(default=4, ge=3, le=8, description="Quantization bits (3-8)")


class SearchInput(BaseModel):
    """Input schema for similarity search.

    Attributes:
        query: Float query vector.
        top_k: Number of results to return (default 5).
    """

    query: list[float] = Field(..., description="Float query vector")
    top_k: int = Field(default=5, ge=1, description="Number of results to return")


class AddVectorInput(BaseModel):
    """Input schema for adding a vector to the index.

    Attributes:
        id: Unique integer identifier for the vector.
        vector: Float vector to store.
    """

    id: int = Field(..., description="Unique vector identifier")
    vector: list[float] = Field(..., description="Float vector to store")


# ---------------------------------------------------------------------------
# Tool server
# ---------------------------------------------------------------------------


class BitPolarToolServer:
    """Manages a BitPolar quantizer and in-memory vector index.

    Lazily initializes the quantizer on first vector insertion or compression.
    Maintains an internal index mapping string IDs to compressed codes and
    original vectors for search and retrieval.

    Args:
        bits: Default quantization precision (3-8).
        seed: Random seed for deterministic compression.

    Example:
        >>> server = BitPolarToolServer(bits=4)
        >>> server.add("v1", [0.1, 0.2, 0.3, 0.4])
        >>> results = server.search([0.1, 0.2, 0.3, 0.4], top_k=1)
    """

    def __init__(self, bits: int = 4, seed: int = 42):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._bits = bits
        self._seed = seed
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._dim: Optional[int] = None
        self._codes: Dict[str, np.ndarray] = {}
        self._vectors: Dict[str, np.ndarray] = {}

    def _ensure_quantizer(self, dim: int) -> None:
        """Lazily create the quantizer for the given dimension.

        Args:
            dim: Vector dimensionality.

        Raises:
            ValueError: If dimension doesn't match an already-initialized quantizer.
        """
        if self._quantizer is None:
            self._dim = dim
            proj = max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )
        elif dim != self._dim:
            raise ValueError(
                f"Dimension mismatch: quantizer expects {self._dim}, got {dim}"
            )

    def compress(self, vector: list[float], bits: int = 4) -> dict:
        """Compress a vector and return metadata about the compression.

        Args:
            vector: Float vector to compress.
            bits: Quantization bits (3-8). Uses server default if a quantizer
                already exists with different bits.

        Returns:
            Dict with 'code_hex', 'bits', and 'dim' keys.
        """
        vec = np.array(vector, dtype=np.float32)
        self._ensure_quantizer(len(vec))
        code = self._quantizer.encode(vec)
        return {
            "code_hex": bytes(code).hex(),
            "bits": self._bits,
            "dim": len(vec),
        }

    def add(self, id: str, vector: list[float]) -> dict:
        """Add a vector to the index under the given ID.

        Args:
            id: Unique string identifier.
            vector: Float vector to store.

        Returns:
            Dict confirming the addition with 'status' and 'id'.
        """
        vec = np.array(vector, dtype=np.float32)
        self._ensure_quantizer(len(vec))
        self._codes[id] = self._quantizer.encode(vec)
        self._vectors[id] = vec
        return {"status": "added", "id": id}

    def search(self, query: list[float], top_k: int = 5) -> list[dict]:
        """Search the index for the most similar vectors.

        Args:
            query: Float query vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'id' and 'score', sorted by descending score.
        """
        if not self._codes:
            return []

        q = np.array(query, dtype=np.float32)
        self._ensure_quantizer(len(q))

        results: list[Dict[str, Any]] = []
        for doc_id, code in self._codes.items():
            score = float(self._quantizer.inner_product(code, q))
            results.append({"id": doc_id, "score": round(score, 6)})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def stats(self) -> dict:
        """Return index statistics.

        Returns:
            Dict with 'count', 'dimension', and 'bits' keys.
        """
        return {
            "count": len(self._codes),
            "dimension": self._dim,
            "bits": self._bits,
        }


# ---------------------------------------------------------------------------
# Module-level server (lazy singleton)
# ---------------------------------------------------------------------------

_server: Optional[BitPolarToolServer] = None


def _get_server() -> BitPolarToolServer:
    """Get or create the module-level BitPolarToolServer singleton.

    Returns:
        The shared BitPolarToolServer instance.
    """
    global _server
    if _server is None:
        _server = BitPolarToolServer()
    return _server


# ---------------------------------------------------------------------------
# Tool functions (compatible with @agent.tool decorator)
# ---------------------------------------------------------------------------


def compress(inp: CompressInput) -> dict:
    """Compress a vector using BitPolar quantization.

    Takes a ``CompressInput`` with vector and bits, returns a dict
    containing the hex-encoded compressed code and metadata.

    Args:
        inp: CompressInput with vector and bits fields.

    Returns:
        Dict with code_hex, bits, and dim.

    Example:
        >>> result = compress(CompressInput(vector=[0.1]*128, bits=4))
    """
    server = _get_server()
    return server.compress(inp.vector, bits=inp.bits)


def search(inp: SearchInput) -> dict:
    """Search for similar vectors in the BitPolar index.

    Takes a ``SearchInput`` with query vector and top_k, returns a dict
    containing a list of matches sorted by descending similarity score.

    Args:
        inp: SearchInput with query and top_k fields.

    Returns:
        Dict with 'results' key mapping to list of dicts with id and score.

    Example:
        >>> result = search(SearchInput(query=[0.1]*128, top_k=5))
    """
    server = _get_server()
    results = server.search(inp.query, top_k=inp.top_k)
    return {"results": results}


def add_vector(inp: AddVectorInput) -> dict:
    """Add a vector to the BitPolar index.

    Takes an ``AddVectorInput`` with id and vector, stores it in the
    module-level index for later search retrieval.

    Args:
        inp: AddVectorInput with id and vector fields.

    Returns:
        Dict confirming addition with status and id.

    Example:
        >>> result = add_vector(AddVectorInput(id=1, vector=[0.1]*128))
    """
    server = _get_server()
    return server.add(str(inp.id), inp.vector)


def stats() -> dict:
    """Return current index statistics.

    Returns:
        Dict with 'count', 'dimension', and 'bits' keys describing
        the current state of the module-level BitPolar index.

    Example:
        >>> info = stats()
        >>> info["count"]
        0
    """
    server = _get_server()
    return server.stats()
