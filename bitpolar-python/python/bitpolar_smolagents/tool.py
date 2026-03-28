"""SmolAgents tool for BitPolar vector search operations.

Implements the HuggingFace SmolAgents Tool interface for compress, search,
add, and stats actions on BitPolar-compressed vector indices.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")

try:
    from smolagents import Tool as _SmolTool

    _HAS_SMOLAGENTS = True
except ImportError:
    _HAS_SMOLAGENTS = False

    class _SmolTool:  # type: ignore[no-redef]
        """Fallback base class when smolagents is not installed."""

        name: str = ""
        description: str = ""
        inputs: dict = {}
        output_type: str = "string"


class BitPolarTool(_SmolTool):
    """SmolAgents tool for BitPolar compressed vector search.

    Provides four actions:
      - **compress**: Quantize a vector and return its compressed code as hex.
      - **add**: Add a vector with an ID to the internal index.
      - **search**: Find the top-k most similar vectors to a query.
      - **stats**: Return index statistics (count, dimension, bits).

    The tool lazily initialises the quantizer on the first vector it sees,
    inferring the dimension automatically.

    Args:
        bits: Quantization precision (3-8, default 4).
        seed: Random seed for deterministic compression.

    Example:
        >>> tool = BitPolarTool(bits=4)
        >>> tool.forward("add", vector=[0.1, 0.2, 0.3, 0.4], vector_id="doc-1")
        '{"status": "added", "id": "doc-1"}'
        >>> tool.forward("search", query=[0.1, 0.2, 0.3, 0.4], top_k=1)
        '[{"id": "doc-1", "score": 0.99}]'
    """

    name: str = "bitpolar_vector_search"
    description: str = (
        "Compress, store, and search vectors using BitPolar quantization. "
        "Supports actions: compress, add, search, stats."
    )
    inputs: Dict[str, Any] = {
        "action": {
            "type": "string",
            "description": "Action to perform: 'compress', 'add', 'search', or 'stats'.",
        },
        "vector": {
            "type": "array",
            "description": "Float vector for compress/add actions.",
            "nullable": True,
        },
        "query": {
            "type": "array",
            "description": "Float query vector for search action.",
            "nullable": True,
        },
        "vector_id": {
            "type": "string",
            "description": "Vector ID for add action.",
            "nullable": True,
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return for search (default 5).",
            "nullable": True,
        },
        "bits": {
            "type": "integer",
            "description": "Quantization bits for compress action (default 4).",
            "nullable": True,
        },
    }
    output_type: str = "string"

    def __init__(self, bits: int = 4, seed: int = 42, **kwargs: Any):
        if _HAS_SMOLAGENTS:
            super().__init__(**kwargs)
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._bits = bits
        self._seed = seed
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._dim: Optional[int] = None
        self._index: Dict[str, np.ndarray] = {}
        self._vectors: Dict[str, np.ndarray] = {}

    def _ensure_quantizer(self, dim: int) -> None:
        """Lazily create the quantizer once we know the vector dimension.

        Args:
            dim: Vector dimensionality.
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

    def _compress(self, vector: List[float]) -> str:
        """Compress a vector and return its hex-encoded code.

        Args:
            vector: Float vector to compress.

        Returns:
            JSON string with hex-encoded compressed code.
        """
        vec = np.array(vector, dtype=np.float32)
        self._ensure_quantizer(len(vec))
        code = self._quantizer.encode(vec)
        hex_code = bytes(code).hex()
        return json.dumps({"code": hex_code, "bits": self._bits, "dim": len(vec)})

    def _add(self, vector_id: str, vector: List[float]) -> str:
        """Add a vector with an ID to the internal index.

        Args:
            vector_id: Unique string identifier for the vector.
            vector: Float vector to store.

        Returns:
            JSON string confirming the addition.
        """
        vec = np.array(vector, dtype=np.float32)
        self._ensure_quantizer(len(vec))
        code = self._quantizer.encode(vec)
        self._index[vector_id] = code
        self._vectors[vector_id] = vec
        return json.dumps({"status": "added", "id": vector_id})

    def _search(self, query: List[float], top_k: int = 5) -> str:
        """Search the index for the top-k most similar vectors.

        Args:
            query: Float query vector.
            top_k: Number of results to return.

        Returns:
            JSON array of results with id and score, sorted by descending score.
        """
        if not self._index:
            return json.dumps([])

        q = np.array(query, dtype=np.float32)
        self._ensure_quantizer(len(q))

        scores: List[Dict[str, Any]] = []
        for doc_id, code in self._index.items():
            score = float(self._quantizer.inner_product(code, q))
            scores.append({"id": doc_id, "score": round(score, 6)})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return json.dumps(scores[:top_k])

    def _stats(self) -> str:
        """Return index statistics.

        Returns:
            JSON string with count, dimension, and bits.
        """
        return json.dumps(
            {
                "count": len(self._index),
                "dimension": self._dim,
                "bits": self._bits,
            }
        )

    def forward(
        self,
        action: str,
        vector: Optional[List[float]] = None,
        query: Optional[List[float]] = None,
        top_k: int = 5,
        bits: int = 4,
        vector_id: Optional[str] = None,
    ) -> str:
        """Dispatch to the appropriate action handler.

        This is the primary SmolAgents entry point. Each action maps to an
        internal method that performs the corresponding vector operation.

        Args:
            action: One of 'compress', 'add', 'search', 'stats'.
            vector: Float vector for compress/add actions.
            query: Float query vector for search action.
            top_k: Number of results to return for search (default 5).
            bits: Quantization bits for compress (default 4, uses instance default).
            vector_id: Unique ID for add action.

        Returns:
            JSON-encoded string result.

        Raises:
            ValueError: If the action is unknown or required arguments are missing.
        """
        if action == "compress":
            if vector is None:
                raise ValueError("compress action requires 'vector' argument")
            return self._compress(vector)

        elif action == "add":
            if vector_id is None:
                raise ValueError("add action requires 'vector_id' argument")
            if vector is None:
                raise ValueError("add action requires 'vector' argument")
            return self._add(vector_id, vector)

        elif action == "search":
            if query is None:
                raise ValueError("search action requires 'query' argument")
            return self._search(query, top_k=top_k)

        elif action == "stats":
            return self._stats()

        else:
            raise ValueError(
                f"Unknown action '{action}'. Must be one of: compress, add, search, stats"
            )

    def __call__(self, action: str, **kwargs: Any) -> str:
        """Dispatch to forward with keyword arguments.

        Provides backwards-compatible callable interface that delegates to
        ``forward()`` for SmolAgents compatibility.

        Args:
            action: One of 'compress', 'add', 'search', 'stats'.
            **kwargs: Action-specific keyword arguments.

        Returns:
            JSON-encoded string result.
        """
        return self.forward(action, **kwargs)
