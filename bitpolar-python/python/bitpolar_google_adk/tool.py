"""Google ADK tool for BitPolar vector quantization.

Provides a Google Agent Development Kit compatible tool class with
methods for compressing, searching, and managing quantized vectors.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")

# Try importing Google ADK decorators
try:
    from google.adk.tools import tool as adk_tool

    _HAS_ADK = True
except ImportError:
    _HAS_ADK = False

    def adk_tool(fn: Any) -> Any:  # type: ignore[misc]
        """No-op decorator stub when google.adk is not installed."""
        return fn


class BitPolarADKTool:
    """Google ADK tool for BitPolar vector quantization.

    Provides methods compatible with Google's Agent Development Kit
    for compressing embeddings, searching compressed indices, and
    managing vector data.

    Args:
        dim: Vector dimension.
        bits: Quantization precision (3-8).
        seed: Random seed for reproducible quantization.

    Example:
        >>> tool = BitPolarADKTool(dim=384, bits=4)
        >>> result = tool.compress(vector=[0.1]*384, bits=4)
        >>> tool.add_vector(id=1, vector=[0.1]*384)
        >>> results = tool.search(query=[0.1]*384, top_k=3)
    """

    def __init__(self, dim: int = 384, bits: int = 4, seed: int = 42) -> None:
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._dim = dim
        self._bits = bits
        self._seed = seed
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._index: dict[int, np.ndarray] = {}

    def _ensure_quantizer(self) -> _bp.TurboQuantizer:
        """Lazily initialize the quantizer.

        Returns:
            The initialized TurboQuantizer instance.
        """
        if self._quantizer is None:
            proj = max(self._dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=self._dim,
                bits=self._bits,
                projections=proj,
                seed=self._seed,
            )
        return self._quantizer

    @adk_tool
    def compress(
        self,
        vector: List[float],
        bits: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compress a float32 embedding vector using BitPolar quantization.

        Takes a full-precision embedding and returns a compact compressed
        representation with compression statistics.

        Args:
            vector: Float32 embedding vector to compress.
            bits: Optional override for quantization precision (3-8).
                Defaults to the instance's configured bits.

        Returns:
            Dict containing:
                - compressed: Base64-encoded compressed code
                - original_bytes: Size of the original vector in bytes
                - compressed_bytes: Size of the compressed code in bytes
                - compression_ratio: Ratio of original to compressed size
        """
        vec = np.array(vector, dtype=np.float32)
        effective_bits = bits if bits is not None else self._bits

        if not (3 <= effective_bits <= 8):
            return {"error": f"bits must be 3-8, got {effective_bits}"}

        if len(vec) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(vec)}"}

        # Use a different quantizer if bits differ from default
        if effective_bits != self._bits:
            proj = max(self._dim // 4, 1)
            quantizer = _bp.TurboQuantizer(
                dim=self._dim, bits=effective_bits, projections=proj, seed=self._seed
            )
        else:
            quantizer = self._ensure_quantizer()

        code = quantizer.encode(vec)
        code_b64 = base64.b64encode(bytes(code)).decode("ascii")

        return {
            "compressed": code_b64,
            "original_bytes": len(vec) * 4,
            "compressed_bytes": len(code),
            "compression_ratio": round(len(vec) * 4 / max(len(code), 1), 2),
        }

    @adk_tool
    def search(
        self,
        query: List[float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Search the compressed vector index for similar vectors.

        Computes compressed inner product between the query and all
        stored vectors, returning the top-k most similar.

        Args:
            query: Float32 query vector.
            top_k: Number of top results to return.

        Returns:
            Dict containing:
                - results: List of {id, score} dicts sorted by score
                - total_searched: Total number of vectors searched
        """
        query_vec = np.array(query, dtype=np.float32)

        if len(query_vec) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(query_vec)}"}

        quantizer = self._ensure_quantizer()

        scored: list[dict[str, Any]] = []
        for vid, code in self._index.items():
            score = quantizer.inner_product(code, query_vec)
            scored.append({"id": vid, "score": round(float(score), 6)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"results": scored[:top_k], "total_searched": len(self._index)}

    @adk_tool
    def add_vector(
        self,
        id: int,
        vector: List[float],
    ) -> Dict[str, Any]:
        """Add a vector to the compressed search index.

        Encodes the vector using BitPolar quantization and stores
        the compressed code in the index.

        Args:
            id: Unique integer identifier for the vector.
            vector: Float32 vector to compress and store.

        Returns:
            Dict with 'success', 'id', and 'index_size'.
        """
        vec = np.array(vector, dtype=np.float32)

        if len(vec) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(vec)}"}

        quantizer = self._ensure_quantizer()
        code = quantizer.encode(vec)
        self._index[id] = code
        return {"success": True, "id": id, "index_size": len(self._index)}

    @adk_tool
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the compressed vector index.

        Returns:
            Dict containing:
                - vector_count: Number of vectors in the index
                - dimension: Vector dimension
                - bits: Quantization precision
                - compressed_bytes: Total compressed storage size
                - original_bytes: Equivalent uncompressed storage size
                - compression_ratio: Ratio of original to compressed
        """
        total_bytes = sum(len(c) for c in self._index.values())
        original_bytes = len(self._index) * self._dim * 4
        return {
            "vector_count": len(self._index),
            "dimension": self._dim,
            "bits": self._bits,
            "compressed_bytes": total_bytes,
            "original_bytes": original_bytes,
            "compression_ratio": round(
                original_bytes / max(total_bytes, 1), 2
            ),
        }
