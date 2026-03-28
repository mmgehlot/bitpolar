"""OpenAI Agents SDK tool definitions and handler for BitPolar.

Provides function-calling compatible tool definitions (JSON schemas) and
a handler class for BitPolar vector quantization operations, compatible
with the OpenAI Agents SDK ``@function_tool`` registration pattern.
"""

from __future__ import annotations

import base64
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


# OpenAI function-calling compatible tool definitions
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bitpolar_compress",
            "description": (
                "Compress a float32 embedding vector to compact bytes using "
                "near-optimal quantization. No training needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Float32 embedding vector to compress",
                    },
                    "bits": {
                        "type": "integer",
                        "default": 4,
                        "description": (
                            "Quantization precision (3=max compression, "
                            "4=recommended, 8=near-lossless)"
                        ),
                    },
                },
                "required": ["vector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bitpolar_search",
            "description": (
                "Search a compressed vector index for the most similar "
                "vectors to a query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Float32 query vector",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results to return",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bitpolar_add_vector",
            "description": "Add a vector to the compressed search index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Unique vector ID"},
                    "vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Float32 vector to add",
                    },
                },
                "required": ["id", "vector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bitpolar_stats",
            "description": (
                "Get statistics about the compressed vector index "
                "(count, memory, compression ratio)."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


class BitPolarAgentTool:
    """OpenAI Agents SDK compatible tool handler for BitPolar.

    Maintains an in-memory compressed vector index and dispatches
    tool calls to the appropriate handler method.

    Compatible with the ``@function_tool`` registration pattern:
        >>> tool = BitPolarAgentTool(dim=384, bits=4)
        >>> result = tool.handle_tool_call("bitpolar_compress", {"vector": [0.1]*384})

    Args:
        dim: Vector dimension.
        bits: Quantization precision (3-8).
        seed: Random seed for reproducible quantization.
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

    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch an OpenAI function tool call to the appropriate handler.

        Args:
            tool_name: One of 'bitpolar_compress', 'bitpolar_search',
                'bitpolar_add_vector', 'bitpolar_stats'.
            arguments: Tool arguments dict matching the function schema.

        Returns:
            Result dict with the tool's output, or an error dict.
        """
        handlers: dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "bitpolar_compress": self._handle_compress,
            "bitpolar_search": self._handle_search,
            "bitpolar_add_vector": self._handle_add_vector,
            "bitpolar_stats": self._handle_stats,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(arguments)
        except Exception as e:
            return {"error": str(e)}

    def _handle_compress(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compress a vector to BitPolar compact codes.

        Args:
            args: Must contain 'vector' (list of floats), optional 'bits' (int).

        Returns:
            Dict with 'compressed' (base64), size stats, compression ratio.
        """
        vector = np.array(args["vector"], dtype=np.float32)
        bits = args.get("bits", self._bits)

        if len(vector) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(vector)}"}

        if not (3 <= bits <= 8):
            return {"error": f"bits must be 3-8, got {bits}"}

        # Use a different quantizer if bits differ from default
        if bits != self._bits:
            proj = max(self._dim // 4, 1)
            quantizer = _bp.TurboQuantizer(
                dim=self._dim, bits=bits, projections=proj, seed=self._seed
            )
        else:
            quantizer = self._ensure_quantizer()

        code = quantizer.encode(vector)
        code_b64 = base64.b64encode(bytes(code)).decode("ascii")

        return {
            "compressed": code_b64,
            "original_bytes": len(vector) * 4,
            "compressed_bytes": len(code),
            "compression_ratio": round(len(vector) * 4 / max(len(code), 1), 2),
        }

    def _handle_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search the compressed index for similar vectors.

        Args:
            args: Must contain 'query' (list of floats), optional 'top_k' (int).

        Returns:
            Dict with 'results' (list of id/score pairs) and 'total_searched'.
        """
        query = np.array(args["query"], dtype=np.float32)
        top_k = args.get("top_k", 5)

        if len(query) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(query)}"}

        quantizer = self._ensure_quantizer()

        scored: list[dict[str, Any]] = []
        for vid, code in self._index.items():
            score = quantizer.inner_product(code, query)
            scored.append({"id": vid, "score": round(float(score), 6)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"results": scored[:top_k], "total_searched": len(self._index)}

    def _handle_add_vector(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add a vector to the compressed index.

        Args:
            args: Must contain 'id' (int) and 'vector' (list of floats).

        Returns:
            Dict with 'success', 'id', and 'index_size'.
        """
        vid = args["id"]
        vector = np.array(args["vector"], dtype=np.float32)

        if len(vector) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(vector)}"}

        quantizer = self._ensure_quantizer()
        code = quantizer.encode(vector)
        self._index[vid] = code
        return {"success": True, "id": vid, "index_size": len(self._index)}

    def _handle_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return statistics about the compressed vector index.

        Args:
            args: Ignored (no parameters needed).

        Returns:
            Dict with vector_count, dimension, bits, size stats, compression_ratio.
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

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        """Return OpenAI function-calling compatible tool definitions."""
        return TOOL_DEFINITIONS


# Module-level singleton for simple usage
_default_tool: Optional[BitPolarAgentTool] = None


def handle_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    dim: int = 384,
    bits: int = 4,
) -> Dict[str, Any]:
    """Module-level convenience function for handling tool calls.

    Creates or reuses a default BitPolarAgentTool singleton.

    Args:
        tool_name: The tool name to dispatch.
        arguments: Tool arguments dict.
        dim: Vector dimension (only used on first call).
        bits: Quantization precision (only used on first call).

    Returns:
        Result dict from the tool handler.
    """
    global _default_tool
    if _default_tool is None:
        _default_tool = BitPolarAgentTool(dim=dim, bits=bits)
    return _default_tool.handle_tool_call(tool_name, arguments)
