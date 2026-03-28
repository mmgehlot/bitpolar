"""MCP tool definitions and server for BitPolar vector quantization.

Provides tool definitions compatible with the Model Context Protocol,
enabling AI agents to compress, search, and manage vector data.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


# MCP Tool Definitions — JSON-compatible descriptions for agent registration
TOOL_DEFINITIONS = [
    {
        "name": "bitpolar_compress",
        "description": "Compress a float32 embedding vector to compact bytes using near-optimal quantization. No training needed.",
        "inputSchema": {
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
                    "description": "Quantization precision (3=max compression, 4=recommended, 8=near-lossless)",
                },
            },
            "required": ["vector"],
        },
    },
    {
        "name": "bitpolar_search",
        "description": "Search a compressed vector index for the most similar vectors to a query.",
        "inputSchema": {
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
    {
        "name": "bitpolar_add_vector",
        "description": "Add a vector to the compressed search index.",
        "inputSchema": {
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
    {
        "name": "bitpolar_index_stats",
        "description": "Get statistics about the compressed vector index (count, memory, compression ratio).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class BitPolarToolServer:
    """MCP tool server for BitPolar vector quantization.

    Maintains an in-memory compressed vector index and exposes
    compress/search/add/stats operations as MCP-compatible tools.

    Args:
        dim: Vector dimension
        bits: Quantization precision (3-8)
        seed: Random seed
    """

    def __init__(self, dim: int = 384, bits: int = 4, seed: int = 42):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._dim = dim
        self._bits = bits
        proj = max(dim // 4, 1)
        self._quantizer = _bp.TurboQuantizer(
            dim=dim, bits=bits, projections=proj, seed=seed
        )
        self._index: dict[int, np.ndarray] = {}  # id → compressed code

    def handle_tool_call(self, tool_name: str, arguments: dict) -> dict:
        """Handle an MCP tool call.

        Args:
            tool_name: One of the registered tool names
            arguments: Tool arguments dict

        Returns:
            Result dict with the tool's output
        """
        handlers = {
            "bitpolar_compress": self._handle_compress,
            "bitpolar_search": self._handle_search,
            "bitpolar_add_vector": self._handle_add,
            "bitpolar_index_stats": self._handle_stats,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(arguments)
        except Exception as e:
            return {"error": str(e)}

    def _handle_compress(self, args: dict) -> dict:
        vector = np.array(args["vector"], dtype=np.float32)
        bits = args.get("bits", self._bits)
        if not (3 <= bits <= 8):
            return {"error": f"bits must be 3-8, got {bits}"}

        if len(vector) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(vector)}"}

        # Use requested bits if different from server default
        quantizer = self._quantizer
        if bits != self._bits:
            proj = max(self._dim // 4, 1)
            quantizer = _bp.TurboQuantizer(
                dim=self._dim, bits=bits, projections=proj, seed=42
            )

        code = quantizer.encode(vector)
        import base64
        code_b64 = base64.b64encode(bytes(code)).decode("ascii")

        return {
            "compressed": code_b64,
            "original_bytes": len(vector) * 4,
            "compressed_bytes": len(code),
            "compression_ratio": round(len(vector) * 4 / max(len(code), 1), 2),
        }

    def _handle_search(self, args: dict) -> dict:
        query = np.array(args["query"], dtype=np.float32)
        top_k = args.get("top_k", 5)

        if len(query) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(query)}"}

        scored = []
        for vid, code in self._index.items():
            score = self._quantizer.inner_product(code, query)
            scored.append({"id": vid, "score": round(float(score), 6)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"results": scored[:top_k], "total_searched": len(self._index)}

    def _handle_add(self, args: dict) -> dict:
        vid = args["id"]
        vector = np.array(args["vector"], dtype=np.float32)

        if len(vector) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(vector)}"}

        code = self._quantizer.encode(vector)
        self._index[vid] = code
        return {"success": True, "id": vid, "index_size": len(self._index)}

    def _handle_stats(self, _args: dict) -> dict:
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
    def tool_definitions(self) -> list[dict]:
        """Return MCP-compatible tool definitions."""
        return TOOL_DEFINITIONS
