"""Vercel AI SDK middleware for BitPolar embedding compression.

Provides a Python-side middleware class that compresses embedding vectors
from any AI SDK provider response before storage or wire transfer,
and decompresses them back to float32 for downstream use.

For the TypeScript/WASM client-side integration, use the @bitpolar/wasm
npm package which provides browser-native compression via WebAssembly.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar is required. Install with: pip install bitpolar")


def _validate_bits(bits: int) -> None:
    """Validate quantization bit-width is in the supported range."""
    if not (3 <= bits <= 8):
        raise ValueError(f"bits must be 3-8, got {bits}")


class BitPolarMiddleware:
    """Middleware for compressing AI SDK embedding responses.

    Sits between an AI embedding provider and storage/transport layer,
    compressing float32 embeddings to compact BitPolar codes and
    decompressing them on retrieval.

    Supports any provider that returns embeddings as lists of floats
    (OpenAI, Cohere, Anthropic, Bedrock, etc.).

    Args:
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed for deterministic compression

    Example:
        >>> from bitpolar_vercel import BitPolarMiddleware
        >>> mw = BitPolarMiddleware(bits=4)
        >>> response = {"embedding": [0.1, 0.2, ...], "model": "text-embedding-3-small"}
        >>> compressed = mw.compress_request(response)
        >>> restored = mw.decompress_response(compressed)
    """

    def __init__(
        self,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
    ):
        _validate_bits(bits)
        self._bits = bits
        self._projections = projections
        self._seed = seed
        self._quantizers: Dict[int, "_bp.TurboQuantizer"] = {}

    def _get_quantizer(self, dim: int) -> "_bp.TurboQuantizer":
        """Get or create a TurboQuantizer for the given dimension."""
        if dim not in self._quantizers:
            proj = self._projections or max(dim // 4, 1)
            self._quantizers[dim] = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )
        return self._quantizers[dim]

    def _compress_vector(self, vector: List[float]) -> Dict[str, Any]:
        """Compress a single float vector to a serializable dict."""
        if not vector:
            raise ValueError("Empty vector")
        arr = np.array(vector, dtype=np.float32)
        dim = len(arr)
        q = self._get_quantizer(dim)
        code = q.encode(arr)
        return {
            "code": base64.b64encode(code.tobytes()).decode("ascii"),
            "dim": dim,
            "bits": self._bits,
        }

    def _decompress_entry(self, entry: Dict[str, Any]) -> List[float]:
        """Decompress a single entry back to float list."""
        code_bytes = base64.b64decode(entry["code"])
        code = np.frombuffer(code_bytes, dtype=np.uint8)
        dim = entry["dim"]
        q = self._get_quantizer(dim)
        return q.decode(code).tolist()

    def compress_request(
        self,
        embedding_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compress embeddings from an AI SDK provider response.

        Handles multiple response formats:
        - Single embedding: {"embedding": [float, ...]}
        - Batch embeddings: {"embeddings": [[float, ...], ...]}
        - OpenAI format: {"data": [{"embedding": [float, ...], "index": 0}, ...]}

        Args:
            embedding_response: Raw response dict from an embedding provider

        Returns:
            Dict with compressed codes replacing float arrays.
            Preserves all non-embedding fields (model, usage, etc.).

        Raises:
            ValueError: If no recognized embedding field is found
        """
        result = dict(embedding_response)
        result["_bitpolar"] = {
            "bits": self._bits,
            "version": "0.2.0",
        }

        # Single embedding
        if "embedding" in result and isinstance(result["embedding"], list):
            result["embedding"] = self._compress_vector(result["embedding"])
            return result

        # Batch embeddings
        if "embeddings" in result and isinstance(result["embeddings"], list):
            compressed = []
            for emb in result["embeddings"]:
                if isinstance(emb, list):
                    compressed.append(self._compress_vector(emb))
                else:
                    compressed.append(emb)
            result["embeddings"] = compressed
            return result

        # OpenAI format: {"data": [{"embedding": [...], ...}, ...]}
        if "data" in result and isinstance(result["data"], list):
            compressed_data = []
            for item in result["data"]:
                item_copy = dict(item)
                if "embedding" in item_copy and isinstance(item_copy["embedding"], list):
                    item_copy["embedding"] = self._compress_vector(
                        item_copy["embedding"]
                    )
                compressed_data.append(item_copy)
            result["data"] = compressed_data
            return result

        raise ValueError(
            "No recognized embedding field found. Expected 'embedding', "
            "'embeddings', or 'data[].embedding'."
        )

    def decompress_response(
        self,
        compressed: Union[Dict[str, Any], List[float]],
    ) -> Union[List[float], List[List[float]], Dict[str, Any]]:
        """Decompress a compressed embedding response.

        Handles the same formats produced by :meth:`compress_request`.

        Args:
            compressed: Compressed response dict from compress_request,
                        or a single compressed entry dict

        Returns:
            Decompressed embeddings in their original format.
            - Single embedding: list of floats
            - Batch: list of float lists
            - OpenAI format: dict with restored float embeddings
        """
        if isinstance(compressed, list):
            # Already decompressed
            return compressed

        # Single compressed entry
        if "code" in compressed and "dim" in compressed:
            return self._decompress_entry(compressed)

        result = dict(compressed)

        # Single embedding
        if "embedding" in result and isinstance(result["embedding"], dict):
            if "code" in result["embedding"]:
                return self._decompress_entry(result["embedding"])

        # Batch embeddings
        if "embeddings" in result and isinstance(result["embeddings"], list):
            decompressed = []
            for entry in result["embeddings"]:
                if isinstance(entry, dict) and "code" in entry:
                    decompressed.append(self._decompress_entry(entry))
                else:
                    decompressed.append(entry)
            return decompressed

        # OpenAI format
        if "data" in result and isinstance(result["data"], list):
            for item in result["data"]:
                if (
                    isinstance(item, dict)
                    and "embedding" in item
                    and isinstance(item["embedding"], dict)
                    and "code" in item["embedding"]
                ):
                    item["embedding"] = self._decompress_entry(item["embedding"])
            result.pop("_bitpolar", None)
            return result

        return result

    def compress_batch(
        self,
        vectors: List[List[float]],
    ) -> List[Dict[str, Any]]:
        """Compress a batch of raw float vectors.

        Convenience method for compressing vectors outside of a provider
        response format.

        Args:
            vectors: List of float vectors (all same dimension)

        Returns:
            List of compressed entry dicts
        """
        return [self._compress_vector(v) for v in vectors]

    def decompress_batch(
        self,
        entries: List[Dict[str, Any]],
    ) -> List[List[float]]:
        """Decompress a batch of compressed entries.

        Args:
            entries: List of compressed entry dicts from compress_batch

        Returns:
            List of float vectors
        """
        return [self._decompress_entry(e) for e in entries]

    def to_json(self, compressed: Dict[str, Any]) -> str:
        """Serialize compressed response to JSON string.

        Args:
            compressed: Compressed response from compress_request

        Returns:
            JSON string
        """
        return json.dumps(compressed)

    def from_json(self, data: str) -> Dict[str, Any]:
        """Deserialize compressed response from JSON string.

        Args:
            data: JSON string from to_json

        Returns:
            Compressed response dict
        """
        return json.loads(data)

    @property
    def bits(self) -> int:
        """Quantization bit-width."""
        return self._bits

    def __repr__(self) -> str:
        return (
            f"BitPolarMiddleware(bits={self._bits}, "
            f"projections={self._projections}, seed={self._seed})"
        )
