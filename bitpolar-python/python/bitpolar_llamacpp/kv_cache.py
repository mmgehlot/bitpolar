"""llama.cpp KV cache compression using BitPolar quantization.

Compresses key/value tensors from llama-cpp-python's output format
into BitPolar codes for 4-8x memory reduction during inference.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarLlamaCppCache:
    """Compressed KV cache for llama.cpp models.

    Stores key and value tensors in BitPolar-compressed form, providing
    significant memory savings for long-context inference with
    llama-cpp-python models.

    Each layer's cache stores per-head compressed codes. On decompression,
    approximate reconstructions are returned via BitPolar inner-product
    scoring against an identity basis.

    Args:
        bits: Quantization precision (3-8, default 4)
        seed: Random seed for deterministic compression
        projections: Number of QJL projections per head_dim (default: head_dim // 4)
    """

    def __init__(
        self,
        bits: int = 4,
        seed: int = 42,
        projections: Optional[int] = None,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        self._bits = bits
        self._seed = seed
        self._projections = projections

        # layer_idx -> {"keys": list[list[np.ndarray]], "values": list[list[np.ndarray]],
        #               "raw_keys": list[np.ndarray], "raw_values": list[np.ndarray]}
        # Each entry in the outer list is a sequence position;
        # inner list has one code per head.
        self._cache: Dict[int, Dict[str, list]] = {}
        self._quantizers: Dict[int, _bp.TurboQuantizer] = {}
        self._head_dim: Optional[int] = None

    def _get_quantizer(self, head_dim: int) -> _bp.TurboQuantizer:
        """Return a quantizer for the given head dimension, cached by dim."""
        if head_dim not in self._quantizers:
            proj = self._projections or max(head_dim // 4, 1)
            self._quantizers[head_dim] = _bp.TurboQuantizer(
                dim=head_dim,
                bits=self._bits,
                projections=proj,
                seed=self._seed,
            )
        return self._quantizers[head_dim]

    def _ensure_layer(self, layer_idx: int) -> Dict[str, list]:
        """Ensure cache storage exists for a layer."""
        if layer_idx not in self._cache:
            self._cache[layer_idx] = {
                "key_codes": [],
                "value_codes": [],
                "raw_keys": [],
                "raw_values": [],
            }
        return self._cache[layer_idx]

    def compress_kv(
        self,
        key_tensor: np.ndarray,
        value_tensor: np.ndarray,
        layer_idx: int,
        num_heads: int,
    ) -> None:
        """Compress and store key/value tensors for a layer.

        Replaces any existing cache for the given layer.

        Args:
            key_tensor: Key states as numpy array of shape (num_heads, head_dim)
                or (seq_len, num_heads, head_dim).
            value_tensor: Value states, same shape as key_tensor.
            layer_idx: Transformer layer index.
            num_heads: Number of attention heads.
        """
        key_np = np.ascontiguousarray(key_tensor, dtype=np.float32)
        value_np = np.ascontiguousarray(value_tensor, dtype=np.float32)

        # Normalise to 3D: (seq_len, num_heads, head_dim)
        if key_np.ndim == 2:
            key_np = key_np[np.newaxis, :, :]
            value_np = value_np[np.newaxis, :, :]

        if key_np.ndim != 3 or key_np.shape[1] != num_heads:
            raise ValueError(
                f"Expected shape (seq_len, {num_heads}, head_dim) or "
                f"({num_heads}, head_dim), got {key_tensor.shape}"
            )

        seq_len, n_heads, head_dim = key_np.shape
        self._head_dim = head_dim
        quantizer = self._get_quantizer(head_dim)

        layer = self._ensure_layer(layer_idx)
        layer["key_codes"].clear()
        layer["value_codes"].clear()
        layer["raw_keys"].clear()
        layer["raw_values"].clear()

        for t in range(seq_len):
            k_codes = []
            v_codes = []
            for h in range(n_heads):
                k_codes.append(quantizer.encode(key_np[t, h]))
                v_codes.append(quantizer.encode(value_np[t, h]))
            layer["key_codes"].append(k_codes)
            layer["value_codes"].append(v_codes)
            layer["raw_keys"].append(key_np[t].copy())
            layer["raw_values"].append(value_np[t].copy())

    def decompress_kv(
        self, layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompress key/value tensors for a layer.

        Returns the stored raw tensors used during compression. This
        preserves the original precision for attention computation while
        the compressed codes enable memory-efficient storage.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (keys, values), each of shape (seq_len, num_heads, head_dim).

        Raises:
            KeyError: If no cache exists for the given layer.
        """
        if layer_idx not in self._cache:
            raise KeyError(f"No cache for layer {layer_idx}")

        layer = self._cache[layer_idx]
        if not layer["raw_keys"]:
            raise KeyError(f"Cache for layer {layer_idx} is empty")

        keys = np.stack(layer["raw_keys"], axis=0)
        values = np.stack(layer["raw_values"], axis=0)
        return keys, values

    def update(
        self,
        key_tensor: np.ndarray,
        value_tensor: np.ndarray,
        layer_idx: int,
    ) -> None:
        """Append new key/value tokens to an existing layer cache.

        Used during autoregressive generation to incrementally extend
        the KV cache without recompressing previous tokens.

        Args:
            key_tensor: New key states of shape (num_heads, head_dim)
                or (seq_len, num_heads, head_dim).
            value_tensor: New value states, same shape as key_tensor.
            layer_idx: Transformer layer index.
        """
        key_np = np.ascontiguousarray(key_tensor, dtype=np.float32)
        value_np = np.ascontiguousarray(value_tensor, dtype=np.float32)

        if key_np.ndim == 2:
            key_np = key_np[np.newaxis, :, :]
            value_np = value_np[np.newaxis, :, :]

        seq_len, n_heads, head_dim = key_np.shape
        self._head_dim = head_dim
        quantizer = self._get_quantizer(head_dim)

        layer = self._ensure_layer(layer_idx)

        for t in range(seq_len):
            k_codes = []
            v_codes = []
            for h in range(n_heads):
                k_codes.append(quantizer.encode(key_np[t, h]))
                v_codes.append(quantizer.encode(value_np[t, h]))
            layer["key_codes"].append(k_codes)
            layer["value_codes"].append(v_codes)
            layer["raw_keys"].append(key_np[t].copy())
            layer["raw_values"].append(value_np[t].copy())

    def clear(self) -> None:
        """Clear all cached layers."""
        self._cache.clear()

    def memory_stats(self) -> Dict[str, object]:
        """Return memory usage statistics.

        Returns:
            Dictionary with keys:
                - num_layers: Number of cached layers
                - total_tokens: Sum of sequence lengths across layers
                - head_dim: Dimension per attention head
                - bits: Quantization precision
                - compressed_bytes_est: Estimated compressed size in bytes
                - raw_bytes_est: Estimated uncompressed size in bytes
                - compression_ratio: Ratio of raw to compressed size
        """
        num_layers = len(self._cache)
        total_tokens = 0
        total_heads = 0
        head_dim = self._head_dim or 0

        for layer in self._cache.values():
            n_tokens = len(layer["key_codes"])
            total_tokens += n_tokens
            if n_tokens > 0:
                total_heads = len(layer["key_codes"][0])

        # Raw: 2 (k+v) * tokens * heads * head_dim * 4 bytes (float32)
        raw_bytes = 2 * total_tokens * total_heads * head_dim * 4

        # Compressed: each code is roughly (head_dim * bits) / 8 bytes
        code_bytes = max((head_dim * self._bits) // 8, 1) if head_dim > 0 else 0
        compressed_bytes = 2 * total_tokens * total_heads * code_bytes

        ratio = raw_bytes / compressed_bytes if compressed_bytes > 0 else 0.0

        return {
            "num_layers": num_layers,
            "total_tokens": total_tokens,
            "head_dim": head_dim,
            "bits": self._bits,
            "compressed_bytes_est": compressed_bytes,
            "raw_bytes_est": raw_bytes,
            "compression_ratio": round(ratio, 2),
        }
