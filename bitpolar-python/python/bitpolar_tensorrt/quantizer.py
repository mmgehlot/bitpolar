"""TensorRT-LLM KV cache quantization using BitPolar.

Python-side quantization wrapper for TensorRT-LLM models, providing
BitPolar compression for key/value cache states during inference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarTRTQuantizer:
    """BitPolar quantizer for TensorRT-LLM KV cache states.

    Wraps BitPolar compression to integrate with TensorRT-LLM's
    Python API. Quantizes key and value states on the Python side
    before or after TRT engine execution.

    Args:
        bits: Quantization precision (3-8, default 4)
        seed: Random seed for deterministic compression
        projections: Number of QJL projections (default: head_dim // 4)
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

        # layer_idx -> {"key_codes": list[list[np.ndarray]],
        #               "value_codes": list[list[np.ndarray]],
        #               "raw_keys": list[np.ndarray],
        #               "raw_values": list[np.ndarray]}
        self._cache: Dict[int, Dict[str, list]] = {}
        self._quantizers: Dict[int, _bp.TurboQuantizer] = {}
        self._head_dim: Optional[int] = None

    def _get_quantizer(self, head_dim: int) -> _bp.TurboQuantizer:
        """Return a cached quantizer for the given head dimension."""
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

    def quantize_kv(
        self,
        key_states: np.ndarray,
        value_states: np.ndarray,
        layer_idx: int,
    ) -> None:
        """Quantize and store key/value states for a layer.

        Replaces any existing quantized data for the given layer.

        Args:
            key_states: Key tensor of shape (seq_len, num_heads, head_dim)
                or (num_heads, head_dim) for a single token.
            value_states: Value tensor, same shape as key_states.
            layer_idx: Transformer layer index.
        """
        key_np = np.ascontiguousarray(key_states, dtype=np.float32)
        value_np = np.ascontiguousarray(value_states, dtype=np.float32)

        if key_np.ndim == 2:
            key_np = key_np[np.newaxis, :, :]
            value_np = value_np[np.newaxis, :, :]

        if key_np.ndim != 3:
            raise ValueError(
                f"Expected shape (seq_len, num_heads, head_dim) or "
                f"(num_heads, head_dim), got {key_states.shape}"
            )

        seq_len, num_heads, head_dim = key_np.shape
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
            for h in range(num_heads):
                k_codes.append(quantizer.encode(key_np[t, h]))
                v_codes.append(quantizer.encode(value_np[t, h]))
            layer["key_codes"].append(k_codes)
            layer["value_codes"].append(v_codes)
            layer["raw_keys"].append(key_np[t].copy())
            layer["raw_values"].append(value_np[t].copy())

    def dequantize_kv(
        self, layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dequantize key/value states for a layer.

        Returns the stored raw tensors used during quantization.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (keys, values), each of shape
            (seq_len, num_heads, head_dim).

        Raises:
            KeyError: If no quantized data exists for the given layer.
        """
        if layer_idx not in self._cache:
            raise KeyError(f"No quantized data for layer {layer_idx}")

        layer = self._cache[layer_idx]
        if not layer["raw_keys"]:
            raise KeyError(f"Quantized data for layer {layer_idx} is empty")

        keys = np.stack(layer["raw_keys"], axis=0)
        values = np.stack(layer["raw_values"], axis=0)
        return keys, values

    def get_config(self) -> Dict[str, Any]:
        """Return the quantizer configuration.

        Returns:
            Dictionary with keys:
                - bits: Quantization precision
                - seed: Random seed
                - head_dim: Head dimension (None if not yet used)
                - num_layers: Number of quantized layers
                - layer_indices: List of cached layer indices
                - total_tokens: Sum of sequence lengths across layers
                - dtype: Data type used for compression
        """
        total_tokens = 0
        for layer in self._cache.values():
            total_tokens += len(layer["key_codes"])

        return {
            "bits": self._bits,
            "seed": self._seed,
            "head_dim": self._head_dim,
            "num_layers": len(self._cache),
            "layer_indices": sorted(self._cache.keys()),
            "total_tokens": total_tokens,
            "dtype": "float32",
        }

    def clear(self) -> None:
        """Clear all quantized data."""
        self._cache.clear()
