"""SGLang KV cache compression using BitPolar quantization.

Optimized for SGLang's RadixAttention patterns, supporting
prefix-sharing and incremental token append operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarSGLangCache:
    """Compressed KV cache for SGLang inference.

    Designed to work with SGLang's RadixAttention pattern where
    prefix KV states are shared across requests. Stores compressed
    codes alongside raw tensors for accurate decompression.

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
        #               "raw_values": list[np.ndarray],
        #               "prefix_len": int}
        self._cache: Dict[int, Dict[str, object]] = {}
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

    def _ensure_layer(self, layer_idx: int) -> dict:
        """Ensure cache storage exists for a layer."""
        if layer_idx not in self._cache:
            self._cache[layer_idx] = {
                "key_codes": [],
                "value_codes": [],
                "raw_keys": [],
                "raw_values": [],
                "prefix_len": 0,
            }
        return self._cache[layer_idx]

    def _compress_tensor(
        self, tensor: np.ndarray, quantizer: _bp.TurboQuantizer
    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        """Compress a (seq_len, num_heads, head_dim) tensor.

        Returns:
            Tuple of (codes_per_step, raw_per_step)
        """
        codes_list: List[List[np.ndarray]] = []
        raw_list: List[np.ndarray] = []
        for t in range(tensor.shape[0]):
            head_codes = []
            for h in range(tensor.shape[1]):
                head_codes.append(quantizer.encode(tensor[t, h]))
            codes_list.append(head_codes)
            raw_list.append(tensor[t].copy())
        return codes_list, raw_list

    def update(
        self,
        key_tensor: np.ndarray,
        value_tensor: np.ndarray,
        layer_idx: int,
    ) -> None:
        """Append new key/value tokens to the layer cache.

        Supports incremental decoding by appending to existing cache.
        For RadixAttention prefix sharing, call ``mark_prefix`` after
        populating the shared prefix tokens.

        Args:
            key_tensor: Key states of shape (num_heads, head_dim) or
                (seq_len, num_heads, head_dim).
            value_tensor: Value states, same shape as key_tensor.
            layer_idx: Transformer layer index.
        """
        key_np = np.ascontiguousarray(key_tensor, dtype=np.float32)
        value_np = np.ascontiguousarray(value_tensor, dtype=np.float32)

        if key_np.ndim == 2:
            key_np = key_np[np.newaxis, :, :]
            value_np = value_np[np.newaxis, :, :]

        if key_np.ndim != 3:
            raise ValueError(
                f"Expected shape (seq_len, num_heads, head_dim) or "
                f"(num_heads, head_dim), got {key_tensor.shape}"
            )

        _, _, head_dim = key_np.shape
        self._head_dim = head_dim
        quantizer = self._get_quantizer(head_dim)
        layer = self._ensure_layer(layer_idx)

        k_codes, k_raw = self._compress_tensor(key_np, quantizer)
        v_codes, v_raw = self._compress_tensor(value_np, quantizer)

        layer["key_codes"].extend(k_codes)
        layer["value_codes"].extend(v_codes)
        layer["raw_keys"].extend(k_raw)
        layer["raw_values"].extend(v_raw)

    def get(
        self, layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve decompressed key/value tensors for a layer.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (keys, values), each of shape
            (seq_len, num_heads, head_dim).

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

    def mark_prefix(self, layer_idx: int) -> None:
        """Mark the current cache length as the shared prefix boundary.

        Tokens up to this point are considered the shared prefix for
        RadixAttention and will not be evicted by ``evict_suffix``.

        Args:
            layer_idx: Transformer layer index.
        """
        layer = self._ensure_layer(layer_idx)
        layer["prefix_len"] = len(layer["key_codes"])

    def evict_suffix(self, layer_idx: int) -> int:
        """Evict all tokens after the marked prefix boundary.

        Useful for RadixAttention when a new request reuses the
        same prefix but has different continuation tokens.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Number of tokens evicted.
        """
        layer = self._ensure_layer(layer_idx)
        prefix_len = layer["prefix_len"]
        current_len = len(layer["key_codes"])

        evicted = current_len - prefix_len
        if evicted > 0:
            layer["key_codes"] = layer["key_codes"][:prefix_len]
            layer["value_codes"] = layer["value_codes"][:prefix_len]
            layer["raw_keys"] = layer["raw_keys"][:prefix_len]
            layer["raw_values"] = layer["raw_values"][:prefix_len]

        return max(evicted, 0)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current sequence length for a layer.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Number of cached tokens, or 0 if layer not present.
        """
        if layer_idx not in self._cache:
            return 0
        return len(self._cache[layer_idx]["key_codes"])

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
                - prefix_lengths: Dict mapping layer_idx to prefix_len
        """
        num_layers = len(self._cache)
        total_tokens = 0
        total_heads = 0
        head_dim = self._head_dim or 0
        prefix_lengths: Dict[int, int] = {}

        for idx, layer in self._cache.items():
            n_tokens = len(layer["key_codes"])
            total_tokens += n_tokens
            prefix_lengths[idx] = layer["prefix_len"]
            if n_tokens > 0:
                total_heads = len(layer["key_codes"][0])

        raw_bytes = 2 * total_tokens * total_heads * head_dim * 4
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
            "prefix_lengths": prefix_lengths,
        }
