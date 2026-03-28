"""HuggingFace Transformers DynamicCache replacement using BitPolar.

Drop-in replacement for `transformers.DynamicCache` that compresses
Key-Value tensors using BitPolar quantization. Reduces KV cache memory
by 2-4x, enabling longer context lengths on the same hardware.

Usage:
    >>> from bitpolar_vllm.dynamic_cache import BitPolarDynamicCache
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> cache = BitPolarDynamicCache(bits=4, head_dim=128)
    >>> # Pass to model.generate() as past_key_values
    >>> outputs = model.generate(input_ids, past_key_values=cache)

Standalone usage (without transformers):
    >>> cache = BitPolarDynamicCache(bits=4, head_dim=128, num_heads=32)
    >>> cache.update(key_states, value_states, layer_idx=0)
    >>> cached_keys, cached_values = cache[0]
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarDynamicCache:
    """KV cache with BitPolar compression for transformer attention.

    Stores compressed Key and Value tensors per layer, decompressing
    on demand for attention computation. Compatible with HuggingFace
    Transformers' cache interface.

    Args:
        bits: Quantization precision (3-8, default 4)
        head_dim: Dimension of each attention head
        num_heads: Number of attention heads (auto-detected if None)
        num_layers: Number of transformer layers (auto-detected if None)
        seed: Base random seed
    """

    def __init__(
        self,
        bits: int = 4,
        head_dim: int = 128,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        seed: int = 42,
        max_seq_len: Optional[int] = None,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._bits = bits
        self._head_dim = head_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._seed = seed
        self._max_seq_len = max_seq_len

        # Per-layer storage: list of (compressed_keys, compressed_values)
        # Each entry is a list of per-position compressed codes
        self._cache: dict[int, dict] = {}
        # Quantizer per (layer, head) for isolation
        self._quantizers: dict[Tuple[int, int], _bp.TurboQuantizer] = {}

    def _get_quantizer(self, layer_idx: int, head_idx: int) -> _bp.TurboQuantizer:
        """Get or create a quantizer for a specific layer and head."""
        key = (layer_idx, head_idx)
        if key not in self._quantizers:
            # Unique seed per layer+head for isolation
            q_seed = self._seed + layer_idx * 1000 + head_idx
            proj = max(self._head_dim // 4, 1)
            self._quantizers[key] = _bp.TurboQuantizer(
                dim=self._head_dim,
                bits=self._bits,
                projections=proj,
                seed=q_seed,
            )
        return self._quantizers[key]

    def update(
        self,
        key_states: np.ndarray,
        value_states: np.ndarray,
        layer_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add new key/value states to the cache for a layer.

        Args:
            key_states: float32 array of shape (batch, num_heads, seq_len, head_dim)
                        or (num_heads, head_dim) for single token
            value_states: Same shape as key_states
            layer_idx: Transformer layer index

        Returns:
            Tuple of (all_keys, all_values) decompressed for attention
        """
        if layer_idx not in self._cache:
            self._cache[layer_idx] = {"keys": [], "values": []}

        # Handle different input shapes
        if key_states.ndim == 2:
            # (num_heads, head_dim) — single token
            num_heads = key_states.shape[0]
            keys_to_compress = [key_states]
            values_to_compress = [value_states]
        elif key_states.ndim == 4:
            # (batch, num_heads, seq_len, head_dim) — batch of tokens
            num_heads = key_states.shape[1]
            seq_len = key_states.shape[2]
            keys_to_compress = [key_states[0, :, t, :] for t in range(seq_len)]
            values_to_compress = [value_states[0, :, t, :] for t in range(seq_len)]
        else:
            raise ValueError(f"Unsupported key_states shape: {key_states.shape}")

        if self._num_heads is None:
            self._num_heads = num_heads

        # Evict oldest positions if over max_seq_len
        if self._max_seq_len is not None:
            current_len = len(self._cache[layer_idx]["keys"])
            incoming = len(keys_to_compress)
            overflow = (current_len + incoming) - self._max_seq_len
            if overflow > 0:
                self._cache[layer_idx]["keys"] = self._cache[layer_idx]["keys"][overflow:]
                self._cache[layer_idx]["values"] = self._cache[layer_idx]["values"][overflow:]

        # Compress each new position
        for k_pos, v_pos in zip(keys_to_compress, values_to_compress):
            k_codes = []
            v_codes = []
            for h in range(num_heads):
                q = self._get_quantizer(layer_idx, h)
                k_vec = np.ascontiguousarray(k_pos[h], dtype=np.float32)
                v_vec = np.ascontiguousarray(v_pos[h], dtype=np.float32)
                k_codes.append(q.encode(k_vec))
                v_codes.append(q.encode(v_vec))
            self._cache[layer_idx]["keys"].append(k_codes)
            self._cache[layer_idx]["values"].append(v_codes)

        # Decompress all cached keys/values for this layer
        return self._decompress_layer(layer_idx, num_heads)

    def _decompress_layer(
        self, layer_idx: int, num_heads: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompress all cached K/V for a layer.

        Returns (keys, values) each of shape (1, num_heads, seq_len, head_dim).
        """
        layer = self._cache[layer_idx]
        seq_len = len(layer["keys"])

        keys = np.zeros((1, num_heads, seq_len, self._head_dim), dtype=np.float32)
        values = np.zeros((1, num_heads, seq_len, self._head_dim), dtype=np.float32)

        for t in range(seq_len):
            for h in range(num_heads):
                q = self._get_quantizer(layer_idx, h)
                keys[0, h, t, :] = q.decode(layer["keys"][t][h])
                values[0, h, t, :] = q.decode(layer["values"][t][h])

        return keys, values

    def clear(self) -> None:
        """Clear all cached K/V states. Call between generation batches."""
        self._cache.clear()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current sequence length for a layer."""
        if layer_idx in self._cache:
            return len(self._cache[layer_idx]["keys"])
        return 0

    def get_max_length(self) -> Optional[int]:
        """Return None (no maximum — grows dynamically)."""
        return None

    def __getitem__(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get decompressed K/V for a layer."""
        if layer_idx not in self._cache:
            raise KeyError(f"Layer {layer_idx} not in cache")
        if self._num_heads is None:
            raise RuntimeError("Cache not initialized — call update() first")
        num_heads = self._num_heads
        return self._decompress_layer(layer_idx, num_heads)

    def __len__(self) -> int:
        """Number of layers in the cache."""
        return len(self._cache)

    def memory_stats(self) -> dict:
        """Return memory usage statistics."""
        total_positions = sum(
            len(layer["keys"]) for layer in self._cache.values()
        )
        num_heads = self._num_heads or 0
        original = total_positions * num_heads * self._head_dim * 4 * 2  # K+V, float32
        compressed = sum(
            sum(len(code) for codes in layer["keys"] for code in codes) +
            sum(len(code) for codes in layer["values"] for code in codes)
            for layer in self._cache.values()
        )
        return {
            "layers": len(self._cache),
            "total_positions": total_positions,
            "original_bytes": original,
            "compressed_bytes": compressed,
            "ratio": original / max(compressed, 1),
            "savings_pct": (1 - compressed / max(original, 1)) * 100,
        }

    def __repr__(self) -> str:
        stats = self.memory_stats()
        return (
            f"BitPolarDynamicCache(layers={stats['layers']}, "
            f"positions={stats['total_positions']}, "
            f"ratio={stats['ratio']:.1f}x)"
        )
