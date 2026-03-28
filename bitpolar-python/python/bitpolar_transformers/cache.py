"""HuggingFace Transformers DynamicCache-compatible KV cache with BitPolar compression.

Implements the DynamicCache interface so it can be used as a drop-in
replacement for HuggingFace's default cache in generation pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")

try:
    import torch
except ImportError:
    raise ImportError("torch required. Install with: pip install torch")


class BitPolarCache:
    """DynamicCache-compatible KV cache with BitPolar compression.

    Drop-in replacement for HuggingFace's ``DynamicCache`` that
    compresses key/value states using BitPolar quantization internally
    while exposing the standard torch Tensor interface.

    Internally converts tensors to numpy for compression, stores both
    compressed codes and raw tensors, and returns torch tensors on
    decompression.

    Args:
        bits: Quantization precision (3-8, default 4)
        seed: Random seed for deterministic compression
        max_seq_len: Maximum sequence length (None for unlimited)
        projections: Number of QJL projections (default: head_dim // 4)
    """

    def __init__(
        self,
        bits: int = 4,
        seed: int = 42,
        max_seq_len: Optional[int] = None,
        projections: Optional[int] = None,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        self._bits = bits
        self._seed = seed
        self._max_seq_len = max_seq_len
        self._projections = projections

        # Per-layer storage: list index = layer_idx
        # Each entry: {"key_codes": list, "value_codes": list,
        #              "key_cache": Tensor, "value_cache": Tensor}
        self._cache: List[Optional[Dict[str, Any]]] = []
        self._quantizers: Dict[int, _bp.TurboQuantizer] = {}

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

    def _grow_cache(self, layer_idx: int) -> None:
        """Ensure the cache list is large enough for layer_idx."""
        while len(self._cache) <= layer_idx:
            self._cache.append(None)

    def _compress_heads(
        self, tensor: torch.Tensor, quantizer: _bp.TurboQuantizer
    ) -> List[List[np.ndarray]]:
        """Compress a (batch, num_heads, seq_len, head_dim) tensor per head per token.

        Returns list of seq_len entries, each a list of num_heads codes.
        """
        # tensor shape: (batch, num_heads, seq_len, head_dim)
        arr = tensor.detach().cpu().float().numpy()
        batch, num_heads, seq_len, _ = arr.shape
        # Process first batch element
        codes: List[List[np.ndarray]] = []
        for t in range(seq_len):
            head_codes = []
            for h in range(num_heads):
                head_codes.append(quantizer.encode(arr[0, h, t]))
            codes.append(head_codes)
        return codes

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key/value states.

        Implements the HuggingFace DynamicCache.update interface.

        Args:
            key_states: Key tensor of shape (batch, num_heads, seq_len, head_dim).
            value_states: Value tensor of the same shape.
            layer_idx: Transformer layer index.
            cache_kwargs: Optional extra arguments (unused, for API compat).

        Returns:
            Tuple of (full_key_states, full_value_states) containing all
            cached tokens for this layer concatenated with the new states.
        """
        self._grow_cache(layer_idx)

        device = key_states.device
        dtype = key_states.dtype
        _, _, _, head_dim = key_states.shape
        quantizer = self._get_quantizer(head_dim)

        new_key_codes = self._compress_heads(key_states, quantizer)
        new_value_codes = self._compress_heads(value_states, quantizer)

        if self._cache[layer_idx] is None:
            self._cache[layer_idx] = {
                "key_codes": new_key_codes,
                "value_codes": new_value_codes,
                "key_cache": key_states,
                "value_cache": value_states,
            }
        else:
            entry = self._cache[layer_idx]
            entry["key_codes"].extend(new_key_codes)
            entry["value_codes"].extend(new_value_codes)
            entry["key_cache"] = torch.cat(
                [entry["key_cache"], key_states], dim=2
            )
            entry["value_cache"] = torch.cat(
                [entry["value_cache"], value_states], dim=2
            )

        # Enforce max_seq_len
        if self._max_seq_len is not None:
            entry = self._cache[layer_idx]
            seq_len = entry["key_cache"].shape[2]
            if seq_len > self._max_seq_len:
                trim = seq_len - self._max_seq_len
                entry["key_codes"] = entry["key_codes"][trim:]
                entry["value_codes"] = entry["value_codes"][trim:]
                entry["key_cache"] = entry["key_cache"][:, :, trim:, :]
                entry["value_cache"] = entry["value_cache"][:, :, trim:, :]

        entry = self._cache[layer_idx]
        return entry["key_cache"].to(device=device, dtype=dtype), entry[
            "value_cache"
        ].to(device=device, dtype=dtype)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current sequence length for a layer.

        Args:
            layer_idx: Transformer layer index (default 0).

        Returns:
            Number of cached tokens, or 0 if layer not present.
        """
        if layer_idx >= len(self._cache) or self._cache[layer_idx] is None:
            return 0
        return self._cache[layer_idx]["key_cache"].shape[2]

    def get_max_length(self) -> Optional[int]:
        """Return the maximum sequence length, or None if unlimited.

        Returns:
            Maximum sequence length or None.
        """
        return self._max_seq_len

    def __getitem__(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached key/value tensors for a layer.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (key_cache, value_cache) tensors.

        Raises:
            KeyError: If no cache exists for the given layer.
        """
        if layer_idx >= len(self._cache) or self._cache[layer_idx] is None:
            raise KeyError(f"No cache for layer {layer_idx}")
        entry = self._cache[layer_idx]
        return entry["key_cache"], entry["value_cache"]

    def __len__(self) -> int:
        """Return the number of cached layers.

        Returns:
            Number of layers with cached data.
        """
        return sum(1 for entry in self._cache if entry is not None)

    def __contains__(self, layer_idx: int) -> bool:
        """Check if a layer has cached data.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            True if the layer has cached data.
        """
        return (
            0 <= layer_idx < len(self._cache)
            and self._cache[layer_idx] is not None
        )

    def __iter__(self):
        """Iterate over cached (key, value) tuples for each layer."""
        for entry in self._cache:
            if entry is not None:
                yield entry["key_cache"], entry["value_cache"]

    def clear(self) -> None:
        """Clear all cached layers."""
        self._cache.clear()
        self._quantizers.clear()

    @property
    def seen_tokens(self) -> int:
        """Total number of tokens seen across layer 0 (HF compat)."""
        return self.get_seq_length(0)
