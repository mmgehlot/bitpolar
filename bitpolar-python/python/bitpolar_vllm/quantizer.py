"""Standalone KV cache quantizer using BitPolar compression.

This module works independently of vLLM and can be used in any
transformer inference pipeline to compress Key-Value caches.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError(
        "bitpolar Rust bindings required. Install with: pip install bitpolar"
    )


class KVCacheQuantizer:
    """Quantizer for transformer Key-Value caches.

    Compresses attention Key and Value tensors per-head using BitPolar's
    TurboQuantizer. Each attention head gets its own quantizer with a
    unique seed offset to prevent correlated projection matrices.

    Args:
        head_dim: Dimension of each attention head (e.g., 128 for Llama-3)
        bits: Quantization precision (3-8, default 4)
        num_heads: Number of attention heads
        seed: Base random seed (each head uses seed + head_index)
        projections: QJL projections per head (default: head_dim/4)

    Example:
        >>> kv_q = KVCacheQuantizer(head_dim=128, bits=4, num_heads=32)
        >>>
        >>> # Compress a batch of key vectors (one per head)
        >>> keys = np.random.randn(32, 128).astype(np.float32)
        >>> compressed = kv_q.compress_keys(keys)
        >>>
        >>> # Decompress for attention computation
        >>> approx_keys = kv_q.decompress_keys(compressed)
        >>> print(f"Memory: {keys.nbytes}B -> {sum(len(c) for c in compressed)}B")
    """

    def __init__(
        self,
        head_dim: int,
        bits: int = 4,
        num_heads: int = 1,
        seed: int = 42,
        projections: Optional[int] = None,
    ):
        self._head_dim = head_dim
        self._bits = bits
        self._num_heads = num_heads
        self._seed = seed
        self._projections = projections or max(head_dim // 4, 1)

        # Create one quantizer per head with offset seeds
        self._quantizers = []
        for h in range(num_heads):
            q = _bp.TurboQuantizer(
                dim=head_dim,
                bits=bits,
                projections=self._projections,
                seed=seed + h,  # per-head seed offset
            )
            self._quantizers.append(q)

    def compress_keys(self, keys: np.ndarray) -> list[np.ndarray]:
        """Compress key vectors for all heads.

        Args:
            keys: float32 array of shape (num_heads, head_dim)

        Returns:
            List of compressed codes (one per head), each a uint8 numpy array
        """
        if keys.dtype != np.float32:
            keys = keys.astype(np.float32)
        if keys.shape != (self._num_heads, self._head_dim):
            raise ValueError(
                f"Expected shape ({self._num_heads}, {self._head_dim}), "
                f"got {keys.shape}"
            )

        codes = []
        for h in range(self._num_heads):
            code = self._quantizers[h].encode(keys[h])
            codes.append(code)
        return codes

    def compress_values(self, values: np.ndarray) -> list[np.ndarray]:
        """Compress value vectors for all heads.

        Same interface as compress_keys — values use the same quantizers.

        Args:
            values: float32 array of shape (num_heads, head_dim)

        Returns:
            List of compressed codes (one per head)
        """
        return self.compress_keys(values)

    def decompress_keys(self, codes: list[np.ndarray]) -> np.ndarray:
        """Decompress key vectors for all heads.

        Args:
            codes: List of compressed codes from compress_keys

        Returns:
            float32 array of shape (num_heads, head_dim)
        """
        if len(codes) != self._num_heads:
            raise ValueError(
                f"Expected {self._num_heads} codes, got {len(codes)}"
            )

        result = np.empty(
            (self._num_heads, self._head_dim), dtype=np.float32
        )
        for h in range(self._num_heads):
            result[h] = self._quantizers[h].decode(codes[h])
        return result

    def decompress_values(self, codes: list[np.ndarray]) -> np.ndarray:
        """Decompress value vectors for all heads."""
        return self.decompress_keys(codes)

    def attention_score(
        self,
        query: np.ndarray,
        key_codes: list[np.ndarray],
    ) -> np.ndarray:
        """Compute approximate attention scores: Q · K^T (compressed).

        Uses BitPolar's asymmetric inner product estimation to compute
        attention scores without decompressing the full key cache.

        Args:
            query: float32 array of shape (num_heads, head_dim)
            key_codes: List of compressed key codes for one cached position

        Returns:
            float32 array of shape (num_heads,) — one score per head
        """
        if query.dtype != np.float32:
            query = query.astype(np.float32)

        scores = np.empty(self._num_heads, dtype=np.float32)
        scale = 1.0 / np.sqrt(self._head_dim)

        for h in range(self._num_heads):
            raw_score = self._quantizers[h].inner_product(
                key_codes[h], query[h]
            )
            scores[h] = raw_score * scale
        return scores

    @property
    def head_dim(self) -> int:
        """Attention head dimension."""
        return self._head_dim

    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self._num_heads

    @property
    def bits(self) -> int:
        """Quantization bit-width."""
        return self._bits

    def memory_savings(self, seq_len: int) -> dict:
        """Calculate memory savings for a given sequence length.

        Args:
            seq_len: Number of tokens in the KV cache

        Returns:
            Dict with original_bytes, compressed_bytes, ratio, savings_pct
        """
        # Original: num_heads × head_dim × float32 × 2 (K+V) × seq_len
        original = self._num_heads * self._head_dim * 4 * 2 * seq_len

        # Compressed: one dummy encode to get code size
        dummy = np.zeros(self._head_dim, dtype=np.float32)
        code = self._quantizers[0].encode(dummy)
        code_size = len(code)

        # Compressed: num_heads × code_size × 2 (K+V) × seq_len
        compressed = self._num_heads * code_size * 2 * seq_len

        return {
            "original_bytes": original,
            "compressed_bytes": compressed,
            "ratio": original / max(compressed, 1),
            "savings_pct": (1 - compressed / max(original, 1)) * 100,
        }

    def __repr__(self) -> str:
        return (
            f"KVCacheQuantizer(heads={self._num_heads}, "
            f"head_dim={self._head_dim}, bits={self._bits})"
        )
