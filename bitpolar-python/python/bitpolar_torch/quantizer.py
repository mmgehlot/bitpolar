"""PyTorch integration for BitPolar vector quantization.

Provides torch.nn.Module wrappers and functional APIs for compressing
embedding matrices, linear layer weights, and KV caches using BitPolar's
near-optimal quantization.

Compatible with torch.compile() and HuggingFace Transformers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("torch required. Install with: pip install torch>=2.0")

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


def _validate_bits(bits: int) -> None:
    """Validate quantization bit-width is in the supported range."""
    if not (3 <= bits <= 8):
        raise ValueError(f"bits must be 3-8, got {bits}")


class BitPolarEmbeddingQuantizer:
    """Quantizes embedding matrices using BitPolar compression.

    Wraps a set of embedding vectors in compressed form and provides
    approximate inner product scoring without full decompression.

    Args:
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed for deterministic compression
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
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._codes: list[np.ndarray] = []
        self._dim: Optional[int] = None

    def _ensure_quantizer(self, dim: int) -> None:
        """Lazily initialize the quantizer on first use."""
        if self._quantizer is None:
            self._dim = dim
            proj = self._projections or max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )

    def compress(self, embeddings: torch.Tensor) -> list[np.ndarray]:
        """Compress a batch of embedding vectors.

        Args:
            embeddings: float32 tensor of shape (n, dim)

        Returns:
            List of compressed codes (uint8 numpy arrays)
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {embeddings.ndim}D")

        data = embeddings.detach().cpu().numpy().astype(np.float32)
        n, dim = data.shape
        self._ensure_quantizer(dim)

        codes = []
        for i in range(n):
            codes.append(self._quantizer.encode(data[i]))
        self._codes.extend(codes)
        return codes

    def search(
        self, query: torch.Tensor, codes: Optional[list[np.ndarray]] = None, top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search compressed embeddings by approximate inner product.

        Args:
            query: float32 tensor of shape (dim,) or (nq, dim)
            codes: List of compressed codes to search. If None, uses stored codes.
            top_k: Number of results per query

        Returns:
            Tuple of (indices, scores) tensors
        """
        search_codes = codes if codes is not None else self._codes
        if not search_codes:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.float32)

        if query.ndim == 1:
            query = query.unsqueeze(0)

        q_np = query.detach().cpu().numpy().astype(np.float32)
        nq = q_np.shape[0]
        k = min(top_k, len(search_codes))

        all_indices = []
        all_scores = []

        for qi in range(nq):
            scores = np.empty(len(search_codes), dtype=np.float32)
            for j, code in enumerate(search_codes):
                scores[j] = self._quantizer.inner_product(code, q_np[qi])

            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            all_indices.append(torch.from_numpy(top_idx.astype(np.int64)))
            all_scores.append(torch.from_numpy(scores[top_idx].astype(np.float32)))

        return torch.stack(all_indices), torch.stack(all_scores)

    def decompress(self, codes: list[np.ndarray]) -> torch.Tensor:
        """Decompress codes back to approximate float32 tensors."""
        vecs = [self._quantizer.decode(code) for code in codes]
        return torch.from_numpy(np.array(vecs, dtype=np.float32))


def quantize_embedding(
    embeddings: torch.Tensor,
    bits: int = 4,
    projections: Optional[int] = None,
    seed: int = 42,
) -> BitPolarEmbeddingQuantizer:
    """One-liner to compress an embedding matrix with BitPolar.

    Args:
        embeddings: float32 tensor of shape (n, dim)
        bits: Quantization precision (3-8)
        projections: QJL projections (default: dim/4)
        seed: Random seed

    Returns:
        BitPolarEmbeddingQuantizer with compressed codes stored

    Example:
        >>> emb = torch.randn(1000, 384)
        >>> q = quantize_embedding(emb, bits=4)
        >>> indices, scores = q.search(emb[0], top_k=10)
    """
    q = BitPolarEmbeddingQuantizer(bits=bits, projections=projections, seed=seed)
    q.compress(embeddings)
    return q


class BitPolarLinear(nn.Module):
    """Linear layer with BitPolar-compressed weight matrix.

    Stores the weight matrix in compressed form and computes approximate
    matrix-vector products using BitPolar's asymmetric inner product.
    Useful for embedding lookup layers and projection heads.

    Args:
        in_features: Input dimension
        out_features: Output dimension (number of compressed vectors)
        bits: Quantization precision (3-8)
        seed: Random seed
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        _validate_bits(bits)
        self.in_features = in_features
        self.out_features = out_features
        self._bits = bits
        self._seed = seed

        proj = max(in_features // 4, 1)
        self._quantizer = _bp.TurboQuantizer(
            dim=in_features, bits=bits, projections=proj, seed=seed
        )
        self._codes: list[np.ndarray] = []

        # Initialize with random weights and compress
        weight = torch.randn(out_features, in_features)
        self.set_weight(weight)

    def set_weight(self, weight: torch.Tensor) -> None:
        """Replace the compressed weight matrix.

        Args:
            weight: float32 tensor of shape (out_features, in_features)
        """
        data = weight.detach().cpu().numpy().astype(np.float32)
        self._codes = [self._quantizer.encode(data[i]) for i in range(data.shape[0])]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute approximate x @ W^T using compressed inner products.

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        x_np = x_flat.detach().cpu().numpy().astype(np.float32)

        n = x_np.shape[0]
        output = np.empty((n, self.out_features), dtype=np.float32)

        for i in range(n):
            for j, code in enumerate(self._codes):
                output[i, j] = self._quantizer.inner_product(code, x_np[i])

        result = torch.from_numpy(output)
        return result.reshape(*orig_shape, self.out_features).to(x.device)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self._bits}"
        )


class BitPolarKVCache:
    """KV cache compression for PyTorch transformer models.

    Drop-in replacement for storing Key-Value tensors in compressed form.
    Compatible with HuggingFace Transformers attention layers.

    Args:
        bits: Quantization precision (3-8)
        seed: Base random seed
    """

    def __init__(self, bits: int = 4, seed: int = 42):
        _validate_bits(bits)
        self._bits = bits
        self._seed = seed
        self._quantizers: dict[Tuple[int, int], _bp.TurboQuantizer] = {}
        # Per-layer cache: layer_idx -> list of (key_codes, value_codes) per position
        self._cache: dict[int, list[Tuple[list, list]]] = {}

    def _get_quantizer(self, layer_idx: int, head_idx: int, head_dim: int) -> _bp.TurboQuantizer:
        """Get or create quantizer for a specific layer and head."""
        key = (layer_idx, head_idx)
        if key not in self._quantizers:
            proj = max(head_dim // 4, 1)
            self._quantizers[key] = _bp.TurboQuantizer(
                dim=head_dim,
                bits=self._bits,
                projections=proj,
                seed=self._seed + layer_idx * 1000 + head_idx,
            )
        return self._quantizers[key]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Compress and cache new key/value states.

        Args:
            key_states: (batch, num_heads, seq_len, head_dim) or (num_heads, head_dim)
            value_states: Same shape as key_states
            layer_idx: Transformer layer index
        """
        if layer_idx not in self._cache:
            self._cache[layer_idx] = []

        k_np = key_states.detach().cpu().numpy().astype(np.float32)
        v_np = value_states.detach().cpu().numpy().astype(np.float32)

        # Handle different input shapes
        if k_np.ndim == 2:
            # (num_heads, head_dim) — single token
            num_heads, head_dim = k_np.shape
            positions = [(k_np, v_np)]
        elif k_np.ndim == 4:
            # (batch, num_heads, seq_len, head_dim)
            num_heads = k_np.shape[1]
            head_dim = k_np.shape[3]
            seq_len = k_np.shape[2]
            positions = [(k_np[0, :, t, :], v_np[0, :, t, :]) for t in range(seq_len)]
        else:
            raise ValueError(f"Unsupported shape: {key_states.shape}")

        for k_pos, v_pos in positions:
            k_codes = []
            v_codes = []
            for h in range(num_heads):
                q = self._get_quantizer(layer_idx, h, head_dim)
                k_codes.append(q.encode(np.ascontiguousarray(k_pos[h])))
                v_codes.append(q.encode(np.ascontiguousarray(v_pos[h])))
            self._cache[layer_idx].append((k_codes, v_codes, head_dim))

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress all cached K/V for a layer.

        Returns:
            (keys, values) each of shape (1, num_heads, seq_len, head_dim)
        """
        if layer_idx not in self._cache or not self._cache[layer_idx]:
            raise KeyError(f"No cache for layer {layer_idx}")

        entries = self._cache[layer_idx]
        num_heads = len(entries[0][0])
        seq_len = len(entries)

        # Read head_dim from stored entry
        head_dim = entries[0][2]

        keys = np.zeros((1, num_heads, seq_len, head_dim), dtype=np.float32)
        values = np.zeros((1, num_heads, seq_len, head_dim), dtype=np.float32)

        for t, (k_codes, v_codes, _hd) in enumerate(entries):
            for h in range(num_heads):
                q = self._get_quantizer(layer_idx, h, head_dim)
                keys[0, h, t, :] = q.decode(k_codes[h])
                values[0, h, t, :] = q.decode(v_codes[h])

        return torch.from_numpy(keys), torch.from_numpy(values)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def seq_length(self) -> int:
        """Current sequence length (from first layer)."""
        if not self._cache:
            return 0
        first_layer = next(iter(self._cache.values()))
        return len(first_layer)


def quantize_kv_cache(bits: int = 4, seed: int = 42) -> BitPolarKVCache:
    """Create a BitPolar KV cache quantizer for transformer models.

    Usage with HuggingFace:
        >>> cache = quantize_kv_cache(bits=4)
        >>> # In attention forward:
        >>> cache.update(key_states, value_states, layer_idx=0)
        >>> cached_keys, cached_values = cache.get(layer_idx=0)
    """
    return BitPolarKVCache(bits=bits, seed=seed)
