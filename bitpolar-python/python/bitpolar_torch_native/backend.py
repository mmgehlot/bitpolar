"""PyTorch PT2E quantization backend for BitPolar.

Implements a PT2E-compatible quantizer that integrates with PyTorch 2's
export-based quantization flow (torch.ao.quantization.quantizer).

Annotates embedding and linear layers for BitPolar compression and applies
the quantization during the prepare/convert workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch is required. Install with: pip install torch>=2.1"
    )

try:
    from torch.ao.quantization.quantizer import Quantizer
except ImportError:
    # Fallback: define a base class so the module loads even without
    # the full torch.ao quantization API (e.g. older PyTorch versions)
    Quantizer = object  # type: ignore[assignment,misc]

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar is required. Install with: pip install bitpolar")


def _validate_bits(bits: int) -> None:
    """Validate quantization bit-width is in the supported range."""
    if not (3 <= bits <= 8):
        raise ValueError(f"bits must be 3-8, got {bits}")


_SUPPORTED_OPS: List[str] = [
    "torch.nn.Linear",
    "torch.nn.Embedding",
    "torch.nn.EmbeddingBag",
]


class _CompressedWeight:
    """Container for a BitPolar-compressed weight matrix.

    Stores compressed codes alongside the quantizer needed for
    decompression and inner product scoring.
    """

    __slots__ = ("codes", "quantizer", "shape", "bits")

    def __init__(
        self,
        codes: List[np.ndarray],
        quantizer: "_bp.TurboQuantizer",
        shape: Tuple[int, ...],
        bits: int,
    ):
        self.codes = codes
        self.quantizer = quantizer
        self.shape = shape
        self.bits = bits

    def decompress(self) -> np.ndarray:
        """Decompress all codes back to float32 matrix."""
        rows = [self.quantizer.decode(c) for c in self.codes]
        return np.array(rows, dtype=np.float32).reshape(self.shape)


class _BitPolarCompressedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with BitPolar-compressed weights.

    The weight matrix is stored in compressed form. Forward computes
    approximate x @ W^T using BitPolar asymmetric inner products.
    """

    def __init__(
        self,
        compressed: _CompressedWeight,
        bias: Optional[torch.Tensor],
    ):
        super().__init__()
        self.in_features = compressed.shape[1]
        self.out_features = compressed.shape[0]
        self._compressed = compressed
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute approximate x @ W^T + bias."""
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        x_np = x_flat.detach().cpu().numpy().astype(np.float32)

        n = x_np.shape[0]
        output = np.empty((n, self.out_features), dtype=np.float32)

        q = self._compressed.quantizer
        codes = self._compressed.codes
        for i in range(n):
            for j, code in enumerate(codes):
                output[i, j] = q.inner_product(code, x_np[i])

        result = torch.from_numpy(output).to(x.device, x.dtype)
        result = result.reshape(*orig_shape, self.out_features)

        if self.bias is not None:
            result = result + self.bias
        return result

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self._compressed.bits}, bias={self.bias is not None}"
        )


class _BitPolarCompressedEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding with BitPolar-compressed weights.

    Embedding vectors are stored compressed. Lookup decompresses on-the-fly
    and scoring uses asymmetric inner product.
    """

    def __init__(
        self,
        compressed: _CompressedWeight,
        padding_idx: Optional[int],
    ):
        super().__init__()
        self.num_embeddings = compressed.shape[0]
        self.embedding_dim = compressed.shape[1]
        self._compressed = compressed
        self.padding_idx = padding_idx

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up embeddings by index, decompressing on-the-fly."""
        idx_flat = indices.reshape(-1).tolist()
        q = self._compressed.quantizer
        codes = self._compressed.codes

        vecs = np.empty((len(idx_flat), self.embedding_dim), dtype=np.float32)
        for i, idx in enumerate(idx_flat):
            if idx < 0 or idx >= self.num_embeddings:
                raise IndexError(
                    f"Embedding index {idx} out of range [0, {self.num_embeddings})"
                )
            vecs[i] = q.decode(codes[idx])

        result = torch.from_numpy(vecs).to(indices.device)
        return result.reshape(*indices.shape, self.embedding_dim)

    def score(self, indices: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Score embeddings at given indices against a query vector.

        Args:
            indices: Integer tensor of embedding indices
            query: float32 tensor of shape (embedding_dim,)

        Returns:
            float32 tensor of scores, same shape as indices
        """
        idx_flat = indices.reshape(-1).tolist()
        q_np = query.detach().cpu().numpy().astype(np.float32)
        q = self._compressed.quantizer
        codes = self._compressed.codes

        scores = np.empty(len(idx_flat), dtype=np.float32)
        for i, idx in enumerate(idx_flat):
            scores[i] = q.inner_product(codes[idx], q_np)

        return torch.from_numpy(scores).reshape(indices.shape).to(indices.device)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"bits={self._compressed.bits}"
        )


class BitPolarQuantizer(Quantizer):
    """PT2E-compatible quantizer for BitPolar compression.

    Integrates with PyTorch 2's export quantization workflow to compress
    embedding tables and linear layer weights using BitPolar's near-optimal
    vector quantization.

    Args:
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed for deterministic compression
        target_ops: Set of op names to quantize. Default: Linear + Embedding.

    Example:
        >>> import torch
        >>> from bitpolar_torch_native import BitPolarQuantizer
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(384, 128),
        ...     torch.nn.ReLU(),
        ...     torch.nn.Linear(128, 10),
        ... )
        >>> quantizer = BitPolarQuantizer(bits=4)
        >>> quantizer.annotate(model)
        >>> quantized = quantizer.quantize(model)
    """

    def __init__(
        self,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
        target_ops: Optional[Set[str]] = None,
    ):
        _validate_bits(bits)
        self._bits = bits
        self._projections = projections
        self._seed = seed
        self._target_ops = target_ops or {
            "torch.nn.Linear",
            "torch.nn.Embedding",
        }
        self._annotations: Dict[str, Dict[str, Any]] = {}
        self._quantizers: Dict[int, "_bp.TurboQuantizer"] = {}

    def _get_quantizer(self, dim: int) -> "_bp.TurboQuantizer":
        """Get or create a TurboQuantizer for the given dimension."""
        if dim not in self._quantizers:
            proj = self._projections or max(dim // 4, 1)
            self._quantizers[dim] = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )
        return self._quantizers[dim]

    def _compress_weight(self, weight: torch.Tensor) -> _CompressedWeight:
        """Compress a 2D weight tensor to BitPolar codes."""
        data = weight.detach().cpu().numpy().astype(np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D weight, got {data.ndim}D")

        n, dim = data.shape
        q = self._get_quantizer(dim)
        codes = [q.encode(data[i]) for i in range(n)]
        return _CompressedWeight(codes=codes, quantizer=q, shape=(n, dim), bits=self._bits)

    def annotate(self, model: nn.Module) -> nn.Module:
        """Mark layers in the model for BitPolar quantization.

        Walks the module tree and tags eligible layers (Linear, Embedding)
        with quantization metadata. Call this before :meth:`quantize`.

        Args:
            model: PyTorch model to annotate

        Returns:
            The same model (modified in-place with annotation attributes)
        """
        self._annotations.clear()

        for name, module in model.named_modules():
            module_type = f"{type(module).__module__}.{type(module).__qualname__}"
            # Normalize common module paths
            normalized = module_type.replace("torch.nn.modules.linear.", "torch.nn.")
            normalized = normalized.replace("torch.nn.modules.sparse.", "torch.nn.")
            normalized = normalized.replace("torch.nn.modules.container.", "torch.nn.")

            if any(op in normalized for op in self._target_ops):
                self._annotations[name] = {
                    "type": normalized,
                    "bits": self._bits,
                    "quantize": True,
                }
                # Tag the module so quantize() can identify it
                module._bitpolar_annotated = True  # type: ignore[attr-defined]

        return model

    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply BitPolar compression to all annotated layers.

        Replaces annotated Linear and Embedding layers with compressed
        equivalents that use BitPolar's asymmetric inner product for
        inference.

        Args:
            model: Annotated PyTorch model (call :meth:`annotate` first)

        Returns:
            Model with compressed layers replacing annotated ones
        """
        replacements: Dict[str, nn.Module] = {}

        for name, module in model.named_modules():
            if not getattr(module, "_bitpolar_annotated", False):
                continue

            if isinstance(module, nn.Linear):
                compressed = self._compress_weight(module.weight)
                bias = module.bias.detach().clone() if module.bias is not None else None
                replacements[name] = _BitPolarCompressedLinear(compressed, bias)

            elif isinstance(module, nn.Embedding):
                compressed = self._compress_weight(module.weight)
                padding_idx = module.padding_idx
                replacements[name] = _BitPolarCompressedEmbedding(compressed, padding_idx)

        # Apply replacements to the model
        for name, replacement in replacements.items():
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], replacement)

        return model

    def get_supported_ops(self) -> List[str]:
        """Return list of operation types supported by this quantizer.

        Returns:
            List of fully-qualified module type strings that can be quantized
        """
        return list(_SUPPORTED_OPS)

    @property
    def bits(self) -> int:
        """Quantization bit-width."""
        return self._bits

    @property
    def annotations(self) -> Dict[str, Dict[str, Any]]:
        """Map of annotated layer names to their quantization config."""
        return dict(self._annotations)

    def __repr__(self) -> str:
        return (
            f"BitPolarQuantizer(bits={self._bits}, "
            f"projections={self._projections}, seed={self._seed}, "
            f"target_ops={self._target_ops})"
        )
