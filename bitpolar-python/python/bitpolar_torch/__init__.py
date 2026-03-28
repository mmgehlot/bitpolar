"""BitPolar PyTorch Integration — near-optimal vector quantization for PyTorch.

Provides tensor compression, KV cache quantization, and embedding quantization
compatible with PyTorch and torchao workflows.

Usage:
    >>> from bitpolar_torch import quantize_embedding, BitPolarLinear
    >>> compressed = quantize_embedding(embedding_tensor, bits=4)
    >>> layer = BitPolarLinear(in_features=384, out_features=128, bits=4)
"""

from bitpolar_torch.quantizer import (
    BitPolarLinear,
    BitPolarEmbeddingQuantizer,
    quantize_embedding,
    quantize_kv_cache,
)

__all__ = [
    "BitPolarLinear",
    "BitPolarEmbeddingQuantizer",
    "quantize_embedding",
    "quantize_kv_cache",
]
__version__ = "0.3.1"
