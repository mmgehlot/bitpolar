"""BitPolar TensorFlow Integration — Keras layers for vector quantization.

Provides custom Keras layers for compressing tensors and embeddings
using BitPolar's near-optimal quantization.

Usage:
    >>> from bitpolar_tensorflow import BitPolarLayer, BitPolarEmbedding
    >>> layer = BitPolarLayer(bits=4)
    >>> compressed = layer(input_tensor)
"""

from bitpolar_tensorflow.layer import (
    BitPolarEmbedding,
    BitPolarLayer,
    compress_tensor,
    decompress_tensor,
)

__all__ = [
    "BitPolarLayer",
    "BitPolarEmbedding",
    "compress_tensor",
    "decompress_tensor",
]
__version__ = "0.3.2"
