"""BitPolar JAX/Flax Integration — near-optimal vector quantization for JAX arrays.

Provides functional compression/decompression APIs and a Flax linen Module
for compressing JAX arrays using BitPolar quantization.

Usage:
    >>> from bitpolar_jax import compress, decompress, inner_product
    >>> codes = compress(jax_array, bits=4)
    >>> reconstructed = decompress(codes, dim=384)
"""

from bitpolar_jax.quantizer import (
    BitPolarFlaxModule,
    compress,
    decompress,
    inner_product,
)

__all__ = [
    "BitPolarFlaxModule",
    "compress",
    "decompress",
    "inner_product",
]
__version__ = "0.3.0"
