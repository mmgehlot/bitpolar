"""BitPolar Apple MLX quantizer.

Compresses MLX and numpy arrays using BitPolar quantization,
enabling efficient on-device inference on Apple Silicon.

Usage:
    >>> from bitpolar_mlx import BitPolarMLXQuantizer
    >>> quantizer = BitPolarMLXQuantizer(bits=4)
    >>> code = quantizer.compress(array)
    >>> score = quantizer.inner_product(code, query)
"""

from bitpolar_mlx.quantizer import BitPolarMLXQuantizer

__all__ = ["BitPolarMLXQuantizer"]
__version__ = "0.2.0"
