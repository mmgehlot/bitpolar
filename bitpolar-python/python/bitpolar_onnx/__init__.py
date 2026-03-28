"""BitPolar ONNX Runtime quantizer.

Compresses embeddings and model weights using BitPolar quantization
for ONNX models, with support for embedding layer compression.

Usage:
    >>> from bitpolar_onnx import BitPolarONNXQuantizer
    >>> quantizer = BitPolarONNXQuantizer(bits=4)
    >>> codes = quantizer.quantize_embeddings(input_array)
    >>> quantizer.compress_model_embeddings("model.onnx", "compressed.onnx")
"""

from bitpolar_onnx.quantizer import BitPolarONNXQuantizer

__all__ = ["BitPolarONNXQuantizer"]
__version__ = "0.3.3"
