"""BitPolar PyTorch PT2E Integration — native quantization backend for PyTorch 2 Export.

Provides a PT2E-compatible quantizer that can be registered as a torch.ao backend
for compressing embedding and linear layer weights using BitPolar quantization.

Usage:
    >>> from bitpolar_torch_native import BitPolarQuantizer
    >>> quantizer = BitPolarQuantizer(bits=4)
    >>> quantizer.annotate(model)
    >>> quantized_model = quantizer.quantize(model)
"""

from bitpolar_torch_native.backend import BitPolarQuantizer

__all__ = ["BitPolarQuantizer"]
__version__ = "0.3.3"
