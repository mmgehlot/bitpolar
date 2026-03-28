"""BitPolar Triton Inference Server Integration — backend and client.

Provides a Triton Python backend for serving BitPolar quantization as an
inference model, plus a client wrapper for submitting compress/search requests.

Usage:
    >>> from bitpolar_triton import BitPolarTritonBackend, BitPolarTritonClient
    >>> client = BitPolarTritonClient(url="localhost:8000", model_name="bitpolar")
    >>> result = client.compress([0.1, 0.2, 0.3])
"""

from bitpolar_triton.backend import BitPolarTritonBackend, BitPolarTritonClient

__all__ = ["BitPolarTritonBackend", "BitPolarTritonClient"]
__version__ = "0.3.3"
