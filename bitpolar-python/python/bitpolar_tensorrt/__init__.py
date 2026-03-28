"""BitPolar TensorRT-LLM quantization integration.

Python-side quantization wrapper for TensorRT-LLM models, providing
BitPolar compression for KV cache states.

Usage:
    >>> from bitpolar_tensorrt import BitPolarTRTQuantizer
    >>> quantizer = BitPolarTRTQuantizer(bits=4)
    >>> quantizer.quantize_kv(key_states, value_states, layer_idx=0)
    >>> keys, values = quantizer.dequantize_kv(layer_idx=0)
"""

from bitpolar_tensorrt.quantizer import BitPolarTRTQuantizer

__all__ = ["BitPolarTRTQuantizer"]
__version__ = "0.3.1"
