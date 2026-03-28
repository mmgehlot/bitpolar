"""BitPolar llama.cpp KV cache compression.

Compresses llama-cpp-python KV cache tensors using BitPolar quantization
for 4-8x memory reduction during inference.

Usage:
    >>> from bitpolar_llamacpp import BitPolarLlamaCppCache
    >>> cache = BitPolarLlamaCppCache(bits=4)
    >>> cache.compress_kv(key_tensor, value_tensor, layer_idx=0, num_heads=32)
    >>> keys, values = cache.decompress_kv(layer_idx=0)
"""

from bitpolar_llamacpp.kv_cache import BitPolarLlamaCppCache

__all__ = ["BitPolarLlamaCppCache"]
__version__ = "0.3.0"
