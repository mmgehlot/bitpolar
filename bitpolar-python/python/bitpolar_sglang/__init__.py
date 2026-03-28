"""BitPolar SGLang KV cache compression.

Compresses SGLang KV cache tensors using BitPolar quantization,
optimized for RadixAttention patterns with prefix-sharing support.

Usage:
    >>> from bitpolar_sglang import BitPolarSGLangCache
    >>> cache = BitPolarSGLangCache(bits=4)
    >>> cache.update(key_tensor, value_tensor, layer_idx=0)
    >>> keys, values = cache.get(layer_idx=0)
"""

from bitpolar_sglang.cache import BitPolarSGLangCache

__all__ = ["BitPolarSGLangCache"]
__version__ = "0.3.0"
