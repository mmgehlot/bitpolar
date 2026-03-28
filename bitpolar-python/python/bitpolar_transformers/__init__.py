"""BitPolar HuggingFace Transformers KV cache.

DynamicCache-compatible KV cache that compresses key/value states
using BitPolar quantization for 4-8x memory reduction during generation.

Usage:
    >>> from bitpolar_transformers import BitPolarCache
    >>> cache = BitPolarCache(bits=4, max_seq_len=2048)
    >>> key_states, value_states = cache.update(keys, values, layer_idx=0)
    >>> seq_len = cache.get_seq_length()
"""

from bitpolar_transformers.cache import BitPolarCache

__all__ = ["BitPolarCache"]
__version__ = "0.3.1"
