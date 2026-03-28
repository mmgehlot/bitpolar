"""BitPolar KV Cache Quantization for vLLM.

Provides KV cache compression for transformer attention layers.

**Stable API** (use directly):
    >>> from bitpolar_vllm import KVCacheQuantizer
    >>> from bitpolar_vllm.dynamic_cache import BitPolarDynamicCache

**Experimental** (vLLM integration — apply() is a pass-through):
    The vLLM quantization registration is not yet functional.
    Use BitPolarDynamicCache as a standalone HuggingFace cache replacement.
"""

from bitpolar_vllm.quantizer import KVCacheQuantizer

__all__ = ["KVCacheQuantizer"]
__version__ = "0.2.0"
