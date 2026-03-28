"""BitPolar AutoGen Integration — compressed memory store.

Provides a Microsoft AutoGen / Semantic Kernel compatible memory store
that compresses embeddings using BitPolar quantization.

Usage:
    >>> from bitpolar_autogen import BitPolarMemoryStore
    >>> store = BitPolarMemoryStore(bits=4)
    >>> store.add("id1", "hello world", embedding=[0.1]*384)
    >>> results = store.search(query_embedding=[0.1]*384, top_k=5)
"""

from bitpolar_autogen.memory import BitPolarMemoryStore

__all__ = ["BitPolarMemoryStore"]
__version__ = "0.3.0"
