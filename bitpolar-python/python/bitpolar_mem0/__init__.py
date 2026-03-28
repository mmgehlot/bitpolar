"""BitPolar Mem0 Integration — vector store backend for Mem0 agent memory.

Usage:
    >>> from bitpolar_mem0 import BitPolarVectorStore
    >>> store = BitPolarVectorStore(dim=384, bits=4)
    >>> store.insert(vectors=[[0.1]*384], payloads=[{"text": "hello"}], ids=["id1"])
    >>> results = store.search(query=[0.1]*384, limit=5)
"""

from bitpolar_mem0.store import BitPolarVectorStore

__all__ = ["BitPolarVectorStore"]
__version__ = "0.3.2"
