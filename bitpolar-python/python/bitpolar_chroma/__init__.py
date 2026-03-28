"""BitPolar ChromaDB Integration — compressed embeddings for ChromaDB.

Usage:
    >>> from bitpolar_chroma import BitPolarChromaStore
    >>> store = BitPolarChromaStore(collection_name="docs", bits=4)
    >>> store.add(texts=["hello", "world"], embeddings=[[0.1]*384, [0.2]*384])
    >>> results = store.search(query_embedding=[0.1]*384, top_k=5)
"""

from bitpolar_chroma.store import BitPolarChromaStore, BitPolarEmbeddingFunction

__all__ = ["BitPolarChromaStore", "BitPolarEmbeddingFunction"]
__version__ = "0.3.1"
