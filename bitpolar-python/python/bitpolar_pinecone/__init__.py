"""BitPolar Pinecone Integration — client-side compression with two-phase search.

Usage:
    >>> from bitpolar_pinecone import BitPolarPineconeIndex
    >>> index = BitPolarPineconeIndex("my-index", dim=384, bits=4)
    >>> index.upsert(ids=["id1"], vectors=[[0.1]*384])
    >>> results = index.search(query=[0.1]*384, top_k=10)
"""

from bitpolar_pinecone.index import BitPolarPineconeIndex

__all__ = ["BitPolarPineconeIndex"]
__version__ = "0.3.1"
