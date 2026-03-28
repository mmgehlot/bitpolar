"""BitPolar Weaviate Integration — client-side compression with two-phase search.

Usage:
    >>> from bitpolar_weaviate import BitPolarWeaviateIndex
    >>> index = BitPolarWeaviateIndex("Document", dim=384, bits=4)
    >>> index.upsert(ids=["id1"], vectors=[[0.1]*384])
    >>> results = index.search(query=[0.1]*384, top_k=10)
"""

from bitpolar_weaviate.index import BitPolarWeaviateIndex

__all__ = ["BitPolarWeaviateIndex"]
__version__ = "0.3.1"
