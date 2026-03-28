"""BitPolar Milvus Integration — client-side compression with two-phase search.

Usage:
    >>> from bitpolar_milvus import BitPolarMilvusIndex
    >>> index = BitPolarMilvusIndex("my_collection", dim=384, bits=4)
    >>> index.upsert(ids=["id1"], vectors=[[0.1]*384])
    >>> results = index.search(query=[0.1]*384, top_k=10)
"""

from bitpolar_milvus.index import BitPolarMilvusIndex

__all__ = ["BitPolarMilvusIndex"]
__version__ = "0.3.3"
