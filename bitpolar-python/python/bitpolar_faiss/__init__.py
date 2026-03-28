"""BitPolar FAISS-compatible Index — drop-in replacement with 4-8x less memory.

API-compatible with faiss.IndexFlatIP and faiss.IndexFlatL2.

Usage:
    >>> from bitpolar_faiss import IndexBitPolarIP
    >>> index = IndexBitPolarIP(384, bits=4)
    >>> index.add(vectors)
    >>> D, I = index.search(queries, k=10)
"""

from bitpolar_faiss.index import IndexBitPolarIP, IndexBitPolarL2, IndexBitPolarIDMap

__all__ = ["IndexBitPolarIP", "IndexBitPolarL2", "IndexBitPolarIDMap"]
__version__ = "0.3.3"
