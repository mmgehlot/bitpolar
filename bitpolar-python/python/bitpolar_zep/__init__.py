"""BitPolar Zep Integration — compressed store with time-decay scoring.

Usage:
    >>> from bitpolar_zep import BitPolarZepStore
    >>> store = BitPolarZepStore(dim=384, bits=4)
    >>> store.add(text="hello", embedding=[0.1]*384)
    >>> results = store.search(query_embedding=[0.1]*384, top_k=5)
"""

from bitpolar_zep.store import BitPolarZepStore

__all__ = ["BitPolarZepStore"]
__version__ = "0.2.0"
