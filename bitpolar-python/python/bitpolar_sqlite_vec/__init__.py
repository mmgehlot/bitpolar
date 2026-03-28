"""BitPolar SQLite-vec Integration — compressed vector storage in SQLite.

Usage:
    >>> from bitpolar_sqlite_vec import BitPolarSQLiteStore
    >>> store = BitPolarSQLiteStore(dim=384, bits=4)
    >>> store.add(id="doc1", vector=[0.1]*384)
    >>> results = store.search(query=[0.1]*384, top_k=5)
"""

from bitpolar_sqlite_vec.store import BitPolarSQLiteStore

__all__ = ["BitPolarSQLiteStore"]
__version__ = "0.3.1"
