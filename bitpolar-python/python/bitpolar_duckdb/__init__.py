"""BitPolar DuckDB Integration — compressed vector storage in DuckDB.

Usage:
    >>> from bitpolar_duckdb import BitPolarDuckDBStore
    >>> store = BitPolarDuckDBStore(dim=384, bits=4)
    >>> store.add(id="doc1", vector=[0.1]*384)
    >>> results = store.search(query=[0.1]*384, top_k=5)
"""

from bitpolar_duckdb.store import BitPolarDuckDBStore

__all__ = ["BitPolarDuckDBStore"]
__version__ = "0.2.0"
