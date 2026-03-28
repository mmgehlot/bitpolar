"""BitPolar Haystack Integration — DocumentStore and Retriever for Haystack 2.x.

Usage:
    >>> from bitpolar_haystack import BitPolarDocumentStore, BitPolarRetriever
    >>> store = BitPolarDocumentStore(bits=4)
    >>> retriever = BitPolarRetriever(document_store=store)
"""

from bitpolar_haystack.document_store import BitPolarDocumentStore
from bitpolar_haystack.retriever import BitPolarRetriever

__all__ = ["BitPolarDocumentStore", "BitPolarRetriever"]
__version__ = "0.3.1"
