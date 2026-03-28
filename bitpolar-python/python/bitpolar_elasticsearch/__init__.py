"""BitPolar Elasticsearch Integration — client-side compression with two-phase search.

Usage:
    >>> from bitpolar_elasticsearch import BitPolarElasticsearchStore
    >>> store = BitPolarElasticsearchStore("my-index", dim=384, bits=4)
    >>> store.add(id="doc1", vector=[0.1]*384, text="hello world")
    >>> results = store.search(query=[0.1]*384, top_k=10)
"""

from bitpolar_elasticsearch.store import BitPolarElasticsearchStore

__all__ = ["BitPolarElasticsearchStore"]
__version__ = "0.3.2"
