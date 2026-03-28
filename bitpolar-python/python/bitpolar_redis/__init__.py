"""BitPolar Redis Integration — compressed vector storage in Redis.

Usage:
    >>> from bitpolar_redis import BitPolarRedisStore
    >>> store = BitPolarRedisStore(dim=384, bits=4)
    >>> store.add(id="doc1", vector=[0.1]*384, metadata={"text": "hello"})
    >>> results = store.search(query=[0.1]*384, top_k=5)
"""

from bitpolar_redis.store import BitPolarRedisStore

__all__ = ["BitPolarRedisStore"]
__version__ = "0.3.1"
