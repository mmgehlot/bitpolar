"""BitPolar Neon Integration — compressed vectors for Neon serverless Postgres.

Usage:
    >>> from bitpolar_neon import BitPolarNeonClient
    >>> client = BitPolarNeonClient("postgresql://...", dim=384, bits=4)
    >>> client.add(id="doc1", vector=[0.1]*384)
    >>> results = client.search(query=[0.1]*384, top_k=5)
"""

from bitpolar_neon.client import BitPolarNeonClient

__all__ = ["BitPolarNeonClient"]
__version__ = "0.3.0"
