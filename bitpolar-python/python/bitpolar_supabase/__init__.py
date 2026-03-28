"""BitPolar Supabase Integration — compressed vectors for Supabase pgvector.

Usage:
    >>> from bitpolar_supabase import BitPolarSupabaseClient
    >>> client = BitPolarSupabaseClient(url="...", key="...", dim=384, bits=4)
    >>> client.add(id="doc1", vector=[0.1]*384)
    >>> results = client.search(query=[0.1]*384, top_k=5)
"""

from bitpolar_supabase.client import BitPolarSupabaseClient

__all__ = ["BitPolarSupabaseClient"]
__version__ = "0.3.2"
