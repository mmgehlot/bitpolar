"""BitPolar Vercel AI SDK Integration — Python-side embedding compression middleware.

Provides middleware for compressing embedding responses from any AI SDK provider
before storage or transmission, reducing payload sizes by up to 32x.

For the TypeScript/WASM client-side usage, see the bitpolar-wasm npm package.

Usage:
    >>> from bitpolar_vercel import BitPolarMiddleware
    >>> mw = BitPolarMiddleware(bits=4)
    >>> compressed = mw.compress_request(embedding_response)
"""

from bitpolar_vercel.middleware import BitPolarMiddleware

__all__ = ["BitPolarMiddleware"]
__version__ = "0.3.1"
