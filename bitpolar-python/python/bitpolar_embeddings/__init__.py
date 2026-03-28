"""BitPolar Embeddings — one-line compression for any embedding matrix.

Integrates with HuggingFace sentence-transformers for seamless ML pipeline
compression. Reduces embedding storage by 4-8x with <2% recall loss.

Usage:
    >>> import numpy as np
    >>> from bitpolar_embeddings import compress_embeddings, BitPolarEncoder
    >>>
    >>> # Compress any embedding matrix
    >>> embeddings = np.random.randn(1000, 384).astype(np.float32)
    >>> compressed = compress_embeddings(embeddings, bits=4)
    >>> print(f"Compression ratio: {compressed.compression_ratio:.1f}x")
    >>>
    >>> # Or wrap a sentence-transformers model
    >>> encoder = BitPolarEncoder("all-MiniLM-L6-v2", bits=4)
    >>> compressed = encoder.encode(["hello world", "semantic search"])
    >>> ids, scores = compressed.search(np.random.randn(384).astype(np.float32))
"""

from bitpolar_embeddings.core import (
    compress_embeddings,
    CompressedEmbeddings,
    BitPolarEncoder,
)

__all__ = [
    "compress_embeddings",
    "CompressedEmbeddings",
    "BitPolarEncoder",
]

__version__ = "0.2.0"
