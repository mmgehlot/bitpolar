"""BitPolar CrewAI Integration — compressed memory backend.

Provides a CrewAI-compatible memory storage backend that compresses
embeddings using BitPolar quantization for memory-efficient agent memory.

Usage:
    >>> from bitpolar_crewai import BitPolarMemoryBackend
    >>> backend = BitPolarMemoryBackend(bits=4)
    >>> backend.save("key1", "value", metadata={}, embedding=[0.1]*384)
    >>> results = backend.search(query_embedding=[0.1]*384, top_k=5)
"""

from bitpolar_crewai.memory import BitPolarMemoryBackend

__all__ = ["BitPolarMemoryBackend"]
__version__ = "0.3.1"
