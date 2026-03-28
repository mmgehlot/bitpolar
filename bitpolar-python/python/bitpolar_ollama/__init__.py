"""BitPolar Ollama embedding compression.

Calls the Ollama REST API for embeddings and compresses them with
BitPolar quantization for memory-efficient storage and search.

Usage:
    >>> from bitpolar_ollama import BitPolarOllamaClient
    >>> client = BitPolarOllamaClient()
    >>> embedding = client.embed("Hello world")
    >>> result = client.embed_and_store("Hello world")
"""

from bitpolar_ollama.client import BitPolarOllamaClient

__all__ = ["BitPolarOllamaClient"]
__version__ = "0.3.3"
