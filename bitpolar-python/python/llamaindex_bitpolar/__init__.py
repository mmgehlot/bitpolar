"""BitPolar VectorStore for LlamaIndex.

Drop-in replacement for any LlamaIndex vector store with 4-8x less memory.

Usage:
    >>> from llamaindex_bitpolar import BitPolarVectorStore
    >>> from llama_index.core import VectorStoreIndex
    >>> vector_store = BitPolarVectorStore(bits=4)
    >>> index = VectorStoreIndex.from_documents(docs, vector_store=vector_store)
"""

from llamaindex_bitpolar.vectorstore import BitPolarVectorStore

__all__ = ["BitPolarVectorStore"]
__version__ = "0.2.0"
