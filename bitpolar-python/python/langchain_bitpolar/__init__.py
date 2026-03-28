"""LangChain VectorStore backed by BitPolar compression.

4-8x less memory than FAISS with <2% recall loss. No external database needed.

Usage:
    >>> from langchain_bitpolar import BitPolarVectorStore
    >>> from langchain_openai import OpenAIEmbeddings
    >>>
    >>> store = BitPolarVectorStore.from_texts(
    ...     texts=["hello", "world"],
    ...     embedding=OpenAIEmbeddings(),
    ...     bits=4,
    ... )
    >>> results = store.similarity_search("greeting", k=1)
"""

from langchain_bitpolar.vectorstore import BitPolarVectorStore

__all__ = ["BitPolarVectorStore"]
__version__ = "0.3.3"
