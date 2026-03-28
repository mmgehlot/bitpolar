"""BitPolar Agno Integration — compressed-embedding knowledge base for Agno (Phidata).

Usage:
    >>> from bitpolar_agno import BitPolarKnowledgeBase
    >>> kb = BitPolarKnowledgeBase(dim=384, bits=4)
    >>> ids = kb.add(texts=["hello"], embeddings=[[0.1]*384])
    >>> results = kb.search(query_embedding=[0.1]*384, top_k=5)
"""

from bitpolar_agno.knowledge import BitPolarKnowledgeBase

__all__ = ["BitPolarKnowledgeBase"]
__version__ = "0.3.1"
