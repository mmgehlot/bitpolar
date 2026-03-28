"""BitPolar Letta (MemGPT) Integration — compressed archival memory tier.

Usage:
    >>> from bitpolar_letta import BitPolarArchivalMemory
    >>> mem = BitPolarArchivalMemory(dim=384, bits=4)
    >>> doc_id = mem.insert(text="hello world", embedding=[0.1]*384)
    >>> results = mem.search(query_embedding=[0.1]*384, top_k=10)
"""

from bitpolar_letta.archival import BitPolarArchivalMemory

__all__ = ["BitPolarArchivalMemory"]
__version__ = "0.3.3"
