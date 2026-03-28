"""BitPolar SmolAgents Integration — vector search tool for HuggingFace SmolAgents.

Usage:
    >>> from bitpolar_smolagents import BitPolarTool
    >>> tool = BitPolarTool(bits=4)
    >>> tool.forward("add", vector=[0.1]*384, vector_id="doc1")
    >>> result = tool.forward("search", query=[0.1]*384, top_k=5)
"""

from bitpolar_smolagents.tool import BitPolarTool

__all__ = ["BitPolarTool"]
__version__ = "0.3.3"
