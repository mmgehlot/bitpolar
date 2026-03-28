"""BitPolar Google ADK Integration — agent development kit tools.

Provides Google ADK-compatible tool methods for BitPolar vector
quantization, enabling Google AI agents to compress and search vectors.

Usage:
    >>> from bitpolar_google_adk import BitPolarADKTool
    >>> tool = BitPolarADKTool(dim=384, bits=4)
    >>> result = tool.compress(vector=[0.1]*384, bits=4)
"""

from bitpolar_google_adk.tool import BitPolarADKTool

__all__ = ["BitPolarADKTool"]
__version__ = "0.3.1"
