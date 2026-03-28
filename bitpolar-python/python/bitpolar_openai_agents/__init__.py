"""BitPolar OpenAI Agents SDK Integration — function-calling tools.

Provides OpenAI function-calling compatible tool definitions and
a handler class for BitPolar vector quantization operations.

Usage:
    >>> from bitpolar_openai_agents import TOOL_DEFINITIONS, BitPolarAgentTool
    >>> tool = BitPolarAgentTool(dim=384, bits=4)
    >>> result = tool.handle_tool_call("bitpolar_compress", {"vector": [0.1]*384})
"""

from bitpolar_openai_agents.tools import (
    BitPolarAgentTool,
    TOOL_DEFINITIONS,
    handle_tool_call,
)

__all__ = ["BitPolarAgentTool", "TOOL_DEFINITIONS", "handle_tool_call"]
__version__ = "0.3.2"
