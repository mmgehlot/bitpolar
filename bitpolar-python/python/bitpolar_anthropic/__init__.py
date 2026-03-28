"""BitPolar Anthropic Agent SDK Integration — MCP tool server.

Provides an MCP-compatible tool server with stdio and SSE transport
support for Claude Desktop and HTTP-based agent integrations.

Usage:
    >>> from bitpolar_anthropic import BitPolarMCPServer
    >>> server = BitPolarMCPServer(dim=384, bits=4)
    >>> server.run_stdio()  # For Claude Desktop
    >>> server.run_sse(port=8080)  # For HTTP transport
"""

from bitpolar_anthropic.server import BitPolarMCPServer

__all__ = ["BitPolarMCPServer"]
__version__ = "0.3.3"
