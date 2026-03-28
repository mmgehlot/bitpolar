"""BitPolar MCP Tool — expose vector quantization as an AI agent tool.

Implements the Model Context Protocol (MCP) tool interface so AI agents
can compress embeddings, search compressed indices, and manage vector
stores as part of their tool repertoire.

Usage with Claude/MCP:
    Register this tool server, then agents can call:
    - compress_embedding(vector, bits) → compressed bytes
    - search_vectors(query, top_k) → ranked results
    - add_vector(id, vector) → success
    - index_stats() → compression ratio, vector count
"""

from bitpolar_mcp.tools import (
    BitPolarToolServer,
    TOOL_DEFINITIONS,
)

__all__ = ["BitPolarToolServer", "TOOL_DEFINITIONS"]
__version__ = "0.3.3"
