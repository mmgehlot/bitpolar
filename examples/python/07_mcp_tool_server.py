"""BitPolar MCP Tool Server — expose vector quantization to AI agents.

Creates an MCP-compatible tool server that AI agents can use to
compress embeddings, search vectors, and manage indices.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/07_mcp_tool_server.py
"""

import numpy as np
from bitpolar_mcp import BitPolarToolServer, TOOL_DEFINITIONS
import json

print("=== BitPolar MCP Tool Server ===\n")

# Show available tools
print("Available MCP tools:")
for tool in TOOL_DEFINITIONS:
    print(f"  {tool['name']}: {tool['description'][:80]}...")

# Create the tool server
server = BitPolarToolServer(dim=128, bits=4, seed=42)

# Simulate AI agent tool calls

# 1. Add vectors to the index
print("\n--- Agent adds vectors ---")
for i in range(100):
    result = server.handle_tool_call("bitpolar_add_vector", {
        "id": i,
        "vector": np.random.randn(128).astype(np.float32).tolist(),
    })
print(f"Added 100 vectors: {result}")

# 2. Check index stats
print("\n--- Agent checks stats ---")
stats = server.handle_tool_call("bitpolar_index_stats", {})
print(json.dumps(stats, indent=2))

# 3. Search
print("\n--- Agent searches ---")
query = np.random.randn(128).astype(np.float32).tolist()
results = server.handle_tool_call("bitpolar_search", {
    "query": query,
    "top_k": 5,
})
print(f"Search results: {json.dumps(results['results'], indent=2)}")

# 4. Compress a single vector
print("\n--- Agent compresses a vector ---")
compress_result = server.handle_tool_call("bitpolar_compress", {
    "vector": np.random.randn(128).astype(np.float32).tolist(),
    "bits": 3,  # Override default bits
})
print(f"Compressed: {compress_result['original_bytes']}B → {compress_result['compressed_bytes']}B "
      f"({compress_result['compression_ratio']}x)")
