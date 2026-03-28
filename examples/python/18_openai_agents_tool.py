"""BitPolar as OpenAI Agents SDK tool.

Exposes BitPolar compress/search operations as tool calls
that an OpenAI Agent can invoke during execution.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/18_openai_agents_tool.py
"""

import json
import numpy as np
from bitpolar_agents import BitPolarAgentTool

print("=== BitPolar as OpenAI Agents Tool ===\n")

# Create the tool (manages an internal compressed index)
tool = BitPolarAgentTool(dim=128, bits=4, seed=42)

# Show tool definitions (what the agent sees)
print("Tool definitions for the agent:")
for defn in tool.definitions():
    print(f"  - {defn['name']}: {defn['description'][:60]}...")

# Simulate tool calls from an agent
print("\n--- Simulating agent tool calls ---\n")

# 1. Compress and add vectors
vectors = np.random.randn(100, 128).astype(np.float32)
result = tool.handle_call("bitpolar_add", {
    "vectors": vectors.tolist(),
    "ids": list(range(100)),
    "metadata": [{"label": f"item_{i}"} for i in range(100)],
})
print(f"bitpolar_add -> {json.dumps(result, indent=2)}")

# 2. Search
query = np.random.randn(128).astype(np.float32)
result = tool.handle_call("bitpolar_search", {
    "query": query.tolist(),
    "top_k": 5,
})
print(f"\nbitpolar_search -> {json.dumps(result, indent=2)}")

# 3. Get stats
result = tool.handle_call("bitpolar_stats", {})
print(f"\nbitpolar_stats -> {json.dumps(result, indent=2)}")

# 4. Compress a single vector (for embedding caching)
vec = np.random.randn(128).astype(np.float32)
result = tool.handle_call("bitpolar_compress", {
    "vector": vec.tolist(),
})
print(f"\nbitpolar_compress -> bytes={result['compressed_size']}, ratio={result['ratio']:.1f}x")
