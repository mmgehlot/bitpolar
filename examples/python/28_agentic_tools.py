"""BitPolar Agentic AI Tools — OpenAI, Google ADK, Anthropic, SmolAgents, PydanticAI, Agno.

Tool definitions and handler patterns for major agent frameworks.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/28_agentic_tools.py
"""

import numpy as np
import json
import bitpolar

DIM = 128
q = bitpolar.TurboQuantizer(dim=DIM, bits=4, projections=32, seed=42)
index = {}  # Simple in-memory index

# =============================================================================
# OpenAI Function Calling Tool
# =============================================================================
print("=== OpenAI — Function Calling Tool ===\n")

OPENAI_TOOLS = [
    {"type": "function", "function": {
        "name": "bitpolar_compress", "description": "Compress a vector embedding",
        "parameters": {"type": "object", "properties": {
            "vector": {"type": "array", "items": {"type": "number"}},
        }, "required": ["vector"]}}},
    {"type": "function", "function": {
        "name": "bitpolar_search", "description": "Search compressed vectors",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "array", "items": {"type": "number"}},
            "top_k": {"type": "integer", "default": 5},
        }, "required": ["query"]}}},
]


def handle_tool_call(name, args):
    if name == "bitpolar_compress":
        vec = np.array(args["vector"], dtype=np.float32)
        code = q.encode(vec)
        vid = len(index)
        index[vid] = (code, f"doc_{vid}")
        return {"id": vid, "original_bytes": len(args["vector"]) * 4, "compressed_bytes": len(code)}
    elif name == "bitpolar_search":
        query = np.array(args["query"], dtype=np.float32)
        scores = [(vid, q.inner_product(code, query), label) for vid, (code, label) in index.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return {"results": [{"id": v, "score": round(s, 4), "label": l}
                            for v, s, l in scores[:args.get("top_k", 5)]]}
    return {"error": f"Unknown tool: {name}"}


# Simulate agent adding vectors
for i in range(10):
    vec = np.random.randn(DIM).astype(np.float32).tolist()
    result = handle_tool_call("bitpolar_compress", {"vector": vec})
print(f"Agent added 10 vectors: {result}")

query = np.random.randn(DIM).astype(np.float32).tolist()
result = handle_tool_call("bitpolar_search", {"query": query, "top_k": 3})
print(f"Search results: {json.dumps(result, indent=2)}")

# =============================================================================
# Google ADK Tool Pattern
# =============================================================================
print("\n=== Google ADK Tool Pattern ===\n")
print("# from google.adk import Tool")
print("# @Tool(name='bitpolar_compress')")
print("# def compress(vector: list[float]) -> dict:")
print("#     code = q.encode(np.array(vector, dtype=np.float32))")
print("#     return {'compressed_bytes': len(code)}")

# =============================================================================
# Anthropic MCP Tool Pattern
# =============================================================================
print("\n=== Anthropic MCP Tool Pattern ===\n")
print("# Server exposes bitpolar_compress and bitpolar_search as MCP tools")
print("# See example 07_mcp_tool_server.py for full implementation")

# =============================================================================
# SmolAgents / PydanticAI / Agno Patterns
# =============================================================================
print("\n=== SmolAgents Tool ===")
print("# class BitPolarTool(Tool):")
print("#     def forward(self, vector): return q.encode(np.array(vector))")

print("\n=== PydanticAI Tool ===")
print("# @agent.tool; def compress(ctx, vector: list[float]) -> bytes:")
print("#     return q.encode(np.array(vector, dtype=np.float32))")

print("\n=== Agno Tool ===")
print("# class BitPolarTools(Toolkit):")
print("#     def compress(self, vector): return q.encode(np.array(vector))")
