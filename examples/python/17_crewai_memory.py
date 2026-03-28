"""BitPolar + CrewAI compressed memory backend.

Provides a memory backend for CrewAI agents that stores
embeddings in compressed form for efficient similarity search.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/17_crewai_memory.py
"""

import numpy as np
from bitpolar_crewai import BitPolarMemoryBackend

print("=== BitPolar + CrewAI Memory Backend ===\n")

# Create memory backend
memory = BitPolarMemoryBackend(dim=384, bits=4, seed=42)

# Simulate agent memories from a CrewAI run
agent_memories = [
    ("researcher", "Found 3 papers on vector quantization published in 2024"),
    ("researcher", "Key finding: random projection methods outperform PQ for streaming data"),
    ("researcher", "BitPolar achieves 600x faster indexing than Product Quantization"),
    ("writer", "Draft introduction completed: 250 words on compression trends"),
    ("writer", "Added comparison table: BitPolar vs PQ vs LSH vs SQ"),
    ("reviewer", "Suggest adding a section on error bounds"),
    ("reviewer", "The compression ratio claims need citation"),
]

# Save memories with mock embeddings
for agent, content in agent_memories:
    embedding = np.random.randn(384).astype(np.float32)
    memory.save(
        content=content,
        embedding=embedding,
        metadata={"agent": agent, "task": "report"},
    )
print(f"Saved {memory.size()} agent memories")

# Search for relevant memories
query = np.random.randn(384).astype(np.float32)
results = memory.search(query, top_k=3)
print(f"\nTop 3 relevant memories:")
for r in results:
    print(f"  [{r['metadata']['agent']}] {r['content'][:60]}... (score: {r['score']:.4f})")

# Filter by agent
researcher_results = memory.search(query, top_k=3, filter={"agent": "researcher"})
print(f"\nResearcher memories only:")
for r in researcher_results:
    print(f"  {r['content'][:60]}... (score: {r['score']:.4f})")

# Memory stats
stats = memory.stats()
print(f"\nMemory stats:")
print(f"  Total memories: {stats['count']}")
print(f"  Compressed size: {stats['compressed_bytes']:,}B")
print(f"  Compression ratio: {stats['ratio']:.1f}x")
print(f"  Agents: {stats['unique_agents']}")
