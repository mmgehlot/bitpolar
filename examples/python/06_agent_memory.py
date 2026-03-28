"""BitPolar Agent Memory — compressed episodic memory for AI agents.

Stores text memories with compressed embeddings for efficient
similarity-based recall. Supports time-decay weighting and LRU eviction.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/06_agent_memory.py
"""

import numpy as np
from bitpolar_embeddings.agent_memory import CompressedMemoryStore

print("=== Compressed Agent Memory ===\n")

# Create memory store: 384-dim embeddings, 4-bit, max 1000 memories
memory = CompressedMemoryStore(
    dim=384,
    bits=4,
    max_memories=1000,
    decay_factor=0.01,  # slight time decay
)

# Simulate adding memories with dummy embeddings
memories = [
    ("The user's name is Alice", np.random.randn(384).astype(np.float32)),
    ("Alice prefers Python over JavaScript", np.random.randn(384).astype(np.float32)),
    ("The project deadline is next Friday", np.random.randn(384).astype(np.float32)),
    ("Alice likes coffee, not tea", np.random.randn(384).astype(np.float32)),
    ("The database runs on PostgreSQL", np.random.randn(384).astype(np.float32)),
]

for text, embedding in memories:
    memory.add(text, embedding, metadata={"source": "conversation"})

print(f"Stored {memory.size} memories ({memory.memory_bytes:,} bytes)")
print(f"Memory store: {memory}")

# Recall relevant memories
query = np.random.randn(384).astype(np.float32)
results = memory.recall(query, top_k=3)
print(f"\nTop 3 recalled memories:")
for r in results:
    print(f"  '{r['text']}' (score: {r['score']:.4f}, accessed: {r['access_count']}x)")

# Forget a memory
memory.forget(0)
print(f"\nAfter forgetting index 0: {memory.size} memories")

# Clear all
memory.clear()
print(f"After clear: {memory.size} memories")
