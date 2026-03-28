"""BitPolar Agent Memory — Mem0, Zep, Letta integrations.

Compressed episodic and archival memory for AI agents with
time-decay search and LRU eviction.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/26_agent_memory.py
"""

import numpy as np
import time
from bitpolar_embeddings.agent_memory import CompressedMemoryStore

DIM = 384

# =============================================================================
# Zep-style Store: Time-Decay Search
# =============================================================================
print("=== BitPolar Zep-Style Memory (Time-Decay) ===\n")

zep_store = CompressedMemoryStore(dim=DIM, bits=4, max_memories=500, decay_factor=0.05)

conversations = [
    ("User asked about quarterly revenue", 0.0),
    ("Revenue was $42M in Q3, up 18%", 0.1),
    ("User requested competitor analysis", 1.0),
    ("Competitor X launched new product", 1.1),
    ("User wants to schedule board meeting", 3.0),
    ("Board meeting set for next Tuesday", 3.1),
]

for text, delay in conversations:
    embedding = np.random.randn(DIM).astype(np.float32)
    zep_store.add(text, embedding, metadata={"timestamp": time.time() + delay, "source": "chat"})

print(f"Stored {zep_store.size} memories ({zep_store.memory_bytes:,} bytes)")

query = np.random.randn(DIM).astype(np.float32)
results = zep_store.recall(query, top_k=3)
print("\nTop 3 recalled (with time decay):")
for r in results:
    print(f"  score={r['score']:.4f} | '{r['text']}'")

# =============================================================================
# Letta-style Archival Memory
# =============================================================================
print("\n=== BitPolar Letta-Style Archival Memory ===\n")

archival = CompressedMemoryStore(dim=DIM, bits=4, max_memories=10000, decay_factor=0.0)

documents = [
    "System prompt: You are a helpful financial analyst assistant.",
    "User preference: Prefers concise bullet-point responses.",
    "Knowledge: S&P 500 returned 26% in 2023.",
    "Knowledge: Federal funds rate is 5.25-5.50% as of Jan 2024.",
    "Knowledge: Bitcoin ETFs were approved in January 2024.",
    "Task history: Generated 3 earnings reports this week.",
    "User note: Always include data sources in responses.",
]

for doc in documents:
    embedding = np.random.randn(DIM).astype(np.float32)
    archival.add(doc, embedding, metadata={"type": "archival"})

print(f"Archival size: {archival.size} entries ({archival.memory_bytes:,} bytes)")

query = np.random.randn(DIM).astype(np.float32)
results = archival.recall(query, top_k=3)
print("\nArchival search (top 3):")
for r in results:
    print(f"  score={r['score']:.4f} | '{r['text'][:60]}...'")

# Show all entries
all_memories = archival.recall(np.random.randn(DIM).astype(np.float32), top_k=100)
print(f"\nAll entries retrieved: {len(all_memories)}")

# Eviction: fill beyond max to trigger LRU
print("\n=== LRU Eviction Demo ===\n")
small_store = CompressedMemoryStore(dim=DIM, bits=4, max_memories=5, decay_factor=0.0)
for i in range(8):
    small_store.add(f"memory_{i}", np.random.randn(DIM).astype(np.float32))
print(f"Added 8 memories to store with max_memories=5")
print(f"Current size: {small_store.size} (oldest evicted)")

# Pattern: Mem0 integration
# from mem0 import Memory
# m = Memory(); m.add("User likes Python", user_id="alice")
# Compress embeddings before storage:
# code = q.encode(embedding); m.add(text, user_id="alice", metadata={"code": code.hex()})
