"""BitPolar + Mem0 vector store backend.

Implements the Mem0 VectorStore interface with BitPolar compression
for memory-efficient agent memory persistence.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/21_mem0_backend.py
"""

import numpy as np
from bitpolar_mem0 import BitPolarVectorStore

print("=== BitPolar + Mem0 Backend ===\n")

# Create vector store
store = BitPolarVectorStore(dim=384, bits=4, seed=42)

# --- Insert memories ---------------------------------------------------------
print("--- Insert ---")
memories = [
    {"id": "m1", "text": "User prefers dark mode in all applications", "user_id": "alice"},
    {"id": "m2", "text": "User's timezone is US/Pacific", "user_id": "alice"},
    {"id": "m3", "text": "User works on a Rust project called BitPolar", "user_id": "alice"},
    {"id": "m4", "text": "User likes coffee, dislikes tea", "user_id": "alice"},
    {"id": "m5", "text": "User's preferred language is Python", "user_id": "bob"},
    {"id": "m6", "text": "User is building a chatbot with LangChain", "user_id": "bob"},
]

for mem in memories:
    embedding = np.random.randn(384).astype(np.float32)
    store.insert(
        vector_id=mem["id"],
        embedding=embedding,
        metadata={"text": mem["text"], "user_id": mem["user_id"]},
    )
print(f"Inserted {store.count()} memories")

# --- Search ------------------------------------------------------------------
print("\n--- Search ---")
query_embedding = np.random.randn(384).astype(np.float32)
results = store.search(query_embedding, top_k=3)
print(f"Top 3 results:")
for r in results:
    print(f"  [{r['id']}] score={r['score']:.4f}: {r['metadata']['text'][:55]}...")

# Search with filter
results = store.search(query_embedding, top_k=3, filters={"user_id": "alice"})
print(f"\nFiltered (user=alice), top 3:")
for r in results:
    print(f"  [{r['id']}] {r['metadata']['text'][:55]}...")

# --- Update ------------------------------------------------------------------
print("\n--- Update ---")
new_embedding = np.random.randn(384).astype(np.float32)
store.update(vector_id="m4", embedding=new_embedding, metadata={"text": "User likes tea now", "user_id": "alice"})
print("Updated m4: 'User likes tea now'")

# --- Delete ------------------------------------------------------------------
print("\n--- Delete ---")
store.delete(vector_id="m5")
print(f"Deleted m5. Remaining: {store.count()} memories")

# Stats
stats = store.stats()
print(f"\nStore: {stats['count']} vectors, {stats['compressed_bytes']:,}B, {stats['ratio']:.1f}x compression")
