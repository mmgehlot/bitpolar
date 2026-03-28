"""BitPolar + LangGraph compressed checkpointer.

Compresses embedding data in LangGraph checkpoints to reduce
storage when persisting agent state.

Prerequisites:
    pip install bitpolar langgraph-checkpoint numpy

Usage:
    python examples/python/16_langgraph_checkpointer.py
"""

import numpy as np
from bitpolar_langgraph import BitPolarCheckpointer

print("=== BitPolar + LangGraph Checkpointer ===\n")

# Create checkpointer with compression settings
checkpointer = BitPolarCheckpointer(dim=384, bits=4, seed=42)

# Simulate a LangGraph checkpoint with embedding data
thread_id = "thread_abc123"
checkpoint = {
    "channel_values": {
        "messages": [
            {"role": "user", "content": "What is vector quantization?"},
            {"role": "assistant", "content": "Vector quantization compresses vectors..."},
        ],
        "memory_embeddings": np.random.randn(50, 384).astype(np.float32),
        "context_vectors": np.random.randn(20, 384).astype(np.float32),
    },
    "metadata": {"step": 5, "model": "gpt-4"},
}

# Put checkpoint (compresses embedding arrays automatically)
original_size = (
    checkpoint["channel_values"]["memory_embeddings"].nbytes
    + checkpoint["channel_values"]["context_vectors"].nbytes
)
checkpointer.put(thread_id, checkpoint_id="cp_001", data=checkpoint)
print(f"Saved checkpoint 'cp_001' for thread '{thread_id}'")

# Get checkpoint back (decompresses transparently)
restored = checkpointer.get(thread_id, checkpoint_id="cp_001")
restored_embeddings = restored["channel_values"]["memory_embeddings"]
restored_context = restored["channel_values"]["context_vectors"]
print(f"Restored memory_embeddings: {restored_embeddings.shape}")
print(f"Restored context_vectors:   {restored_context.shape}")

# Compression stats
stats = checkpointer.stats(thread_id, checkpoint_id="cp_001")
print(f"\nCompression stats:")
print(f"  Original embedding bytes: {stats['original_bytes']:,}")
print(f"  Compressed bytes:         {stats['compressed_bytes']:,}")
print(f"  Ratio:                    {stats['ratio']:.1f}x")

# Verify reconstruction quality
error = np.linalg.norm(
    checkpoint["channel_values"]["memory_embeddings"] - restored_embeddings
) / np.linalg.norm(checkpoint["channel_values"]["memory_embeddings"])
print(f"  Reconstruction error:     {error:.4f}")

# List checkpoints for thread
cps = checkpointer.list(thread_id)
print(f"\nCheckpoints for '{thread_id}': {[c['id'] for c in cps]}")
