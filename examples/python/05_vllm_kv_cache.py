"""BitPolar KV Cache Compression for LLM Inference.

Compresses transformer Key-Value caches to reduce memory usage,
enabling longer context lengths on the same hardware.

Works standalone (no vLLM required) or as a vLLM quantization backend.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/05_vllm_kv_cache.py
"""

import numpy as np
from bitpolar_vllm import KVCacheQuantizer

print("=== BitPolar KV Cache Compression ===\n")

# Simulate a Llama-3-8B-like model: 32 heads, 128 head_dim
num_heads = 32
head_dim = 128

kv_q = KVCacheQuantizer(
    head_dim=head_dim,
    bits=4,
    num_heads=num_heads,
    seed=42,
)
print(f"Quantizer: {kv_q}")

# Simulate compressing keys for one token
keys = np.random.randn(num_heads, head_dim).astype(np.float32)
values = np.random.randn(num_heads, head_dim).astype(np.float32)

# Compress
compressed_keys = kv_q.compress_keys(keys)
compressed_values = kv_q.compress_values(values)
print(f"\nCompressed {num_heads} heads × {head_dim} dim:")
print(f"  Original: {keys.nbytes + values.nbytes:,} bytes")
compressed_bytes = sum(len(c) for c in compressed_keys) + sum(len(c) for c in compressed_values)
print(f"  Compressed: {compressed_bytes:,} bytes")
print(f"  Ratio: {(keys.nbytes + values.nbytes) / compressed_bytes:.1f}x")

# Decompress
decoded_keys = kv_q.decompress_keys(compressed_keys)
error = np.linalg.norm(keys - decoded_keys) / np.linalg.norm(keys)
print(f"  Key reconstruction error: {error:.4f}")

# Compute attention scores on compressed keys
query = np.random.randn(num_heads, head_dim).astype(np.float32)
attention_scores = kv_q.attention_score(query, compressed_keys)
print(f"\nAttention scores (per head): shape={attention_scores.shape}")
print(f"  Mean score: {attention_scores.mean():.6f}")

# Memory savings calculator
savings = kv_q.memory_savings(seq_len=4096)
print(f"\nMemory savings for 4096 tokens:")
print(f"  Original: {savings['original_bytes']/1024/1024:.1f} MB")
print(f"  Compressed: {savings['compressed_bytes']/1024/1024:.1f} MB")
print(f"  Ratio: {savings['ratio']:.1f}x")
print(f"  Savings: {savings['savings_pct']:.0f}%")

# =============================================================================
# HuggingFace DynamicCache replacement
# =============================================================================

print("\n=== HuggingFace DynamicCache Replacement ===\n")

from bitpolar_vllm.dynamic_cache import BitPolarDynamicCache

cache = BitPolarDynamicCache(bits=4, head_dim=64, num_heads=8)

# Simulate adding tokens to the cache
for token_idx in range(20):
    keys = np.random.randn(8, 64).astype(np.float32)   # (num_heads, head_dim)
    values = np.random.randn(8, 64).astype(np.float32)
    cache.update(keys, values, layer_idx=0)

stats = cache.memory_stats()
print(f"Cache: {cache}")
print(f"  Layers: {stats['layers']}")
print(f"  Positions: {stats['total_positions']}")
print(f"  Compression: {stats['ratio']:.1f}x")
print(f"  Savings: {stats['savings_pct']:.0f}%")
