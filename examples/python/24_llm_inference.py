"""BitPolar LLM Inference — KV cache compression for llama.cpp, SGLang, Transformers.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/24_llm_inference.py
"""

import numpy as np
from bitpolar_vllm import KVCacheQuantizer
from bitpolar_vllm.dynamic_cache import BitPolarDynamicCache

NUM_HEADS = 8
HEAD_DIM = 64
SEQ_LEN = 32

# =============================================================================
# llama.cpp-style KV Cache
# =============================================================================
print("=== BitPolar llama.cpp KV Cache ===\n")

kv_q = KVCacheQuantizer(head_dim=HEAD_DIM, bits=4, num_heads=NUM_HEADS, seed=42)

# Simulate token-by-token KV cache updates
compressed_keys, compressed_values = [], []
for t in range(SEQ_LEN):
    k = np.random.randn(NUM_HEADS, HEAD_DIM).astype(np.float32)
    v = np.random.randn(NUM_HEADS, HEAD_DIM).astype(np.float32)
    compressed_keys.append(kv_q.compress_keys(k))
    compressed_values.append(kv_q.compress_values(v))

orig_bytes = SEQ_LEN * NUM_HEADS * HEAD_DIM * 4 * 2
comp_bytes = sum(sum(len(c) for c in ck) + sum(len(c) for c in cv)
                 for ck, cv in zip(compressed_keys, compressed_values))
print(f"Tokens: {SEQ_LEN}, Heads: {NUM_HEADS}, Head dim: {HEAD_DIM}")
print(f"Original: {orig_bytes:,}B | Compressed: {comp_bytes:,}B | Ratio: {orig_bytes/comp_bytes:.1f}x")

decoded = kv_q.decompress_keys(compressed_keys[0])
error = np.linalg.norm(decoded - np.random.randn(NUM_HEADS, HEAD_DIM)) / (NUM_HEADS * HEAD_DIM)
print(f"Decompressed shape: {decoded.shape}")

savings = kv_q.memory_savings(seq_len=2048)
print(f"At 2048 tokens: {savings['original_bytes']/1024/1024:.1f}MB -> {savings['compressed_bytes']/1024/1024:.1f}MB ({savings['savings_pct']:.0f}% saved)")

# =============================================================================
# SGLang-style Batched Cache
# =============================================================================
print("\n=== BitPolar SGLang Batched Cache ===\n")

batch_q = KVCacheQuantizer(head_dim=HEAD_DIM, bits=3, num_heads=NUM_HEADS, seed=7)

batch_keys = np.random.randn(4, NUM_HEADS, HEAD_DIM).astype(np.float32)
compressed_batch = [batch_q.compress_keys(batch_keys[i]) for i in range(4)]
print(f"Batch of 4 requests compressed (3-bit)")
print(f"Per-request: {batch_keys[0].nbytes}B -> {sum(len(c) for c in compressed_batch[0])}B")

query = np.random.randn(NUM_HEADS, HEAD_DIM).astype(np.float32)
scores = batch_q.attention_score(query, compressed_batch[0])
print(f"Attention scores shape: {scores.shape}, mean: {scores.mean():.6f}")

# =============================================================================
# HuggingFace Transformers DynamicCache
# =============================================================================
print("\n=== BitPolar HuggingFace DynamicCache ===\n")

cache = BitPolarDynamicCache(bits=4, head_dim=HEAD_DIM, num_heads=NUM_HEADS)

for layer in range(2):
    for t in range(SEQ_LEN):
        k = np.random.randn(NUM_HEADS, HEAD_DIM).astype(np.float32)
        v = np.random.randn(NUM_HEADS, HEAD_DIM).astype(np.float32)
        cache.update(k, v, layer_idx=layer)

stats = cache.memory_stats()
print(f"Layers: {stats['layers']}, Positions: {stats['total_positions']}")
print(f"Compression: {stats['ratio']:.1f}x, Savings: {stats['savings_pct']:.0f}%")
print(f"Cache object: {cache}")
