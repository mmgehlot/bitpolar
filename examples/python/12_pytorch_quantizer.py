"""BitPolar + PyTorch — compress embeddings and KV cache.

Quantize embedding layers, create BitPolar-aware linear layers,
and compress transformer KV caches for inference.

Prerequisites:
    pip install bitpolar torch numpy

Usage:
    python examples/python/12_pytorch_quantizer.py
"""

import numpy as np
import bitpolar
from bitpolar_torch import quantize_embedding, BitPolarLinear, BitPolarKVCache

print("=== BitPolar + PyTorch ===\n")

# --- 1. Quantize Embeddings -------------------------------------------------
print("--- Embedding Quantization ---")
embeddings = np.random.randn(1000, 384).astype(np.float32)
codes, quantizer = quantize_embedding(embeddings, bits=4, seed=42)
original_bytes = embeddings.nbytes
compressed_bytes = sum(len(c) for c in codes)
print(f"Embeddings: {embeddings.shape[0]} x {embeddings.shape[1]}")
print(f"Compressed: {original_bytes:,}B -> {compressed_bytes:,}B ({original_bytes / compressed_bytes:.1f}x)")

# Search: find nearest neighbor via compressed inner product
query = np.random.randn(384).astype(np.float32)
scores = [quantizer.inner_product(c, query) for c in codes]
top_id = int(np.argmax(scores))
print(f"Nearest neighbor for random query: id={top_id}, score={scores[top_id]:.4f}")

# --- 2. BitPolarLinear Layer -------------------------------------------------
print("\n--- BitPolarLinear Layer ---")
layer = BitPolarLinear(in_features=384, out_features=128, bits=4, seed=42)
x = np.random.randn(8, 384).astype(np.float32)  # batch of 8
out = layer.forward(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Weight compression: {layer.compression_ratio():.1f}x")

# --- 3. BitPolar KV Cache ---------------------------------------------------
print("\n--- BitPolar KV Cache ---")
cache = BitPolarKVCache(
    num_heads=32,
    head_dim=128,
    bits=4,
    max_seq_len=4096,
    seed=42,
)

# Simulate 50 tokens
for t in range(50):
    k = np.random.randn(32, 128).astype(np.float32)
    v = np.random.randn(32, 128).astype(np.float32)
    cache.update(k, v, position=t)

stats = cache.stats()
print(f"Positions cached: {stats['positions']}")
print(f"Original:   {stats['original_bytes']:,}B")
print(f"Compressed: {stats['compressed_bytes']:,}B")
print(f"Ratio:      {stats['ratio']:.1f}x")

# Decompress a specific position
k_dec, v_dec = cache.get(position=25)
print(f"Decompressed position 25: K={k_dec.shape}, V={v_dec.shape}")
