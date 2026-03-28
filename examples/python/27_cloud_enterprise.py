"""BitPolar Cloud & Enterprise — AWS Bedrock, Triton, Vercel AI SDK.

Demonstrates middleware patterns for compressing embeddings in
cloud and enterprise deployment scenarios.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/27_cloud_enterprise.py
"""

import numpy as np
import json
import bitpolar

DIM = 1536  # OpenAI/Bedrock embedding dimension
q = bitpolar.TurboQuantizer(dim=DIM, bits=4, projections=64, seed=42)

# =============================================================================
# Vercel AI SDK Middleware Pattern
# =============================================================================
print("=== Vercel AI SDK — Embedding Middleware ===\n")


class BitPolarMiddleware:
    """Compresses embedding responses before sending to client."""

    def __init__(self, dim, bits=4):
        self.q = bitpolar.TurboQuantizer(dim=dim, bits=bits, projections=64, seed=42)

    def compress_response(self, embeddings):
        compressed = []
        for emb in embeddings:
            code = self.q.encode(np.array(emb, dtype=np.float32))
            compressed.append({"code": code.hex(), "bytes": len(code)})
        return compressed

    def decompress_response(self, compressed):
        restored = []
        for item in compressed:
            code = bytes.fromhex(item["code"])
            vec = self.q.decode(code)
            restored.append(vec.tolist())
        return restored


middleware = BitPolarMiddleware(dim=DIM)

mock_embeddings = [np.random.randn(DIM).astype(np.float32).tolist() for _ in range(5)]
orig_size = len(json.dumps(mock_embeddings))

compressed = middleware.compress_response(mock_embeddings)
comp_size = len(json.dumps(compressed))
print(f"5 embeddings ({DIM}-dim):")
print(f"  JSON original: {orig_size:,} chars")
print(f"  JSON compressed: {comp_size:,} chars")
print(f"  Reduction: {(1 - comp_size / orig_size) * 100:.0f}%")

restored = middleware.decompress_response(compressed)
print(f"  Restored shape: {len(restored)} x {len(restored[0])}")

# =============================================================================
# AWS Bedrock Pattern (commented — needs boto3)
# =============================================================================
print("\n=== AWS Bedrock Pattern (requires boto3) ===\n")
print("Code pattern for Bedrock Titan embeddings:")

# import boto3
# bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
# response = bedrock.invoke_model(
#     modelId="amazon.titan-embed-text-v2:0",
#     body=json.dumps({"inputText": "Hello world"}),
# )
# embedding = json.loads(response["body"].read())["embedding"]
# code = q.encode(np.array(embedding, dtype=np.float32))
# # Store compressed: 6144B -> ~768B
# decoded = q.decode(code)  # Restore when needed

savings = q.code_size_bytes
print(f"  Titan v2 ({DIM}-dim): {DIM*4}B -> {savings}B per vector")
print(f"  Compression: {DIM*4 / savings:.1f}x")

# =============================================================================
# NVIDIA Triton Pattern (commented — needs server)
# =============================================================================
print("\n=== NVIDIA Triton Pattern (requires server) ===\n")
print("Code pattern for Triton inference post-processing:")

# import tritonclient.grpc as grpcclient
# client = grpcclient.InferenceServerClient("localhost:8001")
# inputs = [grpcclient.InferInput("input", [1, DIM], "FP32")]
# inputs[0].set_data_from_numpy(query_vec)
# result = client.infer("embedding_model", inputs)
# embedding = result.as_numpy("output")[0]
# code = q.encode(embedding)  # Compress before caching

print(f"  Post-inference compression: {DIM*4}B -> {savings}B")
print(f"  Cache 1M vectors: {DIM*4*1e6/1024/1024:.0f}MB -> {savings*1e6/1024/1024:.0f}MB")
