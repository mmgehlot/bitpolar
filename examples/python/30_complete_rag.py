"""Complete RAG Pipeline with BitPolar — from embedding to retrieval.

End-to-end retrieval-augmented generation pipeline using BitPolar
for compressed vector storage and search.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/30_complete_rag.py
"""

import numpy as np
import hashlib
import bitpolar

DIM = 256

# =============================================================================
# Step 1: Document Corpus
# =============================================================================
print("=== Step 1: Document Corpus ===\n")

corpus = [
    "BitPolar uses random projections for data-free quantization.",
    "The algorithm achieves 4-8x compression with provably unbiased estimates.",
    "Inner products can be computed directly on compressed codes.",
    "No training data or calibration step is required.",
    "BitPolar supports 1 to 8 bits per dimension.",
    "The Rust implementation processes 10M vectors per second.",
    "Python bindings wrap the Rust core via PyO3.",
    "WASM bindings enable browser-based vector search.",
    "KV cache compression reduces LLM memory by 4-8x.",
    "Agent memory stores use BitPolar for efficient recall.",
    "Vector databases benefit from reduced storage costs.",
    "RAG pipelines use BitPolar to compress document embeddings.",
]
print(f"Corpus: {len(corpus)} documents")


# =============================================================================
# Step 2: Generate Mock Embeddings (deterministic from text)
# =============================================================================
def text_to_embedding(text, dim=DIM):
    h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(h)
    return rng.randn(dim).astype(np.float32)


embeddings = [text_to_embedding(doc) for doc in corpus]
print(f"Embeddings: {len(embeddings)} x {DIM} ({sum(e.nbytes for e in embeddings):,} bytes)")

# =============================================================================
# Step 3: Compress with BitPolar
# =============================================================================
print("\n=== Step 3: Compress ===\n")

q = bitpolar.TurboQuantizer(dim=DIM, bits=4, projections=32, seed=42)
codes = [q.encode(emb) for emb in embeddings]

orig_bytes = sum(e.nbytes for e in embeddings)
comp_bytes = sum(len(c) for c in codes)
print(f"Original: {orig_bytes:,} bytes")
print(f"Compressed: {comp_bytes:,} bytes")
print(f"Ratio: {orig_bytes / comp_bytes:.1f}x ({(1 - comp_bytes / orig_bytes) * 100:.0f}% saved)")

# =============================================================================
# Step 4: Search with Query
# =============================================================================
print("\n=== Step 4: Semantic Search ===\n")

queries = [
    "How fast is BitPolar?",
    "Does it need training data?",
    "What about browser support?",
]

for query_text in queries:
    query_vec = text_to_embedding(query_text)
    scores = [(i, q.inner_product(codes[i], query_vec)) for i in range(len(codes))]
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: '{query_text}'")
    for idx, score in scores[:3]:
        print(f"  [{score:.4f}] {corpus[idx]}")
    print()

# =============================================================================
# Step 5: Verify Quality
# =============================================================================
print("=== Step 5: Quality Check ===\n")

errors = []
for i in range(len(embeddings)):
    decoded = q.decode(codes[i])
    err = np.linalg.norm(embeddings[i] - decoded) / np.linalg.norm(embeddings[i])
    errors.append(err)
print(f"Reconstruction error: mean={np.mean(errors):.4f}, max={np.max(errors):.4f}")

# =============================================================================
# Step 6: Memory Savings at Scale
# =============================================================================
print("\n=== Step 6: Memory Savings at Scale ===\n")

for n in [10_000, 100_000, 1_000_000]:
    orig = n * DIM * 4
    comp = n * (comp_bytes // len(codes))
    print(f"  {n:>10,} docs: {orig/1024/1024:>7.1f} MB -> {comp/1024/1024:>7.1f} MB ({orig/comp:.1f}x)")

print("\nRAG pipeline complete.")
