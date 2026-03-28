"""BitPolar + LangChain VectorStore.

Drop-in replacement for FAISS/Chroma with 4-8x less memory.
No external database needed — everything runs in-memory.

Prerequisites:
    pip install bitpolar numpy langchain-core langchain-openai

Usage:
    python examples/python/03_langchain_vectorstore.py
"""

from langchain_bitpolar import BitPolarVectorStore

# =============================================================================
# Option A: Using a mock embedding for demo (no API key needed)
# =============================================================================

print("=== BitPolar LangChain VectorStore ===\n")

# Simple embedding function for demo purposes
class DemoEmbeddings:
    """Mock embedding that produces deterministic vectors from text."""
    def embed_documents(self, texts):
        import hashlib, numpy as np
        results = []
        for text in texts:
            # Use hash as seed for deterministic but valid float vectors
            h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(h)
            vec = rng.randn(128).astype(np.float32)
            results.append(vec.tolist())
        return results

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Create the store
store = BitPolarVectorStore(
    embedding=DemoEmbeddings(),
    bits=4,
    seed=42,
)

# Add documents
docs = [
    "BitPolar compresses vectors to 3-8 bits",
    "No training or calibration data needed",
    "Inner product estimates are provably unbiased",
    "Works with any embedding model",
    "600x faster indexing than Product Quantization",
    "Supports Rust, Python, WASM, Go, Java, Node.js",
]
ids = store.add_texts(docs)
print(f"Added {len(ids)} documents")
print(f"Store size: {len(store)} documents")

# Search
results = store.similarity_search("How fast is BitPolar?", k=3)
print(f"\nSearch results for 'How fast is BitPolar?':")
for i, doc in enumerate(results):
    print(f"  {i+1}. {doc.page_content}")

# Search with scores
results_with_scores = store.similarity_search_with_score("compression", k=2)
print(f"\nSearch with scores for 'compression':")
for doc, score in results_with_scores:
    print(f"  {doc.page_content} (score: {score:.4f})")

# Delete a document
store.delete([ids[0]])
print(f"\nAfter deleting doc 0: {len(store)} documents remaining")

# =============================================================================
# Option B: Using OpenAI embeddings (requires API key)
# =============================================================================

print("\n=== With OpenAI Embeddings (requires OPENAI_API_KEY) ===\n")
try:
    from langchain_openai import OpenAIEmbeddings
    import os
    if os.getenv("OPENAI_API_KEY"):
        store = BitPolarVectorStore.from_texts(
            texts=docs,
            embedding=OpenAIEmbeddings(),
            bits=4,
        )
        results = store.similarity_search("vector compression", k=2)
        for doc in results:
            print(f"  {doc.page_content}")
    else:
        print("Set OPENAI_API_KEY to try with real embeddings")
except ImportError:
    print("langchain-openai not installed — skipping")
    print("Install with: pip install langchain-openai")
