# BitPolar

**Near-optimal vector quantization with zero training overhead.**

Compress embeddings to 3-8 bits with provably unbiased inner products and no calibration data. Implements [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874), [PolarQuant (AISTATS 2026)](https://arxiv.org/abs/2502.02617), and [QJL (AAAI 2025)](https://arxiv.org/abs/2406.03482) from Google Research.

## Key Properties

- **Data-oblivious** — no training, no codebooks, no calibration data
- **Deterministic** — fully defined by 4 integers: `(dimension, bits, projections, seed)`
- **Provably unbiased** — inner product estimates satisfy `E[estimate] = exact` at 3+ bits
- **Near-optimal** — distortion within ~2.7x of the Shannon rate-distortion limit
- **Instant indexing** — vectors compress on arrival, 600x faster than Product Quantization

## Installation

```bash
pip install bitpolar
```

## Quick Start

```python
import numpy as np
import bitpolar

# Create quantizer — no training needed
q = bitpolar.TurboQuantizer(dim=384, bits=4, projections=96, seed=42)

# Compress a vector
embedding = np.random.randn(384).astype(np.float32)
code = q.encode(embedding)          # ~6x smaller than float32
decoded = q.decode(code)             # approximate reconstruction
score = q.inner_product(code, embedding)  # approximate inner product

# Build a search index
index = bitpolar.VectorIndex(dim=384, bits=4, projections=96, seed=42)
for i in range(1000):
    vec = np.random.randn(384).astype(np.float32)
    index.add(i, vec)

ids, scores = index.search(embedding, top_k=10)
print(f"Top results: {ids}")
```

## One-Liner Compression

```python
from bitpolar_embeddings import compress_embeddings

embeddings = np.random.randn(10000, 384).astype(np.float32)
compressed = compress_embeddings(embeddings, bits=4)

# Search
indices, scores = compressed.search(query_vector, top_k=10)

# Save/load
compressed.save("my_index.bp")
```

## 58 Integrations

BitPolar integrates with every major AI framework:

| Category | Integrations |
|----------|-------------|
| **RAG Frameworks** | LangChain, LlamaIndex, Haystack, DSPy, FAISS drop-in, ChromaDB |
| **Agentic AI** | LangGraph, CrewAI, OpenAI Agents, Google ADK, Anthropic MCP, AutoGen, SmolAgents, PydanticAI, Agno |
| **Agent Memory** | Mem0, Zep, Letta (MemGPT) |
| **Vector Databases** | Qdrant, Milvus, Weaviate, Pinecone, Redis, Elasticsearch, PostgreSQL, DuckDB, SQLite, Supabase, Neon |
| **LLM Inference** | vLLM, HuggingFace Transformers, llama.cpp, SGLang, TensorRT-LLM, Ollama, ONNX Runtime, Apple MLX |
| **ML Frameworks** | PyTorch, JAX/Flax, TensorFlow/Keras, scikit-learn |
| **Cloud** | Spring AI, Vercel AI SDK, AWS Bedrock, NVIDIA Triton |

### LangChain

```python
from langchain_bitpolar import BitPolarVectorStore

store = BitPolarVectorStore(embedding=my_embeddings, bits=4)
store.add_texts(["doc1", "doc2", "doc3"])
results = store.similarity_search("query", k=5)
```

### LlamaIndex

```python
from llamaindex_bitpolar import BitPolarVectorStore
from llama_index.core import VectorStoreIndex

vector_store = BitPolarVectorStore(bits=4)
index = VectorStoreIndex.from_documents(docs, vector_store=vector_store)
```

### FAISS Drop-in Replacement

```python
from bitpolar_faiss import IndexBitPolarIP

index = IndexBitPolarIP(384, bits=4)  # same API as faiss.IndexFlatIP
index.add(vectors)
D, I = index.search(queries, k=10)    # 4-8x less memory
```

### PyTorch

```python
from bitpolar_torch import quantize_embedding, BitPolarKVCache

compressed = quantize_embedding(embedding_tensor, bits=4)
indices, scores = compressed.search(query_tensor, top_k=10)

cache = BitPolarKVCache(bits=4)
cache.update(key_states, value_states, layer_idx=0)
```

### scikit-learn Pipeline

```python
from bitpolar_sklearn import BitPolarTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("embedder", my_embedding_model),
    ("compressor", BitPolarTransformer(bits=4)),
])
```

## How It Works

```
Input f32 vector
    |
    v
[Random Rotation]     WHT: O(d log d) with O(d) memory
    |
    v
[PolarQuant]          Groups d dims into d/2 pairs -> polar coords
    |                  Radii: lossless | Angles: b-bit quantized
    v
[QJL Residual]        1-bit Johnson-Lindenstrauss sketching
    |                  Unbiased correction for quantization error
    v
TurboCode { polar: PolarCode, residual: QjlSketch }
```

**Inner product estimation**: `<v, q> ~ IP_polar(code, q) + IP_qjl(sketch, q)`

## Parameter Guide

| Use Case | Bits | Compression | Quality |
|----------|------|-------------|---------|
| Maximum compression | 3 | ~8x | Good for cold storage |
| **Recommended** | **4** | **~6x** | **Best for search** |
| High quality | 6 | ~4x | Near-lossless retrieval |
| Near-lossless | 8 | ~3x | Minimal distortion |

## Performance

- **Encode**: 1M vectors in ~2 seconds
- **Search**: Sub-millisecond per query on compressed index
- **Indexing**: 600x faster than Product Quantization
- **Memory**: 4-8x reduction vs float32

## Also Available In

| Language | Package |
|----------|---------|
| Rust | `cargo add bitpolar` |
| JavaScript (WASM) | `npm install @mmgehlot/bitpolar-wasm` |
| Node.js | `npm install @mmgehlot/bitpolar` |
| Go | `go get github.com/mmgehlot/bitpolar/bitpolar-go` |
| Java | Maven Central: `io.github.mmgehlot:bitpolar` |
| PostgreSQL | `cargo pgrx install` (SQL functions) |

## References

- **TurboQuant** (ICLR 2026): [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant** (AISTATS 2026): [arXiv 2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL** (AAAI 2025): [arXiv 2406.03482](https://arxiv.org/abs/2406.03482)

## License

MIT OR Apache-2.0
