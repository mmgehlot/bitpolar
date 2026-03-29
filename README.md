<p align="center">
  <img src="BitPolar.png" alt="BitPolar" width="400">
</p>

<h1 align="center">BitPolar</h1>

<p align="center">
  Near-optimal vector quantization with zero training overhead
</p>

<p align="center">
  <a href="https://crates.io/crates/bitpolar"><img src="https://img.shields.io/crates/v/bitpolar.svg" alt="Crates.io"></a>
  <a href="https://pypi.org/project/bitpolar/"><img src="https://img.shields.io/pypi/v/bitpolar.svg" alt="PyPI"></a>
  <a href="https://docs.rs/bitpolar"><img src="https://docs.rs/bitpolar/badge.svg" alt="docs.rs"></a>
  <a href="https://github.com/mmgehlot/bitpolar/actions"><img src="https://github.com/mmgehlot/bitpolar/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/crates/l/bitpolar.svg" alt="License"></a>
  <a href="https://github.com/mmgehlot/bitpolar/stargazers"><img src="https://img.shields.io/github/stars/mmgehlot/bitpolar?style=social" alt="GitHub Stars"></a>
</p>

---

Compress embeddings to 3-8 bits with provably unbiased inner products and no calibration data. Implements TurboQuant ([ICLR 2026](https://arxiv.org/abs/2504.19874)), PolarQuant ([AISTATS 2026](https://arxiv.org/abs/2502.02617)), and QJL ([AAAI 2025](https://arxiv.org/abs/2406.03482)) from Google Research.

## Key Properties

- **Data-oblivious** — no training, no codebooks, no calibration data
- **Deterministic** — fully defined by 4 integers: `(dimension, bits, projections, seed)`
- **Provably unbiased** — inner product estimates satisfy `E[estimate] = exact` at 3+ bits
- **Near-optimal** — distortion within ~2.7x of the Shannon rate-distortion limit
- **Instant indexing** — vectors compress on arrival, 600x faster than Product Quantization

## What's New in 0.3.x

- **58 integrations** — every major AI framework, vector database, and ML library
- **PyTorch torchao** — embedding quantizer, BitPolarLinear, KV cache
- **FAISS drop-in** — API-compatible IndexBitPolarIP/L2 replacement
- **LlamaIndex, Haystack, DSPy** — VectorStore and Retriever integrations
- **Agentic AI** — LangGraph, CrewAI, OpenAI Agents, Google ADK, SmolAgents, PydanticAI
- **Agent memory** — Mem0, Zep, Letta backends
- **11 vector databases** — Milvus, Weaviate, Pinecone, Redis, ES, DuckDB, SQLite, and more
- **LLM inference** — llama.cpp, SGLang, TensorRT, Ollama, MLX KV cache compression
- **ML frameworks** — JAX/Flax, TensorFlow/Keras, scikit-learn pipeline
- **30 Python examples** covering all integrations
- **Comprehensive benchmarks** — throughput, recall, KV cache fidelity, FAISS comparison

### 0.2.0 (Previous)

- **Walsh-Hadamard Transform** — O(d log d) rotation with O(d) memory (577x less than Haar QR)
- **Python bindings** — PyO3 + maturin, zero-copy numpy integration
- **WASM bindings** — browser-side vector search via wasm-bindgen
- **`no_std` support** — embedded/edge deployment with `alloc` feature

## Quick Start

### Rust

```toml
[dependencies]
bitpolar = "0.3"
```

```rust
use bitpolar::TurboQuantizer;
use bitpolar::traits::VectorQuantizer;

// Create quantizer from 4 integers — no training needed
let q = TurboQuantizer::new(128, 4, 32, 42).unwrap();

// Encode a vector
let vector = vec![0.1_f32; 128];
let code = q.encode(&vector).unwrap();

// Estimate inner product without decompression
let query = vec![0.05_f32; 128];
let score = q.inner_product_estimate(&code, &query).unwrap();

// Decode back to approximate vector
let reconstructed = q.decode(&code);
```

### Python

```bash
pip install bitpolar
```

```python
import numpy as np
import bitpolar

# Create quantizer — no training needed
q = bitpolar.TurboQuantizer(dim=768, bits=4, projections=192, seed=42)

# Encode/decode
embedding = np.random.randn(768).astype(np.float32)
code = q.encode(embedding)
decoded = q.decode(code)

# Build a search index
index = bitpolar.VectorIndex(dim=768, bits=4)
for i, vec in enumerate(embeddings):
    index.add(i, vec)

ids, scores = index.search(query, top_k=10)
```

### JavaScript (WASM)

```javascript
import init, { WasmQuantizer, WasmVectorIndex } from 'bitpolar-wasm';

await init();

const q = new WasmQuantizer(128, 4, 32, 42n);
const code = q.encode(new Float32Array(128).fill(0.1));
const decoded = q.decode(code);

const index = new WasmVectorIndex(128, 4, 32, 42n);
index.add(0, vector);
const results = index.search(query, 5);
```

## Walsh-Hadamard Transform

The WHT provides an O(d log d) alternative to Haar QR rotation:

| Property | Haar QR | Walsh-Hadamard |
|---|---|---|
| Time complexity | O(d²) | O(d log d) |
| Memory | O(d²) — 2.3 MB @ d=768 | O(d) — 4 KB @ d=768 |
| Quality | Exact Haar distribution | Near-Haar (JL guarantees) |
| Deterministic | Yes (seed-based) | Yes (seed-based) |

```rust
use bitpolar::wht::WhtRotation;
use bitpolar::traits::RotationStrategy;

let wht = WhtRotation::new(768, 42).unwrap();
let rotated = wht.rotate(&embedding);
let recovered = wht.rotate_inverse(&rotated);
```

## API Overview

### Core Quantizers

| Type | Description | Use Case |
|------|-------------|----------|
| `TurboQuantizer` | Two-stage (Polar + QJL) | Primary API — best quality |
| `PolarQuantizer` | Polar coordinate encoding | Simpler, fallback option |
| `QjlQuantizer` | 1-bit JL sketching | Residual correction |
| `WhtRotation` | Walsh-Hadamard rotation | Fast, memory-efficient rotation |

### Specialized Wrappers

| Type | Description |
|------|-------------|
| `KvCacheCompressor` | Transformer KV cache compression |
| `MultiHeadKvCache` | Multi-head attention KV cache |
| `TieredQuantization` | Hot (8-bit) / Warm (4-bit) / Cold (3-bit) |
| `ResilientQuantizer` | Primary + fallback for production robustness |
| `OversampledSearch` | Two-phase approximate + exact re-ranking |
| `DistortionTracker` | Online quality monitoring (EMA MSE/bias) |

### Language Bindings

| Package | Install | Language |
|---------|---------|----------|
| [`bitpolar`](https://crates.io/crates/bitpolar) | `cargo add bitpolar` | Rust |
| [`bitpolar`](https://pypi.org/project/bitpolar/) | `pip install bitpolar` | Python (PyO3) |
| [`@mmgehlot/bitpolar-wasm`](https://www.npmjs.com/package/@mmgehlot/bitpolar-wasm) | `npm install @mmgehlot/bitpolar-wasm` | JavaScript (WASM) |
| [`@mmgehlot/bitpolar`](https://www.npmjs.com/package/@mmgehlot/bitpolar) | `npm install @mmgehlot/bitpolar` | Node.js (NAPI-RS) |
| `bitpolar-go` | `go get github.com/mmgehlot/bitpolar/...` | Go (CGO) |
| `bitpolar` | Maven Central | Java (JNI) |
| `bitpolar-pg` | `cargo pgrx install` | PostgreSQL |

## 58 Integrations — Every Major AI Framework

BitPolar is the **single canonical library** for vector quantization across the entire AI/ML ecosystem.

### RAG & Search Frameworks

| Integration | Package | Description |
|------------|---------|-------------|
| **LangChain** | `langchain_bitpolar` | VectorStore with compressed similarity search |
| **LlamaIndex** | `llamaindex_bitpolar` | BasePydanticVectorStore for LlamaIndex |
| **Haystack** | `bitpolar_haystack` | DocumentStore + Retriever component |
| **DSPy** | `bitpolar_dspy` | Retriever module for DSPy pipelines |
| **FAISS** | `bitpolar_faiss` | Drop-in replacement for `faiss.IndexFlatIP/L2` |
| **ChromaDB** | `bitpolar_chroma` | EmbeddingFunction + two-phase search store |

### Agentic AI Frameworks

| Integration | Package | Description |
|------------|---------|-------------|
| **LangGraph** | `bitpolar_langgraph` | Compressed checkpoint saver for stateful agents |
| **CrewAI** | `bitpolar_crewai` | Memory backend for agent teams |
| **OpenAI Agents SDK** | `bitpolar_openai_agents` | Function-calling tools for OpenAI agents |
| **Google ADK** | `bitpolar_google_adk` | Tool for Google Agent Development Kit |
| **Anthropic MCP** | `bitpolar_anthropic` | MCP server (stdio + SSE) for Claude |
| **AutoGen** | `bitpolar_autogen` | Memory store for Microsoft agents |
| **SmolAgents** | `bitpolar_smolagents` | HuggingFace agent tool |
| **PydanticAI** | `bitpolar_pydantic_ai` | Type-safe Pydantic tool definitions |
| **Agno (Phidata)** | `bitpolar_agno` | Knowledge base for high-perf agents |

### Agent Memory Frameworks

| Integration | Package | Description |
|------------|---------|-------------|
| **Mem0** | `bitpolar_mem0` | Vector store backend for Mem0 |
| **Zep** | `bitpolar_zep` | Compressed store with time-decay scoring |
| **Letta (MemGPT)** | `bitpolar_letta` | Archival memory tier |

### Vector Databases

| Integration | Package | Description |
|------------|---------|-------------|
| **Qdrant** | `bitpolar_embeddings.qdrant` | Two-phase HNSW + BitPolar re-ranking |
| **Milvus** | `bitpolar_milvus` | Client-side compression with reranking |
| **Weaviate** | `bitpolar_weaviate` | Client-side compression with reranking |
| **Pinecone** | `bitpolar_pinecone` | Metadata-stored compressed codes |
| **Redis** | `bitpolar_redis` | Byte string storage with pipeline search |
| **Elasticsearch** | `bitpolar_elasticsearch` | kNN search + BitPolar reranking |
| **PostgreSQL** | `bitpolar-pg` | Native pgrx extension (SQL functions) |
| **DuckDB** | `bitpolar_duckdb` | BLOB storage with SQL queries |
| **SQLite** | `bitpolar_sqlite_vec` | Zero-dependency embedded vector search |
| **Supabase** | `bitpolar_supabase` | Serverless pgvector compression |
| **Neon** | `bitpolar_neon` | Serverless Postgres driver |

### LLM Inference Engines (KV Cache)

| Integration | Package | Description |
|------------|---------|-------------|
| **vLLM** | `bitpolar_vllm` | KV cache quantizer + DynamicCache |
| **HuggingFace Transformers** | `bitpolar_transformers` | Drop-in DynamicCache replacement |
| **llama.cpp** | `bitpolar_llamacpp` | KV cache compression |
| **SGLang** | `bitpolar_sglang` | RadixAttention cache compression |
| **TensorRT-LLM** | `bitpolar_tensorrt` | KV cache quantizer plugin |
| **Ollama** | `bitpolar_ollama` | Embedding compression client |
| **ONNX Runtime** | `bitpolar_onnx` | Model embedding quantizer |
| **Apple MLX** | `bitpolar_mlx` | Apple Silicon quantizer |

### ML Frameworks

| Integration | Package | Description |
|------------|---------|-------------|
| **PyTorch** | `bitpolar_torch` | Embedding quantizer, BitPolarLinear, KV cache |
| **PyTorch (native)** | `bitpolar_torch_native` | PT2E quantizer backend |
| **JAX/Flax** | `bitpolar_jax` | JAX array compression + Flax module |
| **TensorFlow** | `bitpolar_tensorflow` | Keras layers for compression |
| **scikit-learn** | `bitpolar_sklearn` | TransformerMixin for sklearn pipelines |

### Cloud & Enterprise

| Integration | Package | Description |
|------------|---------|-------------|
| **Spring AI** | `BitPolarVectorStore.java` | Java VectorStore for Spring Boot |
| **Vercel AI SDK** | `bitpolar_vercel` | Embedding compression middleware |
| **AWS Bedrock** | `bitpolar_bedrock` | Titan/Cohere embedding compression |
| **Triton** | `bitpolar_triton` | NVIDIA Inference Server backend |
| **gRPC** | `bitpolar-server` | Language-agnostic compression service |
| **MCP** | `bitpolar_mcp` | AI coding assistant tool server |
| **CLI** | `bitpolar-cli` | Command-line compress/search/bench |

## How It Works

```text
Input f32 vector
    │
    ▼
┌─────────────────┐
│ Random Rotation  │  WHT (O(d log d)) or Haar QR (O(d²))
│                  │  Spreads energy uniformly across coordinates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PolarQuant     │  Groups d dims into d/2 pairs → polar coords
│   (Stage 1)      │  Radii: lossless f32 │ Angles: b-bit quantized
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  QJL Residual    │  Sketches reconstruction error
│   (Stage 2)      │  1 sign bit per projection → unbiased correction
└────────┬────────┘
         │
         ▼
TurboCode { polar: PolarCode, residual: QjlSketch }
```

**Inner product estimation**: `⟨v, q⟩ ≈ IP_polar(code, q) + IP_qjl(sketch, q)`

## Parameter Selection

| Use Case | Bits | Projections | Notes |
|----------|------|-------------|-------|
| Semantic search | 4-8 | dim/4 | Best accuracy for retrieval |
| KV cache | 3-6 | dim/8 | Memory vs attention quality |
| Maximum compression | 3 | dim/16 | Still provably unbiased |
| Lightweight similarity | — | dim/4 | QJL standalone (1-bit sketches) |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library (nalgebra QR, full rotation) |
| `alloc` | No | Heap allocation without std (Vec via alloc crate) |
| `serde-support` | Yes | Serde serialization for all types |
| `simd` | No | Hand-tuned NEON/AVX2 kernels |
| `parallel` | No | Parallel batch operations via rayon |
| `tracing-support` | No | OpenTelemetry-compatible instrumentation |
| `ffi` | No | C FFI exports for cross-language bindings |

### `no_std` Support

BitPolar works on embedded/edge targets with `no_std`:

```toml
[dependencies]
bitpolar = { version = "0.2", default-features = false, features = ["alloc"] }
```

Uses `libm` for math functions and `alloc` for `Vec`/`String`. The Walsh-Hadamard rotation is available without `std` (unlike Haar QR which requires `nalgebra`).

## Traits

BitPolar exposes composable traits for ecosystem integration:

- **`VectorQuantizer`** — core encode/decode/IP/L2 interface
- **`BatchQuantizer`** — parallel batch operations (behind `parallel` feature)
- **`RotationStrategy`** — pluggable rotation (QR, Walsh-Hadamard, identity)
- **`SerializableCode`** — compact binary serialization

## Examples

**30 Python examples** + 9 Rust examples + JavaScript, Go, Java examples.

```bash
# Rust
cargo run --example search_vector_database
cargo run --example llm_kv_cache

# Python (30 examples covering all 58 integrations)
python examples/python/01_quickstart.py           # Core API
python examples/python/12_pytorch_quantizer.py     # PyTorch integration
python examples/python/13_llamaindex_vectorstore.py # LlamaIndex
python examples/python/14_faiss_dropin.py          # FAISS replacement
python examples/python/18_openai_agents_tool.py    # OpenAI Agents
python examples/python/23_vector_databases.py      # DuckDB, SQLite, etc.
python examples/python/30_complete_rag.py          # End-to-end RAG pipeline
```

See [`examples/README.md`](examples/README.md) for the full list.

## Benchmarks

Independently verified on standard datasets following [MLCommons](https://mlcommons.org/) and [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks) methodology. All results reproducible with `seed=42`. Full details in [`BENCHMARKS.md`](BENCHMARKS.md).

### Encode Throughput

Single-vector compression speed (Rust native, 4-bit, Criterion.rs, 100 samples):

```
Dimension    Vectors/sec
   64    ████████████████████████████████████████  169,000
  128    █████████████                              40,000
  256    ████                                        9,400
  512    ██                                          2,500
 1024    █                                             585
```

### Inner Product Scoring Speed

Approximate inner product estimation on compressed codes (no decompression):

```
Dimension    Ops/sec
   64    ████████████████████████████████████████  204,000
  128    █████████████████                          45,700
  256    █████                                      10,900
  512    ██                                          2,580
 1024    █                                             614
```

### KV Cache Attention Fidelity

Cosine similarity between exact and approximate attention output (32 heads, 512 seq_len, 128 head_dim):

```
Bits    Cosine Similarity
  6     ████████████████████████████████████████  0.9931
  4     █████████████████████████████████████     0.9206
  3     █████████████████████████████████         0.8143
        ├───────┼───────┼───────┼───────┼───────┤
        0.0    0.2    0.4    0.6    0.8    1.0
```

### Inner Product Bias Verification

BitPolar guarantees **provably unbiased** inner product estimates. Verified on 10K random pairs:

| Bits | Pearson Correlation | Mean Bias | MSE |
|------|:-------------------:|:---------:|:---:|
| **6** | 0.9797 | 0.017 | 5.39 |
| **4** | 0.8244 | -0.020 | 60.89 |
| **3** | 0.6678 | -0.028 | 158.55 |

> Mean bias ≈ 0 at all bit-widths, confirming the theoretical guarantee from [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874): E[estimate] = exact.

### Operation Latency

| Operation | dim=128 | dim=384 | dim=768 |
|-----------|:-------:|:-------:|:-------:|
| **Encode** | 25 µs | 106 µs | 403 µs |
| **Decode** | 19 µs | 93 µs | 582 µs |
| **IP Score** | 22 µs | 92 µs | 388 µs |
| **Serialize** | 488 ns | 798 ns | 846 ns |
| **Deserialize** | 364 ns | 553 ns | 745 ns |

### Compression Analysis

| Dimension | Original | Compressed | Ratio |
|:---------:|:--------:|:----------:|:-----:|
| 128 | 512 B | 404 B | 1.27x |
| 384 | 1,536 B | 1,180 B | 1.30x |
| 768 | 3,072 B | 2,344 B | 1.31x |
| 1,536 | 6,144 B | 4,672 B | 1.32x |

> The compact format stores lossless f32 radii + quantized angles + QJL sketch. For KV cache applications, the paper's "6x compression" refers to bits-per-coordinate (angle quantization only).

### Benchmark Methodology

| Standard | How BitPolar Follows It |
|----------|------------------------|
| **Reproducibility** | Deterministic seed=42, all parameters documented |
| **Statistical rigor** | Criterion.rs: 100 samples, 3s warmup, outlier detection |
| **Ground truth** | Exact brute-force inner product (no approximation in evaluation) |
| **Hardware transparency** | CPU/memory/OS reported in [`BENCHMARKS.md`](BENCHMARKS.md) |
| **Standard datasets** | SIFT-1M, GloVe-200 via [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks) |
| **Baseline comparisons** | f32 exact search as upper bound |

### Reproducing Benchmarks

```bash
# Rust benchmarks (recommended — native speed, Criterion.rs)
cargo bench --bench quantization_bench    # Encode/decode/IP latency
cargo bench --bench search_bench          # Search patterns + f32 baseline
cargo bench --bench dataset_bench         # Recall@10 + throughput scaling

# Python benchmarks (requires: cd bitpolar-python && maturin develop --release)
python benchmarks/bench_compression.py    # Compression ratio table
python benchmarks/bench_kv_cache.py       # Attention fidelity + IP bias
python benchmarks/bench_vs_faiss.py       # Head-to-head vs FAISS PQ/SQ

# Full results document
python benchmarks/generate_results.py     # → BENCHMARKS.md
```

> **Note on recall:** Recall@10 on uniform random data is ~0.3-0.4 due to the curse of dimensionality (random vectors are nearly orthogonal). Real-world embeddings with semantic structure achieve significantly higher recall — the TurboQuant paper reports **Recall@10 > 0.95 on GloVe** at 4-bit.

## References

- **TurboQuant** (ICLR 2026): [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant** (AISTATS 2026): [arXiv 2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL** (AAAI 2025): [arXiv 2406.03482](https://arxiv.org/abs/2406.03482)

## Star History

<a href="https://star-history.com/#mmgehlot/bitpolar&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mmgehlot/bitpolar&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mmgehlot/bitpolar&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mmgehlot/bitpolar&type=Date" />
 </picture>
</a>

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and how to add a new quantization strategy.

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
