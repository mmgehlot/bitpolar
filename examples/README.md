# BitPolar Examples

Comprehensive examples for every language, integration, and use case — **30 Python examples** across 58 integrations.

## Rust Examples

Run with `cargo run --example <name>`:

### Search Patterns

| Example | Description |
|---|---|
| `search_vector_database` | Database compression + approximate nearest neighbor search |
| `search_brute_force` | Simplest pattern: encode all, scan all, return top-k |
| `search_oversampled` | Two-phase: approximate scoring + exact re-ranking for high recall |

### LLM / KV Cache

| Example | Description |
|---|---|
| `llm_kv_cache` | Single-head KV cache compression for transformers |
| `llm_multi_head_cache` | Multi-head attention KV cache (Llama-style) |

### Architecture Patterns

| Example | Description |
|---|---|
| `pattern_multi_tenant` | Per-tenant seed isolation for SaaS vector databases |
| `pattern_federated_search` | Independent nodes quantize separately, codes are comparable |

### Integrations

| Example | Description |
|---|---|
| `integration_tiered_storage` | Hot/Warm/Cold tiers with different bit-widths |
| `integration_adaptive` | Per-vector adaptive bit-width with promote/demote |

---

## Python Examples

Run with `python examples/python/<file>`:

### Core & Existing Integrations (01-11)

| # | Example | Integration | Prerequisites |
|---|---|---|---|
| 01 | [`01_quickstart.py`](python/01_quickstart.py) | Core BitPolar | `pip install bitpolar numpy` |
| 02 | [`02_huggingface_embeddings.py`](python/02_huggingface_embeddings.py) | HuggingFace sentence-transformers | `pip install bitpolar sentence-transformers` |
| 03 | [`03_langchain_vectorstore.py`](python/03_langchain_vectorstore.py) | LangChain VectorStore | `pip install bitpolar langchain-core` |
| 04 | [`04_qdrant_integration.py`](python/04_qdrant_integration.py) | Qdrant two-phase search | `pip install bitpolar qdrant-client` |
| 05 | [`05_vllm_kv_cache.py`](python/05_vllm_kv_cache.py) | vLLM KV cache + HF DynamicCache | `pip install bitpolar numpy` |
| 06 | [`06_agent_memory.py`](python/06_agent_memory.py) | AI agent episodic memory | `pip install bitpolar numpy` |
| 07 | [`07_mcp_tool_server.py`](python/07_mcp_tool_server.py) | MCP Tool for AI agents | `pip install bitpolar numpy` |
| 08 | [`08_finetuning.py`](python/08_finetuning.py) | Quantization-aware fine-tuning | `pip install bitpolar sentence-transformers torch` |
| 09 | [`09_vector_index.py`](python/09_vector_index.py) | In-memory vector search index | `pip install bitpolar numpy` |
| 10 | [`10_grpc_client.py`](python/10_grpc_client.py) | gRPC compression service | `pip install grpcio grpcio-tools` |
| 11 | [`11_cli_usage.sh`](python/11_cli_usage.sh) | CLI tool (compress/search/bench) | `cargo install --path bitpolar-cli` |

### ML Frameworks & Search (12-15)

| # | Example | Integration | Prerequisites |
|---|---|---|---|
| 12 | [`12_pytorch_quantizer.py`](python/12_pytorch_quantizer.py) | PyTorch torchao quantizer | `pip install bitpolar torch` |
| 13 | [`13_llamaindex_vectorstore.py`](python/13_llamaindex_vectorstore.py) | LlamaIndex VectorStore | `pip install bitpolar llama-index-core` |
| 14 | [`14_faiss_dropin.py`](python/14_faiss_dropin.py) | FAISS drop-in replacement | `pip install bitpolar numpy` |
| 15 | [`15_chromadb_integration.py`](python/15_chromadb_integration.py) | ChromaDB integration | `pip install bitpolar chromadb` |

### Agentic AI Frameworks (16-22)

| # | Example | Integration | Prerequisites |
|---|---|---|---|
| 16 | [`16_langgraph_checkpointer.py`](python/16_langgraph_checkpointer.py) | LangGraph checkpointer | `pip install bitpolar langgraph-checkpoint` |
| 17 | [`17_crewai_memory.py`](python/17_crewai_memory.py) | CrewAI memory backend | `pip install bitpolar` |
| 18 | [`18_openai_agents_tool.py`](python/18_openai_agents_tool.py) | OpenAI Agents SDK tool | `pip install bitpolar` |
| 19 | [`19_haystack_pipeline.py`](python/19_haystack_pipeline.py) | Haystack DocumentStore | `pip install bitpolar haystack-ai` |
| 20 | [`20_dspy_retriever.py`](python/20_dspy_retriever.py) | DSPy retriever module | `pip install bitpolar dspy` |
| 21 | [`21_mem0_backend.py`](python/21_mem0_backend.py) | Mem0 vector store | `pip install bitpolar` |
| 22 | [`22_sklearn_pipeline.py`](python/22_sklearn_pipeline.py) | scikit-learn Pipeline | `pip install bitpolar scikit-learn` |

### Comprehensive Integration Demos (23-30)

| # | Example | Coverage | Prerequisites |
|---|---|---|---|
| 23 | [`23_vector_databases.py`](python/23_vector_databases.py) | DuckDB, SQLite, Milvus, Weaviate, Pinecone, Redis, ES | `pip install bitpolar duckdb` |
| 24 | [`24_llm_inference.py`](python/24_llm_inference.py) | llama.cpp, SGLang, HF Transformers KV cache | `pip install bitpolar numpy` |
| 25 | [`25_ml_frameworks.py`](python/25_ml_frameworks.py) | PyTorch, JAX, TensorFlow, scikit-learn patterns | `pip install bitpolar numpy` |
| 26 | [`26_agent_memory.py`](python/26_agent_memory.py) | Zep, Letta (MemGPT), Mem0 memory backends | `pip install bitpolar numpy` |
| 27 | [`27_cloud_enterprise.py`](python/27_cloud_enterprise.py) | Vercel AI SDK, AWS Bedrock, Triton | `pip install bitpolar` |
| 28 | [`28_agentic_tools.py`](python/28_agentic_tools.py) | OpenAI, Google ADK, Anthropic, SmolAgents, PydanticAI | `pip install bitpolar` |
| 29 | [`29_supabase_neon.py`](python/29_supabase_neon.py) | Supabase, Neon serverless Postgres | `pip install bitpolar` |
| 30 | [`30_complete_rag.py`](python/30_complete_rag.py) | End-to-end RAG pipeline | `pip install bitpolar numpy` |

---

## JavaScript Examples

| # | Example | Runtime | Prerequisites |
|---|---|---|---|
| 01 | [`01_quickstart.js`](javascript/01_quickstart.js) | Browser (WASM) | `wasm-pack build bitpolar-wasm --target web` |
| 02 | [`02_node_bindings.js`](javascript/02_node_bindings.js) | Node.js (NAPI-RS) | `cd bitpolar-node && npm run build` |

---

## Go Example

| Example | Prerequisites |
|---|---|
| [`main.go`](go/main.go) | `cargo build --release --features ffi` |

```bash
cd examples/go
CGO_LDFLAGS="-L../../target/release -lbitpolar" go run main.go
```

---

## Java Example

| Example | Prerequisites |
|---|---|
| [`BitPolarDemo.java`](java/BitPolarDemo.java) | `cargo build --release -p bitpolar-jni` |

```bash
javac examples/java/BitPolarDemo.java -d build
java -Djava.library.path=target/release -cp build BitPolarDemo
```

---

## Browser Demo

```bash
cd bitpolar-demo && python3 -m http.server 9090
# Open http://localhost:9090
```

---

## PostgreSQL

```sql
SELECT bitpolar_compress(ARRAY[0.1, 0.2, ...]::float4[], 4);
SELECT bitpolar_inner_product(compressed_col, query_array, 128, 4, 42)
FROM items ORDER BY 1 DESC LIMIT 10;
```

---

## gRPC Service

```bash
BITPOLAR_DIM=384 BITPOLAR_BITS=4 cargo run -p bitpolar-server
```
