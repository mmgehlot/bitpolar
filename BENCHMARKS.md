# BitPolar Benchmark Results

> **Generated:** 2026-03-28
> **Version:** 0.3.3
> **Hardware:** Apple M-series / Intel x86_64 (results vary by CPU)
> **Methodology:** Criterion.rs with 100 samples, 3s warmup. Recall computed against exact brute-force ground truth. All runs use seed=42.

---

## Encode/Decode/Score Throughput (Rust Native)

Single-vector operations measured with Criterion.rs at 4-bit quantization, projections=dim/4.

| Dimension | Encode | Decode | IP Estimate | L2 Estimate | Serialize | Deserialize |
|-----------|--------|--------|-------------|-------------|-----------|-------------|
| 64 | 5.9 µs | 4.6 µs | 4.9 µs | 5.3 µs | 242 ns | 271 ns |
| 128 | 25 µs | 19 µs | 22 µs | 19 µs | 488 ns | 364 ns |
| 256 | 106 µs | 93 µs | 92 µs | 88 µs | 798 ns | 553 ns |
| 512 | 403 µs | 582 µs | 388 µs | 581 µs | 846 ns | 745 ns |
| 1024 | 1.71 ms | 1.99 ms | 1.63 ms | 2.03 ms | 1.26 µs | 1.03 µs |

**Derived throughput (vectors/second):**

| Dimension | Encode/s | IP Score/s | Serialize/s |
|-----------|----------|-----------|-------------|
| 64 | 169,000 | 204,000 | 4,100,000 |
| 128 | 40,000 | 45,700 | 2,049,000 |
| 256 | 9,400 | 10,900 | 1,253,000 |
| 512 | 2,500 | 2,580 | 1,182,000 |
| 1024 | 585 | 614 | 794,000 |

**Key insight:** Encoding is O(d²) due to Haar QR rotation. The Walsh-Hadamard Transform (WHT) alternative provides O(d log d) with similar quality. Serialization is sub-microsecond at all dimensions.

---

## Search Throughput (Brute-Force Scan)

Time to score all vectors in an index against a single query, then extract top-10. Synthetic L2-normalized random data.

| Dataset Config | Dim | Vectors | Time/Query | QPS |
|---------------|-----|---------|------------|-----|
| SIFT-128 | 128 | 10,000 | 222 ms | 4.5 |
| GloVe-200 | 200 | 10,000 | 584 ms | 1.7 |
| MiniLM-384 | 384 | 5,000 | 1.22 s | 0.8 |
| OpenAI-1536 | 1536 | 1,000 | 3.77 s | 0.3 |

**Note:** These are brute-force linear scans with no index structure. For production use, combine with HNSW or IVF for sub-millisecond search at million-vector scale (see OversampledSearch).

---

## Recall@10 on Synthetic Data

Measured on L2-normalized random vectors (worst case for quantization — no structure to exploit). Real-world datasets with semantic structure achieve higher recall.

| Dataset | Dim | Vectors | Bits | Recall@10 |
|---------|-----|---------|------|-----------|
| SIFT-128 | 128 | 10,000 | 4 | 0.400 |
| GloVe-200 | 200 | 10,000 | 4 | 0.300 |
| MiniLM-384 | 384 | 5,000 | 4 | 0.300 |
| OpenAI-1536 | 1536 | 1,000 | 4 | 0.400 |

**Why recall is moderate on random data:** Uniform random vectors in high dimensions are nearly orthogonal to each other (curse of dimensionality). The inner product differences between neighbors are tiny, making approximate ranking hard. Real embeddings (SIFT, GloVe, transformer outputs) have exploitable structure that quantization preserves much better — the TurboQuant paper reports Recall@10 > 0.95 on GloVe at 4-bit.

---

## KV Cache Compression Quality

Synthetic attention computation: 32 heads, 512 sequence length, 128 head_dim. Compresses K and V tensors with per-head BitPolar quantizers.

### Attention Fidelity

| Bits | Cosine Similarity | Notes |
|------|-------------------|-------|
| 6 | **0.9931** | Near-perfect attention preservation |
| 4 | **0.9206** | Good for most inference tasks |
| 3 | **0.8143** | Aggressive compression, some quality loss |

### Inner Product Preservation (10K random pairs, dim=128)

| Bits | Pearson Correlation | MSE | Mean Bias |
|------|--------------------|----|-----------|
| 6 | **0.9797** | 5.39 | 0.017 |
| 4 | **0.8244** | 60.89 | -0.020 |
| 3 | **0.6678** | 158.55 | -0.028 |

**Key insight:** The mean bias is close to zero at all bit-widths, confirming the theoretical guarantee that BitPolar's inner product estimates are **provably unbiased** (E[estimate] = exact value).

### KV Cache Memory Per Token

| Method | Bytes/Token (K+V) | Savings vs FP32 |
|--------|-------------------|-----------------|
| FP32 (baseline) | 1,024 | — |
| FP16 | 512 | 50% |
| BitPolar | 808 | 21% |

---

## Compression Ratios

Ratio of original float32 size to BitPolar compact code size.

| Dimension | Compression Ratio | Code Size | Original Size |
|-----------|-------------------|-----------|---------------|
| 96 | 1.25x | 307 B | 384 B |
| 128 | 1.27x | 404 B | 512 B |
| 200 | 1.28x | 623 B | 800 B |
| 384 | 1.30x | 1,180 B | 1,536 B |
| 768 | 1.31x | 2,344 B | 3,072 B |
| 1536 | 1.32x | 4,672 B | 6,144 B |

**About the compression ratio:** The TurboCode compact format stores full f32 radii (lossless norms) alongside quantized angles and QJL sketch bits. The radii dominate the code size. The "6x compression" from the TurboQuant paper refers to the KV cache bits-per-coordinate model where norms are factored out during scoring.

---

## Reproducing These Results

```bash
# Rust benchmarks (fastest, most accurate)
cargo bench --bench quantization_bench    # Encode/decode/IP timing
cargo bench --bench search_bench          # Search patterns + baselines
cargo bench --bench dataset_bench         # Recall@10 + search throughput

# Python benchmarks (requires building BitPolar)
cd bitpolar-python && maturin develop --release && cd ..
python benchmarks/bench_compression.py    # Compression ratios
python benchmarks/bench_kv_cache.py       # KV cache quality
python benchmarks/bench_quick.py          # All-in-one quick suite

# Generate this document
python benchmarks/generate_results.py
```

## References

- **TurboQuant** (ICLR 2026): [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant** (AISTATS 2026): [arXiv 2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL** (AAAI 2025): [arXiv 2406.03482](https://arxiv.org/abs/2406.03482)
- **ANN Benchmarks**: [github.com/erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
- **FAISS**: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
