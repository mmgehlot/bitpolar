#!/usr/bin/env python3
"""Head-to-head benchmark: BitPolar vs FAISS.

Compares BitPolar's TurboQuantizer against FAISS baselines on a synthetic
100K-vector dataset with 1000 queries. Reports recall@10, QPS, memory,
and index build time for each method.

Usage:
    python benchmarks/bench_vs_faiss.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

SEED = 42
N_BASE = 100_000
N_QUERY = 1_000
DIM = 128
TOP_K = 10


def generate_dataset():
    """Generate L2-normalised random float32 vectors."""
    rng = np.random.RandomState(SEED)

    base = rng.randn(N_BASE, DIM).astype(np.float32)
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    base /= norms

    queries = rng.randn(N_QUERY, DIM).astype(np.float32)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    queries /= q_norms

    return base, queries


def compute_ground_truth(base, queries, k=TOP_K):
    """Exact top-k by inner product using numpy."""
    print(f"  Computing ground truth (exact top-{k} over {N_BASE} vectors)...")
    gt = np.empty((queries.shape[0], k), dtype=np.int64)
    batch = 256
    for i in range(0, queries.shape[0], batch):
        end = min(i + batch, queries.shape[0])
        scores = queries[i:end] @ base.T  # (batch, N_BASE)
        for j in range(end - i):
            top_idx = np.argpartition(scores[j], -k)[-k:]
            top_idx = top_idx[np.argsort(scores[j][top_idx])[::-1]]
            gt[i + j] = top_idx
    return gt


def recall_at_k(gt, predicted, k=TOP_K):
    """Mean recall@k: fraction of true top-k items found in predicted top-k."""
    assert gt.shape[0] == predicted.shape[0]
    hits = 0
    for i in range(gt.shape[0]):
        hits += len(set(gt[i, :k].tolist()) & set(predicted[i, :k].tolist()))
    return hits / (gt.shape[0] * k)


# ---------------------------------------------------------------------------
# FAISS methods
# ---------------------------------------------------------------------------

def bench_faiss_flat(base, queries, gt):
    """FAISS IndexFlatIP — exact brute-force baseline."""
    import faiss

    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(DIM)
    index.add(base)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores, ids = index.search(queries, TOP_K)
    search_time = time.perf_counter() - t0

    rec = recall_at_k(gt, ids, TOP_K)
    qps = N_QUERY / search_time
    mem_per_vec = DIM * 4  # float32

    return {
        "method": "FAISS FlatIP",
        "build_time_s": round(build_time, 4),
        "memory_per_vector_bytes": mem_per_vec,
        "recall_at_10": round(rec, 4),
        "qps": round(qps, 1),
    }


def bench_faiss_pq(base, queries, gt):
    """FAISS IndexPQ — product quantization, 16 subquantizers, 8-bit codes."""
    import faiss

    n_subq = 16
    n_bits = 8

    t0 = time.perf_counter()
    index = faiss.IndexPQ(DIM, n_subq, n_bits)
    # PQ requires training
    train_vecs = base[:10_000]
    index.train(train_vecs)
    index.add(base)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores, ids = index.search(queries, TOP_K)
    search_time = time.perf_counter() - t0

    rec = recall_at_k(gt, ids, TOP_K)
    qps = N_QUERY / search_time
    mem_per_vec = n_subq * (n_bits // 8)  # 16 bytes

    return {
        "method": "FAISS PQ(16x8)",
        "build_time_s": round(build_time, 4),
        "memory_per_vector_bytes": mem_per_vec,
        "recall_at_10": round(rec, 4),
        "qps": round(qps, 1),
    }


def bench_faiss_sq(base, queries, gt):
    """FAISS IndexScalarQuantizer — 8-bit scalar quantization."""
    import faiss

    t0 = time.perf_counter()
    index = faiss.IndexScalarQuantizer(
        DIM, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    index.train(base[:10_000])
    index.add(base)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores, ids = index.search(queries, TOP_K)
    search_time = time.perf_counter() - t0

    rec = recall_at_k(gt, ids, TOP_K)
    qps = N_QUERY / search_time
    mem_per_vec = DIM  # 1 byte per dimension

    return {
        "method": "FAISS SQ8",
        "build_time_s": round(build_time, 4),
        "memory_per_vector_bytes": mem_per_vec,
        "recall_at_10": round(rec, 4),
        "qps": round(qps, 1),
    }


# ---------------------------------------------------------------------------
# BitPolar methods
# ---------------------------------------------------------------------------

def bench_bitpolar(base, queries, gt, bits):
    """BitPolar TurboQuantizer at the given bit-width."""
    import bitpolar

    projections = DIM // 4

    t0 = time.perf_counter()
    quantizer = bitpolar.TurboQuantizer(
        dim=DIM, bits=bits, projections=projections, seed=SEED
    )
    codes = []
    for i in range(N_BASE):
        codes.append(quantizer.encode(base[i]))
    build_time = time.perf_counter() - t0

    # Measure bytes from the actual encoded data
    total_bytes = sum(len(c) for c in codes)
    mem_per_vec = total_bytes / N_BASE

    # Search: score each query against all codes
    t0 = time.perf_counter()
    predicted = np.empty((N_QUERY, TOP_K), dtype=np.int64)
    for qi in range(N_QUERY):
        q = queries[qi]
        scores_arr = np.empty(N_BASE, dtype=np.float32)
        for vi, code in enumerate(codes):
            scores_arr[vi] = quantizer.inner_product(code, q)
        top_idx = np.argpartition(scores_arr, -TOP_K)[-TOP_K:]
        top_idx = top_idx[np.argsort(scores_arr[top_idx])[::-1]]
        predicted[qi] = top_idx
    search_time = time.perf_counter() - t0

    rec = recall_at_k(gt, predicted, TOP_K)
    qps = N_QUERY / search_time

    return {
        "method": f"BitPolar {bits}-bit",
        "build_time_s": round(build_time, 4),
        "memory_per_vector_bytes": round(mem_per_vec, 1),
        "recall_at_10": round(rec, 4),
        "qps": round(qps, 1),
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(results):
    """Print a formatted comparison table."""
    header = f"{'Method':<20} {'Build(s)':>10} {'Bytes/Vec':>10} {'Recall@10':>10} {'QPS':>12}"
    print("\n" + "=" * len(header))
    print("BitPolar vs FAISS Benchmark")
    print(f"Dataset: {N_BASE} vectors, dim={DIM}, {N_QUERY} queries, top-{TOP_K}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['method']:<20} "
            f"{r['build_time_s']:>10.4f} "
            f"{r['memory_per_vector_bytes']:>10.1f} "
            f"{r['recall_at_10']:>10.4f} "
            f"{r['qps']:>12.1f}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating dataset...")
    base, queries = generate_dataset()
    gt = compute_ground_truth(base, queries)

    results = []

    # --- FAISS methods ---
    has_faiss = True
    try:
        import faiss
    except ImportError:
        has_faiss = False
        print("\n[WARN] faiss not installed. Install with: pip install faiss-cpu")
        print("       Running BitPolar-only benchmarks.\n")

    if has_faiss:
        print("\nBenchmarking FAISS FlatIP...")
        results.append(bench_faiss_flat(base, queries, gt))

        print("Benchmarking FAISS PQ(16x8)...")
        results.append(bench_faiss_pq(base, queries, gt))

        print("Benchmarking FAISS SQ8...")
        results.append(bench_faiss_sq(base, queries, gt))

    # --- BitPolar methods ---
    try:
        import bitpolar
    except ImportError:
        print("[ERROR] bitpolar not installed. Install with: pip install bitpolar")
        if not results:
            sys.exit(1)
        print_table(results)
        return

    print("Benchmarking BitPolar 4-bit...")
    results.append(bench_bitpolar(base, queries, gt, bits=4))

    print("Benchmarking BitPolar 3-bit...")
    results.append(bench_bitpolar(base, queries, gt, bits=3))

    # --- Output ---
    print_table(results)

    # Save JSON
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vs_faiss.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "n_base": N_BASE,
                    "n_query": N_QUERY,
                    "dim": DIM,
                    "top_k": TOP_K,
                    "seed": SEED,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
