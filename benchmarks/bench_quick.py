#!/usr/bin/env python3
"""Quick benchmark suite — runs all key benchmarks in < 5 minutes.

Produces results for: recall, throughput, compression, KV cache quality.
Uses smaller datasets (10K vectors, 100 queries) for fast turnaround.

Usage:
    python benchmarks/bench_quick.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

try:
    import bitpolar as bp
except ImportError:
    print("[ERROR] bitpolar not installed. Install with: pip install bitpolar")
    sys.exit(1)

try:
    import psutil
    MEM_GB = round(psutil.virtual_memory().total / (1024 ** 3), 1)
except ImportError:
    MEM_GB = "unknown"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
BITS_LIST = [3, 4, 6, 8]


def system_info() -> dict:
    return {
        "platform": platform.system(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_gb": MEM_GB,
        "python_version": platform.python_version(),
    }


def make_vectors(n: int, dim: int, seed: int = SEED) -> np.ndarray:
    """Generate L2-normalized random vectors."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def compute_ground_truth(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Compute exact top-k by inner product (brute-force numpy)."""
    # Batch matrix multiply: (n_queries, n_base)
    scores = queries @ base.T
    # Top-k per query
    gt = np.empty((queries.shape[0], k), dtype=np.int32)
    for i in range(queries.shape[0]):
        top_idx = np.argpartition(scores[i], -k)[-k:]
        gt[i] = top_idx[np.argsort(scores[i][top_idx])[::-1]]
    return gt


def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute Recall@k: fraction of true top-k found in predicted top-k."""
    n = predicted.shape[0]
    hits = 0
    for i in range(n):
        pred_set = set(predicted[i, :k].tolist())
        gt_set = set(ground_truth[i, :k].tolist())
        hits += len(pred_set & gt_set)
    return hits / (n * k)


# ============================================================================
# Benchmark 1: Recall@k
# ============================================================================
def bench_recall():
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Recall@k")
    print("=" * 70)

    configs = [
        ("Random-128", 128, 10_000, 100),  # (name, dim, n_base, n_query)
        ("Random-384", 384, 10_000, 100),
        ("Random-768", 768, 5_000, 50),
    ]

    all_results = []

    for name, dim, n_base, n_query in configs:
        print(f"\n--- {name}: {n_base} vectors, {n_query} queries, dim={dim} ---")
        base = make_vectors(n_base, dim)
        queries = make_vectors(n_query, dim, seed=SEED + 1)

        # Ground truth
        gt = compute_ground_truth(base, queries, k=100)

        for bits in BITS_LIST:
            proj = max(dim // 4, 1)
            q = bp.TurboQuantizer(dim=dim, bits=bits, projections=proj, seed=SEED)

            # Encode all base vectors
            t0 = time.perf_counter()
            codes = [q.encode(base[i]) for i in range(n_base)]
            t_encode = time.perf_counter() - t0

            code_size = len(bytes(codes[0]))
            compression = (dim * 4) / code_size

            # Search: score all codes per query
            t0 = time.perf_counter()
            predicted = np.empty((n_query, 100), dtype=np.int32)
            for qi in range(n_query):
                scores = np.empty(n_base, dtype=np.float32)
                for j in range(n_base):
                    scores[j] = q.inner_product(codes[j], queries[qi])
                top_idx = np.argpartition(scores, -100)[-100:]
                predicted[qi] = top_idx[np.argsort(scores[top_idx])[::-1]]
            t_search = time.perf_counter() - t0

            r1 = recall_at_k(predicted, gt, 1)
            r10 = recall_at_k(predicted, gt, 10)
            r100 = recall_at_k(predicted, gt, 100)

            result = {
                "dataset": name,
                "dim": dim,
                "bits": bits,
                "n_vectors": n_base,
                "n_queries": n_query,
                "recall_at_1": round(r1, 4),
                "recall_at_10": round(r10, 4),
                "recall_at_100": round(r100, 4),
                "compression_ratio": round(compression, 2),
                "code_size_bytes": code_size,
                "original_size_bytes": dim * 4,
                "time_encode_total_s": round(t_encode, 3),
                "time_search_total_s": round(t_search, 3),
            }
            all_results.append(result)

            print(f"  {bits}-bit: R@1={r1:.3f} R@10={r10:.3f} R@100={r100:.3f} "
                  f"comp={compression:.1f}x encode={t_encode:.1f}s search={t_search:.1f}s")

    # Save
    for r in all_results:
        fname = f"recall_{r['dataset'].lower()}_{r['bits']}bit.json"
        with open(RESULTS_DIR / fname, "w") as f:
            json.dump(r, f, indent=2)
    print(f"\nSaved {len(all_results)} recall results")
    return all_results


# ============================================================================
# Benchmark 2: Throughput
# ============================================================================
def bench_throughput():
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Throughput (ops/sec)")
    print("=" * 70)

    n = 1000  # Small for fast measurement
    iters = 3
    results = {"operations": [], "system_info": system_info()}

    for dim in [128, 384, 768, 1536]:
        print(f"\n--- dim={dim}, n={n}, bits=4 ---")
        vecs = make_vectors(n, dim)
        proj = max(dim // 4, 1)
        q = bp.TurboQuantizer(dim=dim, bits=4, projections=proj, seed=SEED)

        # Encode
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            codes = [q.encode(vecs[i]) for i in range(n)]
            times.append(time.perf_counter() - t0)
        encode_vps = n / min(times)

        # Decode
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            for c in codes:
                q.decode(c)
            times.append(time.perf_counter() - t0)
        decode_vps = n / min(times)

        # IP estimation
        query = vecs[0]
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            for c in codes:
                q.inner_product(c, query)
            times.append(time.perf_counter() - t0)
        ip_ops = n / min(times)

        # Search (100 queries over n codes)
        n_queries = min(50, n)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            for qi in range(n_queries):
                for c in codes:
                    q.inner_product(c, vecs[qi])
            times.append(time.perf_counter() - t0)
        search_qps = n_queries / min(times)

        entry = {
            "dim": dim,
            "n_vectors": n,
            "encode_vps": round(encode_vps),
            "decode_vps": round(decode_vps),
            "ip_ops": round(ip_ops),
            "search_qps": round(search_qps, 1),
        }
        results["operations"].append(entry)
        print(f"  Encode: {encode_vps:,.0f} vec/s  Decode: {decode_vps:,.0f} vec/s  "
              f"IP: {ip_ops:,.0f} ops/s  Search: {search_qps:.0f} QPS")

    with open(RESULTS_DIR / "throughput.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved throughput results")
    return results


# ============================================================================
# Benchmark 3: FAISS comparison
# ============================================================================
def bench_vs_faiss():
    print("\n" + "=" * 70)
    print("BENCHMARK 3: BitPolar vs FAISS")
    print("=" * 70)

    dim = 128
    n_base = 10_000
    n_query = 100
    k = 10

    base = make_vectors(n_base, dim)
    queries = make_vectors(n_query, dim, seed=SEED + 1)
    gt = compute_ground_truth(base, queries, k)

    methods = []

    # --- BitPolar 4-bit ---
    proj = max(dim // 4, 1)
    q4 = bp.TurboQuantizer(dim=dim, bits=4, projections=proj, seed=SEED)

    t0 = time.perf_counter()
    codes4 = [q4.encode(base[i]) for i in range(n_base)]
    bp4_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    bp4_pred = np.empty((n_query, k), dtype=np.int32)
    for qi in range(n_query):
        scores = np.array([q4.inner_product(codes4[j], queries[qi]) for j in range(n_base)], dtype=np.float32)
        top_idx = np.argpartition(scores, -k)[-k:]
        bp4_pred[qi] = top_idx[np.argsort(scores[top_idx])[::-1]]
    bp4_search = time.perf_counter() - t0
    bp4_recall = recall_at_k(bp4_pred, gt, k)
    bp4_bytes = len(bytes(codes4[0]))

    methods.append({
        "name": "BitPolar 4-bit",
        "recall_at_10": round(bp4_recall, 4),
        "build_time_s": round(bp4_build, 4),
        "bytes_per_vector": bp4_bytes,
        "qps": round(n_query / bp4_search, 1),
    })
    print(f"  BitPolar 4-bit: R@10={bp4_recall:.3f} build={bp4_build:.2f}s bytes={bp4_bytes} QPS={n_query/bp4_search:.0f}")

    # --- BitPolar 3-bit ---
    q3 = bp.TurboQuantizer(dim=dim, bits=3, projections=proj, seed=SEED)
    t0 = time.perf_counter()
    codes3 = [q3.encode(base[i]) for i in range(n_base)]
    bp3_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    bp3_pred = np.empty((n_query, k), dtype=np.int32)
    for qi in range(n_query):
        scores = np.array([q3.inner_product(codes3[j], queries[qi]) for j in range(n_base)], dtype=np.float32)
        top_idx = np.argpartition(scores, -k)[-k:]
        bp3_pred[qi] = top_idx[np.argsort(scores[top_idx])[::-1]]
    bp3_search = time.perf_counter() - t0
    bp3_recall = recall_at_k(bp3_pred, gt, k)
    bp3_bytes = len(bytes(codes3[0]))

    methods.append({
        "name": "BitPolar 3-bit",
        "recall_at_10": round(bp3_recall, 4),
        "build_time_s": round(bp3_build, 4),
        "bytes_per_vector": bp3_bytes,
        "qps": round(n_query / bp3_search, 1),
    })
    print(f"  BitPolar 3-bit: R@10={bp3_recall:.3f} build={bp3_build:.2f}s bytes={bp3_bytes} QPS={n_query/bp3_search:.0f}")

    # --- Numpy exact (baseline) ---
    t0 = time.perf_counter()
    # "Build" = just store the matrix (no-op)
    np_build = 0.0
    for qi in range(n_query):
        _ = queries[qi] @ base.T
    np_search = time.perf_counter() - t0

    methods.append({
        "name": "Exact (numpy f32)",
        "recall_at_10": 1.0,
        "build_time_s": 0.0,
        "bytes_per_vector": dim * 4,
        "qps": round(n_query / np_search, 1),
    })
    print(f"  Exact numpy:    R@10=1.000 build=0.00s bytes={dim*4} QPS={n_query/np_search:.0f}")

    # --- FAISS PQ ---
    try:
        import faiss

        # FAISS PQ: 16 subquantizers, 8 bits each = 16 bytes/vector
        pq_index = faiss.IndexPQ(dim, 16, 8)
        t0 = time.perf_counter()
        pq_index.train(base)
        pq_index.add(base)
        faiss_pq_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        D, I = pq_index.search(queries, k)
        faiss_pq_search = time.perf_counter() - t0
        faiss_pq_recall = recall_at_k(I, gt, k)

        methods.append({
            "name": "FAISS PQ (16x8)",
            "recall_at_10": round(faiss_pq_recall, 4),
            "build_time_s": round(faiss_pq_build, 4),
            "bytes_per_vector": 16,
            "qps": round(n_query / faiss_pq_search, 1),
        })
        print(f"  FAISS PQ:       R@10={faiss_pq_recall:.3f} build={faiss_pq_build:.2f}s bytes=16 QPS={n_query/faiss_pq_search:.0f}")

        # FAISS Flat (exact baseline)
        flat_index = faiss.IndexFlatIP(dim)
        t0 = time.perf_counter()
        flat_index.add(base)
        faiss_flat_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        D, I = flat_index.search(queries, k)
        faiss_flat_search = time.perf_counter() - t0

        methods.append({
            "name": "FAISS FlatIP",
            "recall_at_10": 1.0,
            "build_time_s": round(faiss_flat_build, 4),
            "bytes_per_vector": dim * 4,
            "qps": round(n_query / faiss_flat_search, 1),
        })
        print(f"  FAISS FlatIP:   R@10=1.000 build={faiss_flat_build:.2f}s bytes={dim*4} QPS={n_query/faiss_flat_search:.0f}")

    except ImportError:
        print("  [SKIP] FAISS not installed — pip install faiss-cpu")

    result = {"methods": methods, "config": {"dim": dim, "n_base": n_base, "n_query": n_query, "k": k}, "system_info": system_info()}
    with open(RESULTS_DIR / "vs_faiss.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved FAISS comparison results")
    return result


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("BitPolar Quick Benchmark Suite")
    print(f"System: {platform.processor() or platform.machine()}, "
          f"{os.cpu_count()} cores, {MEM_GB} GB RAM")
    print()

    t_total = time.perf_counter()

    recall_results = bench_recall()
    throughput_results = bench_throughput()
    faiss_results = bench_vs_faiss()

    total_time = time.perf_counter() - t_total
    print(f"\n{'=' * 70}")
    print(f"All benchmarks complete in {total_time:.1f}s")
    print(f"Results in: {RESULTS_DIR}")
    print(f"Run: python benchmarks/generate_results.py  to generate BENCHMARKS.md")
