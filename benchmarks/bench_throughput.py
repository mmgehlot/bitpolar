#!/usr/bin/env python3
"""Throughput benchmarks for BitPolar encode/decode/search operations.

Measures raw operational speed (vectors/sec, ops/sec, QPS) across multiple
dimensions to characterise performance scaling.

Usage:
    # Run all throughput benchmarks:
    python benchmarks/bench_throughput.py

    # Run only specific dimensions:
    python benchmarks/bench_throughput.py --dims 128 384

Results are saved as JSON under benchmarks/results/throughput.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required.  Install with:  pip install numpy")

try:
    from tabulate import tabulate
except ImportError:
    sys.exit("tabulate is required.  Install with:  pip install tabulate")

try:
    import bitpolar
except ImportError:
    sys.exit(
        "bitpolar is required.  Install with:\n"
        "  cd bitpolar-python && pip install -e .\n"
        "or\n"
        "  maturin develop --release  (from bitpolar-python/)"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_bench_dir = Path(__file__).resolve().parent
RESULTS_DIR = _bench_dir / "results"

DEFAULT_DIMS = [128, 384, 768, 1536]
N_VECTORS = 10_000
N_QUERIES_THROUGHPUT = 100
WARMUP_ITERS = 100
MEASURE_ITERS = 10
SEARCH_SCALE_DIM = 128
SEARCH_SCALE_SIZES = [1_000, 10_000, 100_000]
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vectors(n: int, dim: int, seed: int = SEED) -> np.ndarray:
    """Generate n random float32 vectors, L2-normalised."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms
    return vecs


def bench_encode(q: bitpolar.TurboQuantizer, vectors: np.ndarray,
                 warmup: int = WARMUP_ITERS, iters: int = MEASURE_ITERS) -> dict:
    """Benchmark encoding speed. Returns {median_s, vectors_per_sec}."""
    n = vectors.shape[0]

    # Warmup
    for i in range(min(warmup, n)):
        q.encode(vectors[i])

    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for i in range(n):
            q.encode(vectors[i])
        timings.append(time.perf_counter() - t0)

    med = statistics.median(timings)
    return {"median_s": round(med, 6), "vectors_per_sec": round(n / med, 1)}


def bench_decode(q: bitpolar.TurboQuantizer, codes: list,
                 warmup: int = WARMUP_ITERS, iters: int = MEASURE_ITERS) -> dict:
    """Benchmark decoding speed. Returns {median_s, vectors_per_sec}."""
    n = len(codes)

    # Warmup
    for i in range(min(warmup, n)):
        q.decode(codes[i])

    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for i in range(n):
            q.decode(codes[i])
        timings.append(time.perf_counter() - t0)

    med = statistics.median(timings)
    return {"median_s": round(med, 6), "vectors_per_sec": round(n / med, 1)}


def bench_inner_product(q: bitpolar.TurboQuantizer, codes: list,
                        query: np.ndarray,
                        warmup: int = WARMUP_ITERS,
                        iters: int = MEASURE_ITERS) -> dict:
    """Benchmark inner_product speed. Returns {median_s, ops_per_sec}."""
    n = len(codes)

    # Warmup
    for i in range(min(warmup, n)):
        q.inner_product(codes[i], query)

    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for i in range(n):
            q.inner_product(codes[i], query)
        timings.append(time.perf_counter() - t0)

    med = statistics.median(timings)
    return {"median_s": round(med, 6), "ops_per_sec": round(n / med, 1)}


def bench_search(q: bitpolar.TurboQuantizer, codes: list,
                 queries: np.ndarray, top_k: int = 10,
                 warmup: int = 10, iters: int = MEASURE_ITERS) -> dict:
    """Benchmark brute-force search (score all codes per query).

    Returns {median_s, qps, n_codes, n_queries}.
    """
    n_codes = len(codes)
    n_queries = queries.shape[0]

    # Warmup (fewer iterations since search is expensive)
    for wi in range(min(warmup, n_queries)):
        for ci in range(n_codes):
            q.inner_product(codes[ci], queries[wi])

    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for qi in range(n_queries):
            scores = np.empty(n_codes, dtype=np.float32)
            for ci in range(n_codes):
                scores[ci] = q.inner_product(codes[ci], queries[qi])
            # Simulate top-k extraction
            if n_codes >= top_k:
                np.argpartition(scores, -top_k)[-top_k:]
        timings.append(time.perf_counter() - t0)

    med = statistics.median(timings)
    qps = n_queries / med if med > 0 else float("inf")
    return {
        "median_s": round(med, 6),
        "qps": round(qps, 2),
        "n_codes": n_codes,
        "n_queries": n_queries,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BitPolar encode/decode/search throughput benchmarks."
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=DEFAULT_DIMS,
        help="Dimensions to benchmark (default: 128 384 768 1536).",
    )
    parser.add_argument(
        "--skip-scaling",
        action="store_true",
        help="Skip the search-scaling benchmark (saves time).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bits = 4
    results: dict = {"per_dim": [], "search_scaling": []}

    # ==================================================================
    # Per-dimension benchmarks
    # ==================================================================
    print("=" * 70)
    print(f"Throughput benchmarks  (bits={bits}, n_vectors={N_VECTORS}, "
          f"warmup={WARMUP_ITERS}, measure={MEASURE_ITERS} iters, median)")
    print("=" * 70)

    for dim in args.dims:
        projections = max(1, dim // 4)
        print(f"\n--- dim={dim}  projections={projections} ---")

        vectors = make_vectors(N_VECTORS, dim)
        q = bitpolar.TurboQuantizer(dim=dim, bits=bits, projections=projections, seed=SEED)

        # Encode
        enc = bench_encode(q, vectors)
        print(f"  Encode:  {enc['vectors_per_sec']:>12,.0f} vec/s  "
              f"(median {enc['median_s']:.4f}s for {N_VECTORS} vectors)")

        # Pre-encode all for decode / IP / search benchmarks
        codes = [q.encode(vectors[i]) for i in range(N_VECTORS)]

        # Decode
        dec = bench_decode(q, codes)
        print(f"  Decode:  {dec['vectors_per_sec']:>12,.0f} vec/s  "
              f"(median {dec['median_s']:.4f}s for {N_VECTORS} vectors)")

        # Inner product
        query = vectors[0]
        ip = bench_inner_product(q, codes, query)
        print(f"  IP:      {ip['ops_per_sec']:>12,.0f} ops/s  "
              f"(median {ip['median_s']:.4f}s for {N_VECTORS} ops)")

        # Search (100 queries over all 10k codes)
        search_queries = make_vectors(N_QUERIES_THROUGHPUT, dim, seed=SEED + 1)
        srch = bench_search(q, codes, search_queries, top_k=10, warmup=2)
        print(f"  Search:  {srch['qps']:>12,.1f} QPS    "
              f"(median {srch['median_s']:.4f}s for {srch['n_queries']} queries "
              f"x {srch['n_codes']} codes)")

        results["per_dim"].append({
            "dim": dim,
            "bits": bits,
            "projections": projections,
            "n_vectors": N_VECTORS,
            "code_size_bytes": q.code_size_bytes,
            "encode": enc,
            "decode": dec,
            "inner_product": ip,
            "search": srch,
        })

    # ==================================================================
    # Search scaling (dim=128, varying index size)
    # ==================================================================
    if not args.skip_scaling:
        print(f"\n{'=' * 70}")
        print(f"Search scaling  (dim={SEARCH_SCALE_DIM}, bits={bits}, "
              f"sizes={SEARCH_SCALE_SIZES})")
        print(f"{'=' * 70}")

        scale_dim = SEARCH_SCALE_DIM
        projections = max(1, scale_dim // 4)
        q = bitpolar.TurboQuantizer(
            dim=scale_dim, bits=bits, projections=projections, seed=SEED
        )

        max_n = max(SEARCH_SCALE_SIZES)
        all_vecs = make_vectors(max_n, scale_dim)
        all_codes = [q.encode(all_vecs[i]) for i in range(max_n)]
        search_queries = make_vectors(N_QUERIES_THROUGHPUT, scale_dim, seed=SEED + 1)

        for n_index in SEARCH_SCALE_SIZES:
            subset_codes = all_codes[:n_index]
            srch = bench_search(q, subset_codes, search_queries, top_k=10, warmup=2)
            print(f"  n={n_index:>8,}:  {srch['qps']:>10,.1f} QPS  "
                  f"(median {srch['median_s']:.4f}s)")
            results["search_scaling"].append({
                "dim": scale_dim,
                "bits": bits,
                "n_index": n_index,
                "n_queries": srch["n_queries"],
                "qps": srch["qps"],
                "median_s": srch["median_s"],
            })

    # ==================================================================
    # Save JSON
    # ==================================================================
    out_path = RESULTS_DIR / "throughput.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ==================================================================
    # Summary tables
    # ==================================================================
    print(f"\n{'=' * 70}")
    print("Per-dimension summary")
    print(f"{'=' * 70}\n")

    headers = ["Dim", "Code B", "Encode vec/s", "Decode vec/s", "IP ops/s", "Search QPS"]
    rows = []
    for r in results["per_dim"]:
        rows.append([
            r["dim"],
            r["code_size_bytes"],
            f"{r['encode']['vectors_per_sec']:,.0f}",
            f"{r['decode']['vectors_per_sec']:,.0f}",
            f"{r['inner_product']['ops_per_sec']:,.0f}",
            f"{r['search']['qps']:,.1f}",
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))

    if results["search_scaling"]:
        print(f"\n{'=' * 70}")
        print(f"Search scaling (dim={SEARCH_SCALE_DIM})")
        print(f"{'=' * 70}\n")

        headers = ["Index Size", "QPS", "Median (s)"]
        rows = []
        for r in results["search_scaling"]:
            rows.append([
                f"{r['n_index']:,}",
                f"{r['qps']:,.1f}",
                f"{r['median_s']:.4f}",
            ])
        print(tabulate(rows, headers=headers, tablefmt="github"))

    print("\nDone.")


if __name__ == "__main__":
    main()
