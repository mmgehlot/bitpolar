#!/usr/bin/env python3
"""Recall@k benchmarks for BitPolar on standard ANN datasets.

Measures how well BitPolar's quantized inner-product search preserves the
true nearest-neighbor ranking at various bit widths.

Usage:
    # Run on synthetic random-384 (no download required, CI-friendly):
    python benchmarks/bench_recall.py

    # Run a specific dataset:
    python benchmarks/bench_recall.py --dataset random-384

    # Run all datasets including sift-1m and glove-200 (downloads ~500 MB):
    python benchmarks/bench_recall.py --all

    # Specify bit widths:
    python benchmarks/bench_recall.py --bits 3 4 6 8

Results are saved as JSON under benchmarks/results/.
"""

from __future__ import annotations

import argparse
import json
import os
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

# Ensure the benchmarks package is importable when run from the repo root.
_bench_dir = Path(__file__).resolve().parent
if str(_bench_dir) not in sys.path:
    sys.path.insert(0, str(_bench_dir))

from download_datasets import load_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = _bench_dir / "results"
DEFAULT_BITS = [3, 4, 6, 8]
DEFAULT_DATASETS = ["random-384"]
ALL_DATASETS = ["random-384", "sift-1m", "glove-200"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute Recall@k averaged over all queries.

    Args:
        predicted: (n_queries, top_k_predicted) indices from quantized search.
        ground_truth: (n_queries, >=k) true nearest-neighbor indices.
        k: Number of neighbors to evaluate.

    Returns:
        Mean recall in [0, 1].
    """
    n_queries = predicted.shape[0]
    hits = 0
    for i in range(n_queries):
        gt_set = set(ground_truth[i, :k].tolist())
        pred_set = set(predicted[i, :k].tolist())
        hits += len(gt_set & pred_set)
    return hits / (n_queries * k)


def run_recall_benchmark(
    dataset_name: str,
    bits: int,
    base_vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """Run a single recall benchmark and return a results dict."""

    dim = base_vectors.shape[1]
    n_vectors = base_vectors.shape[0]
    n_queries = queries.shape[0]
    projections = max(1, dim // 4)

    print(f"  [{dataset_name} / {bits}-bit]  dim={dim}  n_base={n_vectors}  "
          f"n_queries={n_queries}  projections={projections}")

    # ------------------------------------------------------------------
    # Create quantizer
    # ------------------------------------------------------------------
    q = bitpolar.TurboQuantizer(dim=dim, bits=bits, projections=projections, seed=42)
    code_size = q.code_size_bytes

    # ------------------------------------------------------------------
    # Encode all base vectors
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    codes = []
    for i in range(n_vectors):
        codes.append(q.encode(base_vectors[i]))
    t_encode = time.perf_counter() - t0
    print(f"    Encode: {n_vectors} vectors in {t_encode:.2f}s "
          f"({n_vectors / t_encode:.0f} vec/s)")

    # ------------------------------------------------------------------
    # Search: for each query, score all codes and collect top-k
    # ------------------------------------------------------------------
    max_k = min(100, n_vectors)
    predicted = np.empty((n_queries, max_k), dtype=np.int32)

    t0 = time.perf_counter()
    for qi in range(n_queries):
        query = queries[qi]
        scores = np.empty(n_vectors, dtype=np.float32)
        for ci in range(n_vectors):
            scores[ci] = q.inner_product(codes[ci], query)
        # Top-k by descending score
        top_idx = np.argpartition(scores, -max_k)[-max_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        predicted[qi] = top_idx
    t_search = time.perf_counter() - t0
    print(f"    Search: {n_queries} queries in {t_search:.2f}s "
          f"({n_queries / t_search:.1f} qps)")

    # ------------------------------------------------------------------
    # Compute recall
    # ------------------------------------------------------------------
    gt_k = ground_truth.shape[1]
    r1 = recall_at_k(predicted, ground_truth, min(1, gt_k))
    r10 = recall_at_k(predicted, ground_truth, min(10, gt_k))
    r100 = recall_at_k(predicted, ground_truth, min(100, gt_k))

    original_size = dim * 4  # float32
    compression_ratio = original_size / code_size if code_size > 0 else float("inf")

    result = {
        "dataset": dataset_name,
        "bits": bits,
        "dim": dim,
        "n_vectors": n_vectors,
        "n_queries": n_queries,
        "recall_at_1": round(r1, 6),
        "recall_at_10": round(r10, 6),
        "recall_at_100": round(r100, 6),
        "compression_ratio": round(compression_ratio, 2),
        "code_size_bytes": code_size,
        "original_size_bytes": original_size,
        "time_encode_total_s": round(t_encode, 4),
        "time_search_total_s": round(t_search, 4),
    }

    print(f"    Recall@1={r1:.4f}  @10={r10:.4f}  @100={r100:.4f}  "
          f"compression={compression_ratio:.1f}x")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BitPolar Recall@k benchmarks on standard ANN datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run a specific dataset (e.g. random-384, sift-1m, glove-200).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all datasets including sift-1m and glove-200 (requires download).",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=DEFAULT_BITS,
        help="Bit widths to benchmark (default: 3 4 6 8).",
    )
    args = parser.parse_args()

    if args.dataset:
        datasets = [args.dataset]
    elif args.all:
        datasets = ALL_DATASETS
    else:
        datasets = DEFAULT_DATASETS

    bits_list = args.bits

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        try:
            base_vectors, queries, ground_truth = load_dataset(ds_name)
        except Exception as e:
            print(f"  SKIP: could not load {ds_name}: {e}")
            continue

        # Ensure float32
        base_vectors = base_vectors.astype(np.float32, copy=False)
        queries = queries.astype(np.float32, copy=False)

        for bits in bits_list:
            result = run_recall_benchmark(ds_name, bits, base_vectors, queries, ground_truth)
            all_results.append(result)

            # Save per-configuration JSON
            out_path = RESULTS_DIR / f"recall_{ds_name}_{bits}bit.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"    Saved: {out_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if all_results:
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}\n")

        headers = [
            "Dataset", "Bits", "Dim", "Recall@1", "Recall@10", "Recall@100",
            "Compress", "Code B", "Enc s", "Search s",
        ]
        rows = []
        for r in all_results:
            rows.append([
                r["dataset"],
                r["bits"],
                r["dim"],
                f"{r['recall_at_1']:.4f}",
                f"{r['recall_at_10']:.4f}",
                f"{r['recall_at_100']:.4f}",
                f"{r['compression_ratio']:.1f}x",
                r["code_size_bytes"],
                f"{r['time_encode_total_s']:.2f}",
                f"{r['time_search_total_s']:.2f}",
            ])
        print(tabulate(rows, headers=headers, tablefmt="github"))

    print("\nDone.")


if __name__ == "__main__":
    main()
