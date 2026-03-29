#!/usr/bin/env python3
"""Memory and compression analysis for BitPolar.

Measures compression ratio and bits-per-dimension across a matrix of
vector dimensions and quantization bit-widths. Encodes 1000 random
vectors per configuration and reports average compressed size.

Usage:
    python benchmarks/bench_compression.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SEED = 42
N_VECTORS = 1_000
DIMS = [96, 128, 200, 384, 768, 1536]
BITS_LIST = [3, 4, 6, 8]


def main():
    try:
        import bitpolar
    except ImportError:
        print("[ERROR] bitpolar not installed. Install with: pip install bitpolar")
        sys.exit(1)

    # Optional: use tabulate for pretty output
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False

    rng = np.random.RandomState(SEED)

    results = []

    for dim in DIMS:
        vectors = rng.randn(N_VECTORS, dim).astype(np.float32)
        # L2 normalise
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms

        for bits in BITS_LIST:
            projections = max(dim // 4, 1)
            quantizer = bitpolar.TurboQuantizer(
                dim=dim, bits=bits, projections=projections, seed=SEED
            )

            total_compressed = 0
            for i in range(N_VECTORS):
                code = quantizer.encode(vectors[i])
                total_compressed += len(code)

            original_bytes = dim * 4  # float32
            compressed_bytes = total_compressed / N_VECTORS
            compression_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")
            bits_per_dim = (compressed_bytes * 8) / dim

            results.append({
                "dim": dim,
                "bits": bits,
                "original_bytes": original_bytes,
                "compressed_bytes": round(compressed_bytes, 1),
                "compression_ratio": round(compression_ratio, 2),
                "bits_per_dimension": round(bits_per_dim, 3),
            })

    # Print table
    print("=" * 80)
    print("BitPolar Compression Analysis")
    print(f"N={N_VECTORS} vectors per config, L2-normalised, seed={SEED}")
    print("=" * 80)

    if has_tabulate:
        table_data = []
        for r in results:
            table_data.append([
                r["dim"],
                r["bits"],
                r["original_bytes"],
                r["compressed_bytes"],
                f"{r['compression_ratio']:.2f}x",
                f"{r['bits_per_dimension']:.3f}",
            ])
        headers = ["Dim", "Bits", "Orig (B)", "Comp (B)", "Ratio", "Bits/Dim"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        header = (
            f"{'Dim':>6} {'Bits':>6} {'Orig(B)':>10} {'Comp(B)':>10} "
            f"{'Ratio':>10} {'Bits/Dim':>10}"
        )
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['dim']:>6} "
                f"{r['bits']:>6} "
                f"{r['original_bytes']:>10} "
                f"{r['compressed_bytes']:>10.1f} "
                f"{r['compression_ratio']:>9.2f}x "
                f"{r['bits_per_dimension']:>10.3f}"
            )

    print()

    # Save JSON
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compression.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "n_vectors": N_VECTORS,
                    "dims": DIMS,
                    "bits_list": BITS_LIST,
                    "seed": SEED,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
