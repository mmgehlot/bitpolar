#!/usr/bin/env python3
"""KV cache quality benchmarks for BitPolar.

Evaluates BitPolar compression quality in the context of transformer KV caches:
  1. Attention fidelity: cosine similarity of full vs compressed attention output
  2. Inner-product preservation: correlation, MSE, and bias analysis
  3. Compression ratio table for various bit-widths

Usage:
    python benchmarks/bench_kv_cache.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    """Element-wise cosine similarity, averaged over all elements."""
    # a, b: same shape (heads, seq_len, head_dim) or (seq_len, head_dim)
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def cosine_similarity_per_head(a, b):
    """Per-head cosine similarity. a, b: (n_heads, seq_len, head_dim)."""
    n_heads = a.shape[0]
    sims = []
    for h in range(n_heads):
        sims.append(cosine_similarity(a[h], b[h]))
    return sims


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def pearson_correlation(x, y):
    """Pearson correlation between two 1-D arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx, my = x.mean(), y.mean()
    num = np.sum((x - mx) * (y - my))
    den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
    if den == 0:
        return 0.0
    return float(num / den)


# ---------------------------------------------------------------------------
# Test 1: Attention Fidelity
# ---------------------------------------------------------------------------

def test_attention_fidelity():
    """Compare exact attention vs compressed-KV attention."""
    try:
        import bitpolar
    except ImportError:
        print("[ERROR] bitpolar not installed. Install with: pip install bitpolar")
        sys.exit(1)

    print("=" * 70)
    print("Test 1: Attention Fidelity")
    print("=" * 70)

    rng = np.random.RandomState(SEED)
    n_heads = 32
    seq_len = 512
    head_dim = 128

    Q = rng.randn(n_heads, seq_len, head_dim).astype(np.float32)
    K = rng.randn(n_heads, seq_len, head_dim).astype(np.float32)
    V = rng.randn(n_heads, seq_len, head_dim).astype(np.float32)

    scale = 1.0 / np.sqrt(head_dim)

    # Exact attention output
    print("  Computing exact attention...")
    attn_weights_exact = softmax(np.einsum("hqd,hkd->hqk", Q, K) * scale, axis=-1)
    output_exact = np.einsum("hqk,hkd->hqd", attn_weights_exact, V)

    results = []
    for bits in [3, 4, 6]:
        print(f"  Compressing K, V at {bits}-bit...")
        projections = head_dim // 4

        K_approx = np.zeros_like(K)
        V_approx = np.zeros_like(V)

        for h in range(n_heads):
            # Per-head quantizer with distinct seed
            q_k = bitpolar.TurboQuantizer(
                dim=head_dim, bits=bits, projections=projections,
                seed=SEED + h,
            )
            q_v = bitpolar.TurboQuantizer(
                dim=head_dim, bits=bits, projections=projections,
                seed=SEED + n_heads + h,
            )
            for s in range(seq_len):
                code_k = q_k.encode(K[h, s])
                K_approx[h, s] = q_k.decode(code_k)
                code_v = q_v.encode(V[h, s])
                V_approx[h, s] = q_v.decode(code_v)

        # Approximate attention
        attn_weights_approx = softmax(
            np.einsum("hqd,hkd->hqk", Q, K_approx) * scale, axis=-1
        )
        output_approx = np.einsum("hqk,hkd->hqd", attn_weights_approx, V_approx)

        overall_sim = cosine_similarity(output_exact, output_approx)
        per_head_sims = cosine_similarity_per_head(output_exact, output_approx)
        mean_head_sim = float(np.mean(per_head_sims))
        min_head_sim = float(np.min(per_head_sims))

        result = {
            "bits": bits,
            "overall_cosine_similarity": round(overall_sim, 6),
            "mean_per_head_cosine_similarity": round(mean_head_sim, 6),
            "min_per_head_cosine_similarity": round(min_head_sim, 6),
        }
        results.append(result)
        print(
            f"    {bits}-bit: overall cos_sim={overall_sim:.6f}, "
            f"mean_head={mean_head_sim:.6f}, min_head={min_head_sim:.6f}"
        )

    print()
    return results


# ---------------------------------------------------------------------------
# Test 2: Inner Product Preservation (Bias test)
# ---------------------------------------------------------------------------

def test_ip_preservation():
    """Check inner-product estimation: correlation, MSE, bias."""
    try:
        import bitpolar
    except ImportError:
        print("[ERROR] bitpolar not installed. Install with: pip install bitpolar")
        sys.exit(1)

    print("=" * 70)
    print("Test 2: Inner Product Preservation (Bias Test)")
    print("=" * 70)

    rng = np.random.RandomState(SEED)
    n_pairs = 10_000
    dim = 128
    projections = dim // 4

    keys = rng.randn(n_pairs, dim).astype(np.float32)
    queries_arr = rng.randn(n_pairs, dim).astype(np.float32)

    results = []
    for bits in [3, 4, 6]:
        quantizer = bitpolar.TurboQuantizer(
            dim=dim, bits=bits, projections=projections, seed=SEED
        )

        exact_ips = np.empty(n_pairs, dtype=np.float64)
        approx_ips = np.empty(n_pairs, dtype=np.float64)

        for i in range(n_pairs):
            exact_ips[i] = float(np.dot(keys[i], queries_arr[i]))
            code = quantizer.encode(keys[i])
            approx_ips[i] = quantizer.inner_product(code, queries_arr[i])

        corr = pearson_correlation(exact_ips, approx_ips)
        mse = float(np.mean((exact_ips - approx_ips) ** 2))
        bias = float(np.mean(approx_ips - exact_ips))

        result = {
            "bits": bits,
            "n_pairs": n_pairs,
            "pearson_correlation": round(corr, 6),
            "mse": round(mse, 6),
            "mean_bias": round(bias, 6),
        }
        results.append(result)
        print(
            f"  {bits}-bit: pearson_r={corr:.6f}, mse={mse:.6f}, bias={bias:.6f}"
        )

    print()
    return results


# ---------------------------------------------------------------------------
# Test 3: Compression Ratio Table
# ---------------------------------------------------------------------------

def test_compression_ratios():
    """Compute compression ratios for various bit-widths at head_dim=128."""
    try:
        import bitpolar
    except ImportError:
        print("[ERROR] bitpolar not installed. Install with: pip install bitpolar")
        sys.exit(1)

    print("=" * 70)
    print("Test 3: Compression Ratio (head_dim=128)")
    print("=" * 70)

    head_dim = 128
    rng = np.random.RandomState(SEED)
    sample_vec = rng.randn(head_dim).astype(np.float32)

    original_bytes_per_kv = head_dim * 4 * 2  # K + V, float32

    results = []
    header = f"{'Bits':>6} {'Orig B/KV':>12} {'Comp B/KV':>12} {'Ratio':>8} {'Savings':>10}"
    print(header)
    print("-" * len(header))

    for bits in [3, 4, 6, 8]:
        projections = head_dim // 4
        quantizer = bitpolar.TurboQuantizer(
            dim=head_dim, bits=bits, projections=projections, seed=SEED
        )
        code = quantizer.encode(sample_vec)
        compressed_bytes_per_vec = len(code)
        compressed_bytes_per_kv = compressed_bytes_per_vec * 2  # K + V

        ratio = original_bytes_per_kv / compressed_bytes_per_kv
        savings = (1.0 - compressed_bytes_per_kv / original_bytes_per_kv) * 100.0

        result = {
            "bits": bits,
            "original_bytes_per_kv": original_bytes_per_kv,
            "compressed_bytes_per_kv": compressed_bytes_per_kv,
            "compression_ratio": round(ratio, 2),
            "savings_pct": round(savings, 2),
        }
        results.append(result)
        print(
            f"{bits:>6} "
            f"{original_bytes_per_kv:>12} "
            f"{compressed_bytes_per_kv:>12} "
            f"{ratio:>8.2f}x "
            f"{savings:>9.1f}%"
        )

    print()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    attn_results = test_attention_fidelity()
    ip_results = test_ip_preservation()
    compression_results = test_compression_ratios()

    output = {
        "config": {
            "seed": SEED,
            "attention_fidelity": {
                "n_heads": 32,
                "seq_len": 512,
                "head_dim": 128,
            },
            "ip_preservation": {
                "n_pairs": 10_000,
                "dim": 128,
            },
        },
        "attention_fidelity": attn_results,
        "ip_preservation": ip_results,
        "compression_ratios": compression_results,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kv_cache.json"
    with open(out_path, "w") as f:
        json.dump(output, indent=2, fp=f)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
