#!/usr/bin/env python3
"""Download and cache standard ANN benchmark datasets for BitPolar benchmarking.

Supported datasets:
    - SIFT-1M: 1M 128-dim image descriptors (IRISA)
    - GloVe-1.2M: 1.2M 200-dim word embeddings (ann-benchmarks HDF5)
    - Random synthetic datasets at various dimensions

Datasets are cached to ~/.cache/bitpolar-benchmarks/ to avoid re-downloading.

Usage:
    python benchmarks/download_datasets.py              # Download all
    python benchmarks/download_datasets.py sift         # Download SIFT only
    python benchmarks/download_datasets.py --list       # List available datasets
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

CACHE_DIR = Path.home() / ".cache" / "bitpolar-benchmarks"

# Dataset registry: name -> (url, sha256, description)
DATASETS = {
    "sift-1m": {
        "base_url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "ann_bench_url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "description": "SIFT-1M: 1M 128-dim image descriptors",
        "dim": 128,
        "n_base": 1_000_000,
        "n_query": 10_000,
    },
    "glove-200": {
        "ann_bench_url": "http://ann-benchmarks.com/glove-200-angular.hdf5",
        "description": "GloVe-1.2M: 1.2M 200-dim word embeddings",
        "dim": 200,
        "n_base": 1_183_514,
        "n_query": 10_000,
    },
}


def ensure_cache_dir() -> Path:
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress bar and resume support."""
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if partially downloaded
    existing_size = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        total = int(resp.headers.get("content-length", 0)) + existing_size

        mode = "ab" if existing_size > 0 else "wb"
        with open(dest, mode) as f, tqdm(
            total=total,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            desc=desc or dest.name,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Try manual download: {url}")
        raise


def read_fvecs(path: Path) -> np.ndarray:
    """Read vectors from .fvecs format (IRISA format).

    Format: for each vector, 4-byte int (dim) followed by dim * 4-byte floats.
    """
    with open(path, "rb") as f:
        data = f.read()

    # Read first dimension
    dim = struct.unpack("<i", data[:4])[0]
    record_size = 4 + dim * 4  # 4 bytes for dim + dim floats

    n = len(data) // record_size
    vectors = np.empty((n, dim), dtype=np.float32)

    for i in range(n):
        offset = i * record_size + 4  # skip the dim int
        vectors[i] = np.frombuffer(data[offset : offset + dim * 4], dtype=np.float32)

    return vectors


def read_ivecs(path: Path) -> np.ndarray:
    """Read integer vectors from .ivecs format (ground truth indices)."""
    with open(path, "rb") as f:
        data = f.read()

    dim = struct.unpack("<i", data[:4])[0]
    record_size = 4 + dim * 4

    n = len(data) // record_size
    vectors = np.empty((n, dim), dtype=np.int32)

    for i in range(n):
        offset = i * record_size + 4
        vectors[i] = np.frombuffer(data[offset : offset + dim * 4], dtype=np.int32)

    return vectors


def load_hdf5_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from ann-benchmarks HDF5 format.

    Returns:
        Tuple of (base_vectors, query_vectors, ground_truth_neighbors)
    """
    import h5py

    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
    return train, test, neighbors


def generate_random_dataset(
    n_base: int, n_query: int, dim: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic random dataset with ground truth.

    Vectors are drawn from standard normal distribution and L2-normalized
    to simulate unit-norm embeddings (common in production).

    Returns:
        (base_vectors, query_vectors, ground_truth_top100)
    """
    rng = np.random.RandomState(seed)

    base = rng.randn(n_base, dim).astype(np.float32)
    # L2 normalize to unit vectors (simulates embedding models)
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    base /= norms

    queries = rng.randn(n_query, dim).astype(np.float32)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    queries /= q_norms

    # Compute ground truth: exact inner product, top-100
    print(f"  Computing ground truth for {n_query} queries over {n_base} vectors...")
    k = min(100, n_base)
    neighbors = np.empty((n_query, k), dtype=np.int32)

    batch_size = 100
    for i in range(0, n_query, batch_size):
        end = min(i + batch_size, n_query)
        # Exact inner product: queries @ base.T
        scores = queries[i:end] @ base.T  # (batch, n_base)
        # Top-k indices per query
        for j in range(end - i):
            top_idx = np.argpartition(scores[j], -k)[-k:]
            top_idx = top_idx[np.argsort(scores[j][top_idx])[::-1]]
            neighbors[i + j] = top_idx

    return base, queries, neighbors


def download_sift1m() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download and load SIFT-1M dataset.

    Returns (base, queries, ground_truth) as numpy arrays.
    """
    cache = ensure_cache_dir()
    hdf5_path = cache / "sift-128-euclidean.hdf5"

    if not hdf5_path.exists():
        print("Downloading SIFT-1M from ann-benchmarks...")
        url = DATASETS["sift-1m"]["ann_bench_url"]
        download_file(url, hdf5_path, "SIFT-1M")

    print("Loading SIFT-1M...")
    return load_hdf5_dataset(hdf5_path)


def download_glove200() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download and load GloVe-200 dataset.

    Returns (base, queries, ground_truth) as numpy arrays.
    """
    cache = ensure_cache_dir()
    hdf5_path = cache / "glove-200-angular.hdf5"

    if not hdf5_path.exists():
        print("Downloading GloVe-200 from ann-benchmarks...")
        url = DATASETS["glove-200"]["ann_bench_url"]
        download_file(url, hdf5_path, "GloVe-200")

    print("Loading GloVe-200...")
    return load_hdf5_dataset(hdf5_path)


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset by name.

    Args:
        name: One of 'sift-1m', 'glove-200', 'random-384', 'random-768', 'random-1536'

    Returns:
        (base_vectors, query_vectors, ground_truth_neighbors)
    """
    if name == "sift-1m":
        return download_sift1m()
    elif name == "glove-200":
        return download_glove200()
    elif name.startswith("random-"):
        dim = int(name.split("-")[1])
        # Adjust size by dimension to keep benchmarks reasonable
        if dim <= 256:
            n_base, n_query = 100_000, 1_000
        elif dim <= 768:
            n_base, n_query = 50_000, 500
        else:
            n_base, n_query = 10_000, 100
        print(f"Generating random-{dim} dataset ({n_base} base, {n_query} queries)...")
        return generate_random_dataset(n_base, n_query, dim)
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys()) + ['random-384', 'random-768', 'random-1536']}")


def get_system_info() -> dict:
    """Collect system information for reproducibility."""
    import platform

    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_gb = round(mem.total / (1024 ** 3), 1)
    except ImportError:
        mem_gb = "unknown"

    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_gb": mem_gb,
    }


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name}: {info['description']} (dim={info['dim']}, n={info['n_base']})")
        print("  random-384: Synthetic 384-dim (100K vectors)")
        print("  random-768: Synthetic 768-dim (50K vectors)")
        print("  random-1536: Synthetic 1536-dim (10K vectors)")
        sys.exit(0)

    targets = sys.argv[1:] if len(sys.argv) > 1 else ["sift-1m", "glove-200"]

    for name in targets:
        print(f"\n=== {name} ===")
        try:
            base, queries, gt = load_dataset(name)
            print(f"  Base: {base.shape}, Queries: {queries.shape}, GT: {gt.shape}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\nSystem info:", get_system_info())
