"""Core compression API for BitPolar embeddings.

This module provides the primary interface for compressing embedding matrices
using BitPolar's near-optimal vector quantization. No training or calibration
data is needed — compression is data-oblivious and deterministic.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

# Import the Rust-backed BitPolar bindings (PyO3)
try:
    import bitpolar as _bp
except ImportError:
    raise ImportError(
        "bitpolar Rust bindings not found. Install with: pip install bitpolar"
    )


def compress_embeddings(
    embeddings: np.ndarray,
    bits: int = 4,
    projections: Optional[int] = None,
    seed: int = 42,
) -> CompressedEmbeddings:
    """Compress an embedding matrix using BitPolar quantization.

    This is the primary one-line API. Takes any float32 embedding matrix
    and returns a CompressedEmbeddings object with search, decompress,
    and save/load capabilities.

    Args:
        embeddings: float32 array of shape (n, dim). Each row is one vector.
        bits: Quantization precision (3-8). Lower = more compression, less accuracy.
            - 3 bits: ~8x compression, good for cold storage
            - 4 bits: ~6x compression, good for search (recommended)
            - 6-8 bits: ~3-4x compression, near-lossless
        projections: Number of QJL projections for residual correction.
            Default: dim/4 (good balance of accuracy vs size).
        seed: Random seed for deterministic rotation/projection matrices.
            Same seed + same parameters = identical compression.

    Returns:
        CompressedEmbeddings with .search(), .decompress(), .save()/.load()

    Example:
        >>> embeddings = np.random.randn(1000, 384).astype(np.float32)
        >>> compressed = compress_embeddings(embeddings, bits=4)
        >>> ids, scores = compressed.search(embeddings[0], top_k=10)
    """
    if not (3 <= bits <= 8):
        raise ValueError(f"bits must be 3-8, got {bits}")
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings matrix cannot be empty")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got {embeddings.ndim}D")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    n, dim = embeddings.shape
    proj = projections if projections is not None else max(dim // 4, 1)

    # Create the quantizer (builds rotation + projection matrices)
    quantizer = _bp.TurboQuantizer(
        dim=dim, bits=bits, projections=proj, seed=seed
    )

    # Encode all vectors
    codes = []
    for i in range(n):
        codes.append(quantizer.encode(embeddings[i]))

    return CompressedEmbeddings(
        quantizer=quantizer,
        codes=codes,
        original_shape=(n, dim),
        bits=bits,
        projections=proj,
        seed=seed,
    )


class CompressedEmbeddings:
    """Container for a compressed embedding matrix.

    Provides search, decompression, persistence, and statistics.
    Created by compress_embeddings() or BitPolarEncoder.encode().
    """

    def __init__(
        self,
        quantizer: _bp.TurboQuantizer,
        codes: list[np.ndarray],
        original_shape: tuple[int, int],
        bits: int,
        projections: int,
        seed: int,
    ):
        self._quantizer = quantizer
        self._codes = codes
        self._shape = original_shape
        self._bits = bits
        self._projections = projections
        self._seed = seed

    @property
    def shape(self) -> tuple[int, int]:
        """Original embedding matrix shape (n_vectors, dimension)."""
        return self._shape

    @property
    def n_vectors(self) -> int:
        """Number of compressed vectors."""
        return len(self._codes)

    @property
    def dim(self) -> int:
        """Vector dimension."""
        return self._shape[1]

    @property
    def bits(self) -> int:
        """Quantization bit-width."""
        return self._bits

    @property
    def compression_ratio(self) -> float:
        """Ratio of original size to compressed size."""
        original_bytes = self._shape[0] * self._shape[1] * 4  # float32
        compressed_bytes = sum(len(c) for c in self._codes)
        if compressed_bytes == 0:
            return float("inf")
        return original_bytes / compressed_bytes

    @property
    def memory_bytes(self) -> int:
        """Total bytes used by compressed codes."""
        return sum(len(c) for c in self._codes)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search compressed embeddings by approximate inner product.

        Uses BitPolar's asymmetric distance: compressed codes vs full query.
        Much faster than decompressing all vectors and computing dot products.

        Args:
            query: float32 array of shape (dim,)
            top_k: Number of results to return

        Returns:
            Tuple of (indices, scores) where:
            - indices: int64 array of top-k vector indices
            - scores: float32 array of approximate inner product scores
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError(
                f"Query must be 1D array of length {self.dim}, "
                f"got shape {query.shape}"
            )

        # Score all vectors using compressed IP estimation
        scores = np.empty(self.n_vectors, dtype=np.float32)
        for i, code in enumerate(self._codes):
            scores[i] = self._quantizer.inner_product(code, query)

        # Return top-k by descending score
        top_k = min(top_k, self.n_vectors)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return top_indices.astype(np.int64), scores[top_indices]

    def decompress(
        self, indices: Optional[Union[np.ndarray, Sequence[int]]] = None
    ) -> np.ndarray:
        """Decompress selected or all embeddings back to float32.

        Args:
            indices: Which vectors to decompress. None = all.

        Returns:
            float32 array of shape (n_selected, dim)
        """
        if indices is None:
            indices = range(self.n_vectors)
        return np.array(
            [self._quantizer.decode(self._codes[i]) for i in indices],
            dtype=np.float32,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save compressed embeddings to a .bp file.

        File format: JSON header + binary codes.
        """
        path = Path(path)
        header = {
            "magic": "BITPOLAR",
            "version": 1,
            "n_vectors": self.n_vectors,
            "dim": self.dim,
            "bits": self._bits,
            "projections": self._projections,
            "seed": self._seed,
            "code_lengths": [len(c) for c in self._codes],
        }
        with open(path, "wb") as f:
            header_bytes = json.dumps(header).encode("utf-8")
            # Write header length (4 bytes) + header + all codes
            f.write(struct.pack("<I", len(header_bytes)))
            f.write(header_bytes)
            for code in self._codes:
                f.write(bytes(code))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CompressedEmbeddings":
        """Load compressed embeddings from a .bp file."""
        path = Path(path)
        with open(path, "rb") as f:
            header_len = struct.unpack("<I", f.read(4))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))

            if header.get("magic") != "BITPOLAR":
                raise ValueError(f"Not a BitPolar file (magic: {header.get('magic', 'missing')})")

            quantizer = _bp.TurboQuantizer(
                dim=header["dim"],
                bits=header["bits"],
                projections=header["projections"],
                seed=header["seed"],
            )

            codes = []
            for length in header["code_lengths"]:
                code_bytes = f.read(length)
                codes.append(np.frombuffer(code_bytes, dtype=np.uint8).copy())

        return cls(
            quantizer=quantizer,
            codes=codes,
            original_shape=(header["n_vectors"], header["dim"]),
            bits=header["bits"],
            projections=header["projections"],
            seed=header["seed"],
        )

    def __len__(self) -> int:
        return self.n_vectors

    def __repr__(self) -> str:
        return (
            f"CompressedEmbeddings(n={self.n_vectors}, dim={self.dim}, "
            f"bits={self._bits}, ratio={self.compression_ratio:.1f}x, "
            f"memory={self.memory_bytes:,}B)"
        )


class BitPolarEncoder:
    """Drop-in wrapper for sentence-transformers models with compression.

    Encodes text to compressed embeddings in one step. Requires
    sentence-transformers to be installed.

    Example:
        >>> encoder = BitPolarEncoder("all-MiniLM-L6-v2", bits=4)
        >>> compressed = encoder.encode(["hello world", "semantic search"])
        >>> ids, scores = compressed.search(query_vector)
    """

    def __init__(
        self,
        model_name_or_path: str,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        """Initialize encoder with a sentence-transformers model.

        Args:
            model_name_or_path: HuggingFace model name or local path
            bits: Quantization precision (3-8)
            projections: QJL projections (default: dim/4)
            seed: Deterministic seed
            device: torch device ("cpu", "cuda", etc.)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )

        kwargs = {}
        if device is not None:
            kwargs["device"] = device

        self._model = SentenceTransformer(model_name_or_path, **kwargs)
        self._bits = bits
        self._projections = projections
        self._seed = seed

    def encode(
        self,
        sentences: Union[str, list[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> CompressedEmbeddings:
        """Encode sentences to compressed embeddings.

        Args:
            sentences: Text input(s) to encode
            batch_size: Batch size for the embedding model
            show_progress_bar: Show encoding progress
            **kwargs: Additional args passed to SentenceTransformer.encode()

        Returns:
            CompressedEmbeddings with search and decompress capabilities
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = self._model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            **kwargs,
        )

        return compress_embeddings(
            embeddings,
            bits=self._bits,
            projections=self._projections,
            seed=self._seed,
        )

    @property
    def dim(self) -> int:
        """Embedding dimension of the underlying model."""
        return self._model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        return (
            f"BitPolarEncoder(model={self._model._model_config.get('name', '?')}, "
            f"bits={self._bits}, dim={self.dim})"
        )
