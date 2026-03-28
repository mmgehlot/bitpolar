"""scikit-learn integration for BitPolar vector quantization.

Provides sklearn-compatible transformers for compressing feature matrices
and performing compressed nearest-neighbor search within sklearn pipelines.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    raise ImportError(
        "scikit-learn is required. Install with: pip install scikit-learn"
    )

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar is required. Install with: pip install bitpolar")


def _validate_bits(bits: int) -> None:
    """Validate quantization bit-width is in the supported range."""
    if not (3 <= bits <= 8):
        raise ValueError(f"bits must be 3-8, got {bits}")


class BitPolarTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer for BitPolar vector compression.

    Compresses 2D float feature matrices into compact uint8 code arrays
    and supports inverse transformation back to approximate float32.

    Follows the sklearn estimator API: fit() learns the input dimension,
    transform() compresses, inverse_transform() decompresses.

    Parameters:
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed for deterministic compression

    Example:
        >>> from bitpolar_sklearn import BitPolarTransformer
        >>> import numpy as np
        >>> X = np.random.randn(100, 384).astype(np.float32)
        >>> t = BitPolarTransformer(bits=4)
        >>> codes = t.fit_transform(X)
        >>> codes.shape
        (100,)
        >>> restored = t.inverse_transform(codes)
        >>> restored.shape
        (100, 384)
    """

    def __init__(
        self,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
    ):
        _validate_bits(bits)
        self.bits = bits
        self.projections = projections
        self.seed = seed
        self._quantizer: Optional["_bp.TurboQuantizer"] = None
        self._dim: Optional[int] = None

    def _ensure_quantizer(self, dim: int) -> None:
        """Lazily initialize the quantizer on first use."""
        if self._quantizer is None or self._dim != dim:
            self._dim = dim
            proj = self.projections or max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self.bits, projections=proj, seed=self.seed
            )

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "BitPolarTransformer":
        """Learn the input dimension and initialize the quantizer.

        Args:
            X: float array of shape (n_samples, n_features)
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty dataset")
        self._ensure_quantizer(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compress feature vectors to BitPolar codes.

        Args:
            X: float array of shape (n_samples, n_features)

        Returns:
            numpy object array of shape (n_samples,) where each
            element is a uint8 code array
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        if X.shape[0] == 0:
            return np.empty(0, dtype=object)

        if self._quantizer is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        if X.shape[1] != self._dim:
            raise ValueError(
                f"Expected {self._dim} features, got {X.shape[1]}"
            )

        n = X.shape[0]
        codes = np.empty(n, dtype=object)
        for i in range(n):
            codes[i] = self._quantizer.encode(np.ascontiguousarray(X[i]))

        return codes

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Decompress BitPolar codes back to approximate float32 vectors.

        Args:
            X: numpy object array of uint8 codes from :meth:`transform`

        Returns:
            float32 array of shape (n_samples, dim)
        """
        if self._quantizer is None or self._dim is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        X = np.asarray(X)
        n = len(X)
        result = np.empty((n, self._dim), dtype=np.float32)
        for i in range(n):
            result[i] = self._quantizer.decode(X[i])

        return result

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "bits": self.bits,
            "projections": self.projections,
            "seed": self.seed,
        }

    def set_params(self, **params: object) -> "BitPolarTransformer":
        """Set parameters, re-initializing quantizer if needed."""
        for key, value in params.items():
            setattr(self, key, value)
        # Reset quantizer so it re-initializes on next fit
        self._quantizer = None
        self._dim = None
        return self


class BitPolarSearchTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer for compressed nearest-neighbor search.

    Stores a compressed index of reference vectors and scores new queries
    against them using BitPolar's asymmetric inner product. Designed for
    use in sklearn pipelines where the final step is a similarity search.

    Parameters:
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed
        top_k: Number of nearest neighbors to return (default 10)

    Example:
        >>> from bitpolar_sklearn import BitPolarSearchTransformer
        >>> import numpy as np
        >>> X_ref = np.random.randn(1000, 384).astype(np.float32)
        >>> searcher = BitPolarSearchTransformer(bits=4, top_k=5)
        >>> searcher.fit(X_ref)
        >>> query = np.random.randn(3, 384).astype(np.float32)
        >>> results = searcher.transform(query)  # shape: (3, 5)
    """

    def __init__(
        self,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
        top_k: int = 10,
    ):
        _validate_bits(bits)
        self.bits = bits
        self.projections = projections
        self.seed = seed
        self.top_k = top_k
        self._quantizer: Optional["_bp.TurboQuantizer"] = None
        self._dim: Optional[int] = None
        self._codes: list[np.ndarray] = []
        self._n_samples: int = 0

    def _ensure_quantizer(self, dim: int) -> None:
        """Lazily initialize the quantizer."""
        if self._quantizer is None or self._dim != dim:
            self._dim = dim
            proj = self.projections or max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self.bits, projections=proj, seed=self.seed
            )

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "BitPolarSearchTransformer":
        """Build a compressed index from reference vectors.

        Args:
            X: float array of shape (n_samples, n_features)
            y: Ignored. Present for API compatibility.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        n, dim = X.shape
        self._ensure_quantizer(dim)
        self._n_samples = n

        self._codes = []
        for i in range(n):
            self._codes.append(
                self._quantizer.encode(np.ascontiguousarray(X[i]))
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Search the compressed index for nearest neighbors.

        Args:
            X: float query array of shape (n_queries, n_features)

        Returns:
            int array of shape (n_queries, top_k) containing indices
            of the nearest neighbors in the fitted reference set
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got {X.ndim}D")

        if self._quantizer is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        if X.shape[1] != self._dim:
            raise ValueError(
                f"Expected {self._dim} features, got {X.shape[1]}"
            )

        nq = X.shape[0]
        k = min(self.top_k, len(self._codes))
        indices = np.empty((nq, k), dtype=np.int64)

        for qi in range(nq):
            scores = np.empty(len(self._codes), dtype=np.float32)
            for j, code in enumerate(self._codes):
                scores[j] = self._quantizer.inner_product(
                    code, np.ascontiguousarray(X[qi])
                )

            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            indices[qi] = top_idx

        return indices

    def search(
        self,
        query: np.ndarray,
        top_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search with scores returned alongside indices.

        Args:
            query: float array of shape (dim,) or (n_queries, dim)
            top_k: Override the default top_k for this search

        Returns:
            Tuple of (indices, scores), each of shape (n_queries, k)
        """
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if self._quantizer is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        k = min(top_k or self.top_k, len(self._codes))
        nq = query.shape[0]
        indices = np.empty((nq, k), dtype=np.int64)
        scores_out = np.empty((nq, k), dtype=np.float32)

        for qi in range(nq):
            scores = np.empty(len(self._codes), dtype=np.float32)
            for j, code in enumerate(self._codes):
                scores[j] = self._quantizer.inner_product(
                    code, np.ascontiguousarray(query[qi])
                )

            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            indices[qi] = top_idx
            scores_out[qi] = scores[top_idx]

        return indices, scores_out

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            "bits": self.bits,
            "projections": self.projections,
            "seed": self.seed,
            "top_k": self.top_k,
        }

    def set_params(self, **params: object) -> "BitPolarSearchTransformer":
        """Set parameters, clearing the index."""
        for key, value in params.items():
            setattr(self, key, value)
        self._quantizer = None
        self._dim = None
        self._codes = []
        self._n_samples = 0
        return self
