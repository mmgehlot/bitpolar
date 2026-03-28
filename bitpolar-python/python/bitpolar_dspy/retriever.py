"""DSPy retriever module backed by BitPolar compressed embeddings.

Provides a BitPolarRM class that extends dspy.Retrieve for integration
into DSPy pipelines, compressing corpus embeddings at init time and
performing approximate inner product search at query time.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")

try:
    import dspy

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

    # Minimal fallback so the class can still be instantiated without dspy
    class _FakeRetrieve:
        """Fallback base when dspy is not installed."""

        def __init__(self, k: int = 5):
            self.k = k

    class _FakePrediction:
        """Fallback Prediction when dspy is not installed."""

        def __init__(self, **kwargs: Any):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class dspy:  # type: ignore[no-redef]
        Retrieve = _FakeRetrieve
        Prediction = _FakePrediction


class BitPolarRM(dspy.Retrieve):
    """DSPy retriever module using BitPolar compressed embeddings.

    Compresses all corpus embeddings at construction time and performs
    approximate nearest-neighbor search via compressed inner product.
    Supports both pre-computed query embeddings and string queries
    (when an embedding function is provided).

    Args:
        texts: List of passage strings corresponding to each embedding.
        embeddings: 2D array-like of shape (n, dim) with corpus vectors.
            All embeddings are compressed on init for efficient search.
        bits: Quantization precision (3-8, default 4).
        k: Default number of passages to retrieve.
        seed: Random seed for deterministic compression.
        embed_fn: Optional callable that maps a list of strings to a list
            of float vectors. Required if ``forward`` is called with string
            queries instead of pre-computed embeddings.

    Example:
        >>> embeddings = [[0.1]*128 for _ in range(100)]
        >>> texts = [f"passage {i}" for i in range(100)]
        >>> rm = BitPolarRM(texts=texts, embeddings=embeddings, bits=4, k=5)
        >>> pred = rm.forward([0.1]*128, k=3)
        >>> len(pred.passages)
        3
    """

    def __init__(
        self,
        texts: List[str],
        embeddings: Any,
        bits: int = 4,
        k: int = 5,
        seed: int = 42,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        super().__init__(k=k)
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        self._bits = bits
        self._seed = seed
        self._texts = list(texts)
        self._embed_fn = embed_fn

        # Convert to numpy and compress all corpus embeddings
        corpus = np.array(embeddings, dtype=np.float32)
        if corpus.ndim == 1:
            corpus = corpus.reshape(1, -1)
        if corpus.size == 0:
            raise ValueError("embeddings corpus cannot be empty")

        if corpus.shape[0] != len(self._texts):
            raise ValueError(
                f"embeddings ({corpus.shape[0]}) and texts ({len(self._texts)}) "
                f"must have same length"
            )

        self._dim = corpus.shape[1]
        proj = max(self._dim // 4, 1)
        self._quantizer = _bp.TurboQuantizer(
            dim=self._dim, bits=self._bits, projections=proj, seed=self._seed
        )

        # Pre-compress all corpus vectors
        self._codes: List[np.ndarray] = []
        self._corpus_vectors = corpus
        for i in range(corpus.shape[0]):
            self._codes.append(self._quantizer.encode(corpus[i]))

    def _embed_queries(self, queries: List[str]) -> np.ndarray:
        """Embed string queries using the configured embedding function.

        Args:
            queries: List of query strings.

        Returns:
            float32 array of shape (n_queries, dim).

        Raises:
            ValueError: If no embedding function was provided.
        """
        if self._embed_fn is None:
            raise ValueError(
                "String queries require an embed_fn. Pass embed_fn= at construction "
                "time, or provide pre-computed embedding vectors to forward()."
            )
        embeddings = self._embed_fn(queries)
        return np.array(embeddings, dtype=np.float32)

    def _search_single(self, query_vec: np.ndarray, k: int) -> List[str]:
        """Search for the top-k passages given a single query vector.

        Args:
            query_vec: float32 array of shape (dim,).
            k: Number of passages to return.

        Returns:
            List of passage strings, ordered by descending similarity.
        """
        n = len(self._codes)
        scores = np.empty(n, dtype=np.float32)
        for i, code in enumerate(self._codes):
            scores[i] = self._quantizer.inner_product(code, query_vec)

        k = min(k, n)
        if k < n:
            top_idx = np.argpartition(scores, -k)[-k:]
        else:
            top_idx = np.arange(n)
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [self._texts[i] for i in top_idx[:k]]

    def forward(
        self,
        query_or_queries: Union[str, List[str], List[float], np.ndarray],
        k: Optional[int] = None,
    ) -> Any:
        """Retrieve the top-k passages for the given query or queries.

        Accepts multiple input forms:
        - A single string query (requires embed_fn).
        - A list of string queries (requires embed_fn).
        - A single embedding vector (list of floats or 1D ndarray).
        - A 2D ndarray of query embeddings.

        Args:
            query_or_queries: Query input in one of the supported forms.
            k: Number of passages to retrieve. Defaults to the ``k``
                set at init time.

        Returns:
            ``dspy.Prediction`` with a ``passages`` attribute containing
            the retrieved text passages. For a single query this is a flat
            list of strings; for multiple queries it is a list of lists.

        Example:
            >>> pred = rm.forward([0.1]*128, k=3)
            >>> pred.passages
            ['passage 42', 'passage 7', 'passage 99']
        """
        effective_k = k if k is not None else self.k

        # Determine query type and convert to numpy array(s)
        if isinstance(query_or_queries, str):
            # Single string query
            query_vecs = self._embed_queries([query_or_queries])
            passages = self._search_single(query_vecs[0], effective_k)
            return dspy.Prediction(passages=passages)

        if isinstance(query_or_queries, np.ndarray):
            query_arr = query_or_queries.astype(np.float32)
            if query_arr.ndim == 1:
                passages = self._search_single(query_arr, effective_k)
                return dspy.Prediction(passages=passages)
            # Multiple query vectors
            all_passages = []
            for i in range(query_arr.shape[0]):
                all_passages.append(self._search_single(query_arr[i], effective_k))
            return dspy.Prediction(passages=all_passages)

        if isinstance(query_or_queries, list):
            if len(query_or_queries) == 0:
                return dspy.Prediction(passages=[])

            # Check if it's a list of strings or a list of floats
            if isinstance(query_or_queries[0], str):
                # List of string queries
                query_vecs = self._embed_queries(query_or_queries)
                if query_vecs.shape[0] == 1:
                    passages = self._search_single(query_vecs[0], effective_k)
                    return dspy.Prediction(passages=passages)
                all_passages = []
                for i in range(query_vecs.shape[0]):
                    all_passages.append(
                        self._search_single(query_vecs[i], effective_k)
                    )
                return dspy.Prediction(passages=all_passages)

            elif isinstance(query_or_queries[0], (int, float)):
                # Single embedding as list of floats
                query_vec = np.array(query_or_queries, dtype=np.float32)
                passages = self._search_single(query_vec, effective_k)
                return dspy.Prediction(passages=passages)

            elif isinstance(query_or_queries[0], (list, np.ndarray)):
                # Multiple embeddings as list of lists
                all_passages = []
                for q in query_or_queries:
                    query_vec = np.array(q, dtype=np.float32)
                    all_passages.append(
                        self._search_single(query_vec, effective_k)
                    )
                return dspy.Prediction(passages=all_passages)

        raise TypeError(
            f"Unsupported query type: {type(query_or_queries)}. "
            "Expected str, List[str], List[float], or np.ndarray."
        )
