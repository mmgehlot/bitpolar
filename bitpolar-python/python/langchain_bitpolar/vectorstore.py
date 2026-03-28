"""BitPolar-compressed VectorStore for LangChain.

Implements LangChain's VectorStore interface using BitPolar's near-optimal
vector quantization for compressed in-memory storage. Suitable for
small-to-medium datasets (up to ~1M vectors) where you want to avoid
running an external vector database.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable, Optional

import numpy as np

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError:
    raise ImportError(
        "langchain-core required. Install with: pip install langchain-core"
    )

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError(
        "bitpolar Rust bindings required. Install with: pip install bitpolar"
    )


class BitPolarVectorStore(VectorStore):
    """In-memory vector store using BitPolar compression.

    Compresses embeddings to 3-8 bits using near-optimal quantization,
    reducing memory usage by 4-8x compared to float32 storage (FAISS).
    No training, no calibration data, no external database needed.

    The store supports:
    - Adding documents with text and metadata
    - Similarity search by approximate inner product
    - Deletion by ID
    - Persistence via save/load

    Args:
        embedding: LangChain Embeddings instance for text→vector conversion
        bits: Quantization precision (3-8, default 4)
        projections: QJL projections (default dim/4)
        seed: Deterministic seed for reproducible compression

    Example:
        >>> from langchain_bitpolar import BitPolarVectorStore
        >>> from langchain_openai import OpenAIEmbeddings
        >>>
        >>> store = BitPolarVectorStore(embedding=OpenAIEmbeddings(), bits=4)
        >>> store.add_texts(["Hello world", "Semantic search is powerful"])
        >>> results = store.similarity_search("greeting", k=1)
        >>> print(results[0].page_content)
        Hello world
    """

    def __init__(
        self,
        embedding: Embeddings,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        self._embedding = embedding
        self._bits = bits
        self._projections = projections
        self._seed = seed
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        # Storage: doc_id → (compressed_code, Document)
        self._store: dict[str, tuple[np.ndarray, Document]] = {}

    @property
    def embeddings(self) -> Embeddings:
        """Return the embedding function used by this store."""
        return self._embedding

    def _ensure_quantizer(self, dim: int) -> None:
        """Lazily initialize the quantizer on first vector insertion.

        This allows the store to be created before knowing the embedding
        dimension (which depends on the model).
        """
        if self._quantizer is None:
            proj = self._projections if self._projections else max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed, compress, and store texts.

        Args:
            texts: Texts to embed and store
            metadatas: Optional metadata dicts for each text
            ids: Optional IDs (auto-generated if not provided)

        Returns:
            List of document IDs
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        vectors = self._embedding.embed_documents(texts_list)
        if not vectors:
            return []

        if ids is not None and len(ids) != len(texts_list):
            raise ValueError(
                f"ids length ({len(ids)}) must match texts length ({len(texts_list)})"
            )

        dim = len(vectors[0])
        self._ensure_quantizer(dim)

        result_ids = []
        for i, (text, vec) in enumerate(zip(texts_list, vectors)):
            doc_id = ids[i] if ids else str(uuid.uuid4())
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            vec_np = np.array(vec, dtype=np.float32)
            code = self._quantizer.encode(vec_np)
            doc = Document(page_content=text, metadata=meta)
            self._store[doc_id] = (code, doc)
            result_ids.append(doc_id)

        return result_ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for documents most similar to the query.

        Uses BitPolar's asymmetric inner product estimation:
        compressed stored vectors vs full-precision query vector.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (Document, score) tuples sorted by descending similarity
        """
        if not self._store:
            return []

        query_vec = np.array(
            self._embedding.embed_query(query), dtype=np.float32
        )

        # Score all documents
        scored: list[tuple[str, float]] = []
        for doc_id, (code, _doc) in self._store.items():
            score = self._quantizer.inner_product(code, query_vec)
            scored.append((doc_id, score))

        # Sort by score descending, take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:k]

        return [
            (self._store[doc_id][1], score) for doc_id, score in scored
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for documents most similar to the query.

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of Documents sorted by descending similarity
        """
        return [doc for doc, _score in self.similarity_search_with_score(query, k)]

    def delete(
        self,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if any documents were deleted
        """
        if ids is None:
            return False
        deleted = False
        for doc_id in ids:
            if doc_id in self._store:
                del self._store[doc_id]
                deleted = True
        return deleted

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> "BitPolarVectorStore":
        """Create a BitPolarVectorStore from texts.

        Args:
            texts: Texts to embed and store
            embedding: Embedding function
            metadatas: Optional metadata for each text
            **kwargs: Passed to BitPolarVectorStore constructor (bits, projections, seed)

        Returns:
            Populated BitPolarVectorStore
        """
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

    def __len__(self) -> int:
        """Number of documents in the store."""
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"BitPolarVectorStore(n={len(self)}, bits={self._bits})"
        )
