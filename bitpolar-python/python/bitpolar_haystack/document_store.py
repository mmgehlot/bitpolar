"""Haystack 2.x DocumentStore backed by BitPolar compressed embeddings.

Implements the Haystack document store protocol with compressed embedding
storage for memory-efficient retrieval pipelines.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")

try:
    from haystack import Document, default_from_dict, default_to_dict
    from haystack.document_stores.types import DuplicatePolicy

    _HAS_HAYSTACK = True
except ImportError:
    _HAS_HAYSTACK = False

    class DuplicatePolicy:  # type: ignore[no-redef]
        """Fallback DuplicatePolicy when haystack is not installed."""

        OVERWRITE = "overwrite"
        SKIP = "skip"
        FAIL = "fail"

    class Document:  # type: ignore[no-redef]
        """Minimal Document fallback when haystack is not installed."""

        def __init__(
            self,
            id: Optional[str] = None,
            content: Optional[str] = None,
            embedding: Optional[List[float]] = None,
            meta: Optional[Dict[str, Any]] = None,
            score: Optional[float] = None,
            **kwargs: Any,
        ):
            self.id = id or str(uuid.uuid4())
            self.content = content
            self.embedding = embedding
            self.meta = meta or {}
            self.score = score


class BitPolarDocumentStore:
    """Haystack 2.x-compatible document store with BitPolar compression.

    Stores documents with their embeddings compressed via BitPolar quantization,
    providing 4-8x memory reduction for embedding storage. Supports the full
    Haystack 2.x DocumentStore protocol including filtering, duplicate policies,
    and serialization.

    Internally stores each document and its compressed code together in
    ``_docs``, a dict mapping document IDs to ``(Document, code)`` tuples.

    Args:
        bits: Quantization precision (3-8, default 4).
        seed: Random seed for deterministic compression.

    Example:
        >>> store = BitPolarDocumentStore(bits=4)
        >>> doc = Document(content="Hello world", embedding=[0.1]*384)
        >>> store.write_documents([doc])
        1
        >>> store.count_documents()
        1
    """

    def __init__(self, bits: int = 4, seed: int = 42):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._bits = bits
        self._seed = seed
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._dim: Optional[int] = None

        # Internal storage: doc_id -> (Document, compressed_code)
        self._docs: Dict[str, Tuple[Document, Optional[np.ndarray]]] = {}

    def _ensure_quantizer(self, dim: int) -> None:
        """Lazily create the quantizer for the given dimension.

        Args:
            dim: Vector dimensionality.

        Raises:
            ValueError: If dimension doesn't match an existing quantizer.
        """
        if self._quantizer is None:
            self._dim = dim
            proj = max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )
        elif dim != self._dim:
            raise ValueError(
                f"Dimension mismatch: quantizer expects {self._dim}, got {dim}"
            )

    def count_documents(self) -> int:
        """Return the number of documents stored.

        Returns:
            Integer count of documents in the store.
        """
        return len(self._docs)

    def filter_documents(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve documents matching the given metadata filters.

        Supports simple equality filters on document metadata fields.
        If no filters are provided, returns all documents.

        Args:
            filters: Optional dict of metadata field names to required values.
                Supports flat equality matching: ``{"field": "value"}``.
                Nested Haystack filter syntax with ``"operator"`` and
                ``"conditions"`` keys is also supported for ``AND``/``OR``.

        Returns:
            List of matching Document objects.
        """
        docs = [doc for doc, _code in self._docs.values()]
        if filters is None:
            return docs

        return [doc for doc in docs if self._matches_filters(doc, filters)]

    def _matches_filters(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """Check if a document matches the given filters.

        Supports both flat equality dicts and Haystack 2.x filter syntax
        with nested operator/conditions structures.

        Args:
            doc: Document to test against filters.
            filters: Filter dict, either flat equality or nested.

        Returns:
            True if the document matches all filter conditions.
        """
        if "operator" in filters and "conditions" in filters:
            operator = filters["operator"].upper()
            conditions = filters["conditions"]
            if operator == "AND":
                return all(
                    self._matches_single_condition(doc, cond) for cond in conditions
                )
            elif operator == "OR":
                return any(
                    self._matches_single_condition(doc, cond) for cond in conditions
                )
            return False

        # Flat equality filter
        meta = doc.meta if hasattr(doc, "meta") and doc.meta else {}
        for key, value in filters.items():
            if meta.get(key) != value:
                return False
        return True

    def _matches_single_condition(
        self, doc: Document, condition: Dict[str, Any]
    ) -> bool:
        """Evaluate a single Haystack filter condition against a document.

        Supports operators: ==, !=, >, >=, <, <=, in, not in.

        Args:
            doc: Document to test.
            condition: Dict with 'field', 'operator', and 'value' keys.

        Returns:
            True if the condition is satisfied.
        """
        field = condition.get("field", "")
        op = condition.get("operator", "==")
        value = condition.get("value")

        # Resolve field value from meta or top-level attributes
        if field.startswith("meta."):
            meta_key = field[5:]
            meta = doc.meta if hasattr(doc, "meta") and doc.meta else {}
            doc_val = meta.get(meta_key)
        else:
            doc_val = getattr(doc, field, None)

        if op == "==":
            return doc_val == value
        elif op == "!=":
            return doc_val != value
        elif op == ">":
            return doc_val is not None and doc_val > value
        elif op == ">=":
            return doc_val is not None and doc_val >= value
        elif op == "<":
            return doc_val is not None and doc_val < value
        elif op == "<=":
            return doc_val is not None and doc_val <= value
        elif op == "in":
            return doc_val in value if value is not None else False
        elif op == "not in":
            return doc_val not in value if value is not None else True
        return False

    def write_documents(
        self,
        documents: List[Document],
        policy: str = "fail",
    ) -> int:
        """Write documents to the store.

        Embeddings are compressed using BitPolar quantization on write.
        Supports three duplicate handling policies:

        - ``"overwrite"``: Replace existing documents with the same ID.
        - ``"skip"``: Keep the existing document, ignore the new one.
        - ``"fail"``: Raise an error if a duplicate ID is encountered.

        Args:
            documents: List of Document objects to write.
            policy: How to handle duplicate document IDs. One of
                ``"fail"`` (default), ``"skip"``, or ``"overwrite"``.

        Returns:
            Number of documents actually written.

        Raises:
            ValueError: If policy is ``"fail"`` and a duplicate ID is found.
        """
        # Normalize policy to string for compatibility with both
        # haystack DuplicatePolicy enum and plain strings
        policy_str = policy.value if hasattr(policy, "value") else str(policy)

        written = 0
        for doc in documents:
            if doc.id is None:
                doc.id = str(uuid.uuid4())

            if doc.id in self._docs:
                if policy_str == "skip":
                    continue
                elif policy_str == "fail":
                    raise ValueError(
                        f"Document with ID '{doc.id}' already exists and policy is 'fail'"
                    )
                # "overwrite" falls through

            # Compress embedding if present
            code: Optional[np.ndarray] = None
            if doc.embedding is not None:
                dim = len(doc.embedding)
                self._ensure_quantizer(dim)
                vec = np.array(doc.embedding, dtype=np.float32)
                code = self._quantizer.encode(vec)

            self._docs[doc.id] = (doc, code)
            written += 1

        return written

    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents by their IDs.

        Silently ignores IDs that are not found in the store.

        Args:
            document_ids: List of document IDs to delete.
        """
        for doc_id in document_ids:
            self._docs.pop(doc_id, None)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search for similar documents using compressed inner product.

        Computes BitPolar-compressed inner product scores between the query
        and all stored documents (optionally filtered), returning the top-k.

        Args:
            query_embedding: Float query vector.
            top_k: Maximum number of results.
            filters: Optional metadata filters to pre-filter candidates.

        Returns:
            List of Document objects with score set, sorted by descending score.
        """
        if not self._docs:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        if self._quantizer is not None and len(q) != self._dim:
            raise ValueError(f"Query dimension {len(q)} != stored dimension {self._dim}")
        self._ensure_quantizer(len(q))

        if filters:
            candidates = self.filter_documents(filters)
        else:
            candidates = [doc for doc, _code in self._docs.values()]

        scored: List[Tuple[float, Document]] = []
        for doc in candidates:
            entry = self._docs.get(doc.id)
            if entry is None:
                continue
            _stored_doc, code = entry
            if code is None:
                continue
            score = float(self._quantizer.inner_product(code, q))
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Document] = []
        for score, doc in scored[:top_k]:
            result_doc = copy.copy(doc)
            result_doc.score = score
            results.append(result_doc)

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the store configuration to a dictionary.

        Returns:
            Dict with type and init_parameters for Haystack serialization.
        """
        if _HAS_HAYSTACK:
            return default_to_dict(self, bits=self._bits, seed=self._seed)
        return {
            "type": "bitpolar_haystack.document_store.BitPolarDocumentStore",
            "init_parameters": {"bits": self._bits, "seed": self._seed},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BitPolarDocumentStore":
        """Deserialize a store from a dictionary.

        Args:
            data: Serialized dict from ``to_dict()``.

        Returns:
            A new BitPolarDocumentStore instance.
        """
        if _HAS_HAYSTACK:
            return default_from_dict(cls, data)
        params = data.get("init_parameters", {})
        return cls(**params)
