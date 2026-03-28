"""LlamaIndex VectorStore backed by BitPolar compression.

Implements the LlamaIndex BasePydanticVectorStore interface using
BitPolar's near-optimal quantization for 4-8x memory reduction.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
except ImportError:
    raise ImportError(
        "llama-index-core required. Install with: pip install llama-index-core>=0.11"
    )

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


class BitPolarVectorStore(BasePydanticVectorStore):
    """LlamaIndex VectorStore using BitPolar compression.

    Stores document embeddings in compressed form using near-optimal
    vector quantization. No training required — instant indexing.

    Args:
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed for deterministic compression
    """

    # Pydantic fields (LlamaIndex requires Pydantic model)
    stores_text: bool = True
    is_embedding_query: bool = True

    # Configuration stored as Pydantic fields
    _bits: int = 4
    _projections: Optional[int] = None
    _seed: int = 42

    # Internal state (not serialized)
    _quantizer: Optional[Any] = None
    _store: Dict[str, Any] = {}  # id -> (code, node)
    _dim: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._bits = bits
        self._projections = projections
        self._seed = seed
        self._quantizer = None
        self._store = {}
        self._dim = None

    def _ensure_quantizer(self, dim: int) -> None:
        """Initialize quantizer on first vector insertion."""
        if self._quantizer is None:
            self._dim = dim
            proj = self._projections or max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )

    @property
    def client(self) -> None:
        """No external client needed — in-memory store."""
        return None

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes with embeddings to the compressed store.

        Args:
            nodes: List of BaseNode objects with embeddings set

        Returns:
            List of node IDs that were added
        """
        ids = []
        for node in nodes:
            embedding = node.get_embedding()
            if embedding is None:
                continue

            vec = np.array(embedding, dtype=np.float32)
            self._ensure_quantizer(len(vec))

            code = self._quantizer.encode(vec)
            node_id = node.node_id or str(uuid.uuid4())
            self._store[node_id] = (code, node)
            ids.append(node_id)

        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete all nodes associated with a document reference ID.

        Args:
            ref_doc_id: The document reference ID to delete
        """
        to_delete = []
        for node_id, (_, node) in self._store.items():
            if hasattr(node, "ref_doc_id") and node.ref_doc_id == ref_doc_id:
                to_delete.append(node_id)

        for node_id in to_delete:
            del self._store[node_id]

    def query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Search compressed embeddings by approximate inner product.

        Args:
            query: VectorStoreQuery with query_embedding and similarity_top_k

        Returns:
            VectorStoreQueryResult with nodes, similarities, and ids
        """
        if query.query_embedding is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        q_vec = np.array(query.query_embedding, dtype=np.float32)
        top_k = query.similarity_top_k or 10

        if not self._store or self._quantizer is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        # Score all stored codes against the query
        scored = []
        for node_id, (code, node) in self._store.items():
            # Apply metadata filters if provided
            if query.filters and not self._matches_filters(node, query.filters):
                continue
            score = self._quantizer.inner_product(code, q_vec)
            scored.append((node_id, score, node))

        # Sort by descending score and take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        result_nodes = []
        result_scores = []
        result_ids = []

        for node_id, score, node in scored:
            result_nodes.append(node)
            result_scores.append(float(score))
            result_ids.append(node_id)

        return VectorStoreQueryResult(
            nodes=result_nodes,
            similarities=result_scores,
            ids=result_ids,
        )

    def _matches_filters(self, node: BaseNode, filters: Any) -> bool:
        """Check if a node matches metadata filters.

        Supports basic equality, range, and in-list filters on node metadata.
        """
        if filters is None:
            return True

        # LlamaIndex MetadataFilters support
        if hasattr(filters, "filters"):
            for f in filters.filters:
                meta = node.metadata or {}
                key = f.key
                value = meta.get(key)
                if hasattr(f, "operator"):
                    op = str(f.operator).lower()
                    if op in ("==", "eq") and value != f.value:
                        return False
                    elif op in ("!=", "ne") and value == f.value:
                        return False
                    elif op in (">", "gt") and (value is None or value <= f.value):
                        return False
                    elif op in ("<", "lt") and (value is None or value >= f.value):
                        return False
                    elif op == "in" and value not in f.value:
                        return False
                elif value != f.value:
                    return False
        return True

    def __len__(self) -> int:
        return len(self._store)
