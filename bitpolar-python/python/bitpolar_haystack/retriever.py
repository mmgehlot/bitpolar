"""Haystack 2.x Retriever component backed by BitPolar compressed search.

Provides a pipeline-ready component that queries a BitPolarDocumentStore
using compressed inner product scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from haystack import Document, component, default_from_dict, default_to_dict

    _HAS_HAYSTACK = True
except ImportError:
    _HAS_HAYSTACK = False

    # Provide a no-op component decorator fallback
    class _FakeComponent:
        @staticmethod
        def output_types(**kwargs):
            def decorator(cls):
                return cls
            return decorator

    class component:  # type: ignore[no-redef]
        output_types = _FakeComponent.output_types

        def __init_subclass__(cls, **kwargs):
            pass

    class Document:  # type: ignore[no-redef]
        """Minimal Document fallback."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

from bitpolar_haystack.document_store import BitPolarDocumentStore


if _HAS_HAYSTACK:
    @component
    class BitPolarRetriever:
        """Haystack 2.x retriever that searches a BitPolarDocumentStore.

        Uses BitPolar compressed inner product scoring to find the most
        relevant documents given a query embedding. Designed for use in
        Haystack pipelines.

        Args:
            document_store: The BitPolarDocumentStore to search against.
            top_k: Default number of documents to retrieve.
            filters: Default metadata filters applied to every query.

        Example:
            >>> store = BitPolarDocumentStore(bits=4)
            >>> retriever = BitPolarRetriever(document_store=store, top_k=10)
            >>> # In a pipeline:
            >>> result = retriever.run(query_embedding=[0.1]*384)
        """

        def __init__(
            self,
            document_store: BitPolarDocumentStore,
            top_k: int = 10,
            filters: Optional[Dict[str, Any]] = None,
        ):
            self._document_store = document_store
            self._top_k = top_k
            self._filters = filters

        @component.output_types(documents=List[Document])
        def run(
            self,
            query_embedding: List[float],
            top_k: Optional[int] = None,
            filters: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, List[Document]]:
            """Retrieve documents similar to the query embedding.

            Searches the document store using BitPolar compressed inner
            product scoring and returns the top results.

            Args:
                query_embedding: Float query vector.
                top_k: Number of documents to return. Overrides the default
                    set at init time if provided.
                filters: Metadata filters. Overrides the default filters
                    set at init time if provided.

            Returns:
                Dict with a single key ``"documents"`` mapping to a list
                of Document objects sorted by descending similarity score.
            """
            effective_top_k = top_k if top_k is not None else self._top_k
            effective_filters = filters if filters is not None else self._filters

            documents = self._document_store.search(
                query_embedding=query_embedding,
                top_k=effective_top_k,
                filters=effective_filters,
            )
            return {"documents": documents}

        def to_dict(self) -> Dict[str, Any]:
            """Serialize the retriever for Haystack pipeline export.

            Returns:
                Dict with type, init_parameters (including nested store config).
            """
            return default_to_dict(
                self,
                document_store=self._document_store.to_dict(),
                top_k=self._top_k,
                filters=self._filters,
            )

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "BitPolarRetriever":
            """Deserialize a retriever from a dictionary.

            Args:
                data: Serialized dict from ``to_dict()``.

            Returns:
                A new BitPolarRetriever instance.
            """
            init_params = data.get("init_parameters", {})
            store_data = init_params.pop("document_store", {})
            store = BitPolarDocumentStore.from_dict(store_data)
            return cls(document_store=store, **init_params)

else:
    # Fallback when haystack is not installed
    class BitPolarRetriever:  # type: ignore[no-redef]
        """Haystack 2.x retriever that searches a BitPolarDocumentStore.

        Uses BitPolar compressed inner product scoring to find the most
        relevant documents given a query embedding. Designed for use in
        Haystack pipelines.

        Args:
            document_store: The BitPolarDocumentStore to search against.
            top_k: Default number of documents to retrieve.
            filters: Default metadata filters applied to every query.

        Example:
            >>> store = BitPolarDocumentStore(bits=4)
            >>> retriever = BitPolarRetriever(document_store=store, top_k=10)
            >>> result = retriever.run(query_embedding=[0.1]*384)
        """

        def __init__(
            self,
            document_store: BitPolarDocumentStore,
            top_k: int = 10,
            filters: Optional[Dict[str, Any]] = None,
        ):
            self._document_store = document_store
            self._top_k = top_k
            self._filters = filters

        def run(
            self,
            query_embedding: List[float],
            top_k: Optional[int] = None,
            filters: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, List[Any]]:
            """Retrieve documents similar to the query embedding.

            Searches the document store using BitPolar compressed inner
            product scoring and returns the top results.

            Args:
                query_embedding: Float query vector.
                top_k: Number of documents to return. Overrides the default
                    set at init time if provided.
                filters: Metadata filters. Overrides the default filters
                    set at init time if provided.

            Returns:
                Dict with a single key ``"documents"`` mapping to a list
                of Document objects sorted by descending similarity score.
            """
            effective_top_k = top_k if top_k is not None else self._top_k
            effective_filters = filters if filters is not None else self._filters

            documents = self._document_store.search(
                query_embedding=query_embedding,
                top_k=effective_top_k,
                filters=effective_filters,
            )
            return {"documents": documents}

        def to_dict(self) -> Dict[str, Any]:
            """Serialize the retriever configuration.

            Returns:
                Dict with type and init_parameters.
            """
            return {
                "type": "bitpolar_haystack.retriever.BitPolarRetriever",
                "init_parameters": {
                    "document_store": self._document_store.to_dict(),
                    "top_k": self._top_k,
                    "filters": self._filters,
                },
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "BitPolarRetriever":
            """Deserialize a retriever from a dictionary.

            Args:
                data: Serialized dict from ``to_dict()``.

            Returns:
                A new BitPolarRetriever instance.
            """
            init_params = data.get("init_parameters", {})
            store_data = init_params.pop("document_store", {})
            store = BitPolarDocumentStore.from_dict(store_data)
            return cls(document_store=store, **init_params)
