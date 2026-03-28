"""Ollama embedding compression using BitPolar quantization.

Calls the Ollama REST API for embeddings, then compresses them
with BitPolar for memory-efficient storage and approximate search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")

try:
    import requests as _requests
except ImportError:
    raise ImportError("requests required. Install with: pip install requests")


class BitPolarOllamaClient:
    """Ollama client with BitPolar embedding compression.

    Wraps the Ollama embedding API and compresses returned vectors
    using BitPolar quantization for 4-8x memory reduction.

    Args:
        host: Ollama server URL (default "http://localhost:11434")
        bits: Quantization precision (3-8, default 4)
        seed: Random seed for deterministic compression
        projections: Number of QJL projections (default: dim // 4)
        timeout: HTTP request timeout in seconds (default 30)
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        bits: int = 4,
        seed: int = 42,
        projections: Optional[int] = None,
        timeout: int = 30,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        self._host = host.rstrip("/")
        self._bits = bits
        self._seed = seed
        self._projections = projections
        self._timeout = timeout
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._dim: Optional[int] = None

    def _ensure_quantizer(self, dim: int) -> _bp.TurboQuantizer:
        """Lazily initialise the quantizer on first use or dim change."""
        if self._quantizer is None or self._dim != dim:
            proj = self._projections or max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim,
                bits=self._bits,
                projections=proj,
                seed=self._seed,
            )
            self._dim = dim
        return self._quantizer

    def _call_embed_api(self, text: str, model: str) -> np.ndarray:
        """Call the Ollama embedding API and return raw vector.

        Args:
            text: Input text to embed.
            model: Ollama model name.

        Returns:
            float32 numpy array of shape (dim,).

        Raises:
            RuntimeError: If the API call fails.
        """
        url = f"{self._host}/api/embeddings"
        payload = {"model": model, "prompt": text}

        try:
            resp = _requests.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except _requests.RequestException as exc:
            raise RuntimeError(f"Ollama API request failed: {exc}") from exc

        data = resp.json()
        if "embedding" not in data:
            raise RuntimeError(
                f"Unexpected Ollama response format: {list(data.keys())}"
            )

        return np.asarray(data["embedding"], dtype=np.float32)

    def embed(
        self, text: str, model: str = "nomic-embed-text"
    ) -> np.ndarray:
        """Get a raw (uncompressed) embedding from Ollama.

        Args:
            text: Input text to embed.
            model: Ollama model name (default "nomic-embed-text").

        Returns:
            float32 numpy array of shape (dim,).
        """
        return self._call_embed_api(text, model)

    def embed_and_store(
        self, text: str, model: str = "nomic-embed-text"
    ) -> Dict[str, Any]:
        """Embed text and return a compressed BitPolar code with metadata.

        Args:
            text: Input text to embed.
            model: Ollama model name (default "nomic-embed-text").

        Returns:
            Dictionary with keys:
                - code: Compressed BitPolar code as numpy array
                - raw_embedding: Original float32 embedding
                - dim: Embedding dimension
                - model: Model name used
                - bits: Quantization precision
                - text: Original input text
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        raw = self._call_embed_api(text, model)
        quantizer = self._ensure_quantizer(len(raw))
        code = quantizer.encode(raw)

        return {
            "code": code,
            "raw_embedding": raw,
            "dim": len(raw),
            "model": model,
            "bits": self._bits,
            "text": text,
        }

    def search(
        self,
        query: str,
        stored_codes: List[Dict[str, Any]],
        model: str = "nomic-embed-text",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search stored codes for the most similar to a query.

        Embeds the query via Ollama, then scores against all stored
        codes using BitPolar inner product.

        Args:
            query: Query text to search for.
            stored_codes: List of dicts from ``embed_and_store``, each
                containing at least a "code" key.
            model: Ollama model name (default "nomic-embed-text").
            top_k: Number of top results to return.

        Returns:
            List of dicts sorted by descending similarity, each with:
                - score: Inner product similarity score
                - index: Original index in stored_codes
                - text: Original text (if available)
                - metadata: Any extra keys from the stored entry
        """
        if not stored_codes:
            return []

        query_vec = self._call_embed_api(query, model)
        quantizer = self._ensure_quantizer(len(query_vec))

        scored: List[Dict[str, Any]] = []
        for i, entry in enumerate(stored_codes):
            code = entry["code"]
            score = float(quantizer.inner_product(code, query_vec))
            result: Dict[str, Any] = {
                "score": score,
                "index": i,
            }
            if "text" in entry:
                result["text"] = entry["text"]
            # Include any extra metadata
            for key in entry:
                if key not in ("code", "raw_embedding", "text"):
                    result.setdefault(key, entry[key])
            scored.append(result)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: min(top_k, len(scored))]
