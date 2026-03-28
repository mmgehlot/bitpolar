"""AWS Bedrock middleware for BitPolar embedding compression.

Wraps the boto3 bedrock-runtime client to automatically compress embedding
responses from Amazon Titan, Cohere, and other Bedrock embedding models,
and provides compressed search over stored embeddings.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import boto3
except ImportError:
    raise ImportError(
        "boto3 is required. Install with: pip install boto3"
    )

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar is required. Install with: pip install bitpolar")


def _validate_bits(bits: int) -> None:
    """Validate quantization bit-width is in the supported range."""
    if not (3 <= bits <= 8):
        raise ValueError(f"bits must be 3-8, got {bits}")


class BitPolarBedrockClient:
    """AWS Bedrock client with automatic BitPolar embedding compression.

    Wraps boto3's bedrock-runtime client to embed text via Bedrock models
    and immediately compress the resulting vectors using BitPolar quantization.
    Also supports batch embedding and compressed similarity search.

    Args:
        bits: Quantization precision (3-8, default 4)
        seed: Random seed for deterministic compression
        region_name: AWS region (default: from boto3 session)
        boto3_session: Optional pre-configured boto3.Session

    Example:
        >>> from bitpolar_bedrock import BitPolarBedrockClient
        >>> client = BitPolarBedrockClient(bits=4, region_name="us-east-1")
        >>> result = client.embed_and_compress("Hello world")
        >>> result["code"]  # compressed uint8 bytes
        >>> result["dim"]   # original dimension
    """

    def __init__(
        self,
        bits: int = 4,
        seed: int = 42,
        region_name: Optional[str] = None,
        boto3_session: Optional["boto3.Session"] = None,
    ):
        _validate_bits(bits)
        self._bits = bits
        self._seed = seed

        session = boto3_session or boto3.Session(region_name=region_name)
        self._client = session.client("bedrock-runtime")
        self._quantizers: Dict[int, "_bp.TurboQuantizer"] = {}

    def _get_quantizer(self, dim: int) -> "_bp.TurboQuantizer":
        """Get or create a TurboQuantizer for the given dimension."""
        if dim not in self._quantizers:
            proj = max(dim // 4, 1)
            self._quantizers[dim] = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )
        return self._quantizers[dim]

    def _invoke_embedding(
        self,
        text: str,
        model_id: str,
    ) -> List[float]:
        """Call Bedrock to get an embedding vector for a single text.

        Args:
            text: Input text to embed
            model_id: Bedrock model identifier

        Returns:
            Embedding as a list of floats
        """
        # Build request body based on model family
        if "titan" in model_id.lower():
            body = json.dumps({"inputText": text})
        elif "cohere" in model_id.lower():
            body = json.dumps({
                "texts": [text],
                "input_type": "search_document",
            })
        else:
            # Generic fallback
            body = json.dumps({"inputText": text})

        response = self._client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())

        # Parse response based on model family
        if "titan" in model_id.lower():
            return result["embedding"]
        elif "cohere" in model_id.lower():
            return result["embeddings"][0]
        elif "embedding" in result:
            return result["embedding"]
        elif "embeddings" in result:
            return result["embeddings"][0]
        else:
            raise ValueError(
                f"Could not extract embedding from response keys: {list(result.keys())}"
            )

    def embed_and_compress(
        self,
        text: str,
        model_id: str = "amazon.titan-embed-text-v2:0",
    ) -> Dict[str, Any]:
        """Embed text via Bedrock and compress the result with BitPolar.

        Args:
            text: Input text to embed
            model_id: Bedrock embedding model identifier

        Returns:
            Dict with keys:
                - code: compressed uint8 numpy array
                - dim: original vector dimension
                - model_id: model used for embedding
                - bits: quantization bit-width

        Example:
            >>> result = client.embed_and_compress("Hello world")
            >>> result["code"].dtype
            dtype('uint8')
        """
        embedding = self._invoke_embedding(text, model_id)
        arr = np.array(embedding, dtype=np.float32)
        dim = len(arr)

        q = self._get_quantizer(dim)
        code = q.encode(arr)

        return {
            "code": code,
            "dim": dim,
            "model_id": model_id,
            "bits": self._bits,
        }

    def batch_embed_and_compress(
        self,
        texts: List[str],
        model_id: str = "amazon.titan-embed-text-v2:0",
    ) -> List[Dict[str, Any]]:
        """Embed and compress multiple texts.

        Processes texts sequentially through Bedrock (Bedrock does not
        support true batch embedding for all models). Each result is
        independently compressed.

        Args:
            texts: List of input texts
            model_id: Bedrock embedding model identifier

        Returns:
            List of compressed result dicts (same format as embed_and_compress)

        Example:
            >>> results = client.batch_embed_and_compress(
            ...     ["Hello", "World"],
            ...     model_id="amazon.titan-embed-text-v2:0",
            ... )
            >>> len(results)
            2
        """
        results = []
        for text in texts:
            results.append(self.embed_and_compress(text, model_id))
        return results

    def search(
        self,
        query_text: str,
        stored_codes: List[Dict[str, Any]],
        model_id: str = "amazon.titan-embed-text-v2:0",
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Embed a query and search against pre-compressed codes.

        Embeds the query text via Bedrock, then scores it against each
        stored compressed code using BitPolar's asymmetric inner product.

        Args:
            query_text: Query text to embed and search with
            stored_codes: List of compressed result dicts from
                          embed_and_compress or batch_embed_and_compress
            model_id: Bedrock embedding model identifier
            top_k: Number of top results to return

        Returns:
            List of dicts with keys:
                - index: position in stored_codes
                - score: approximate inner product score
            Sorted by score descending, length min(top_k, len(stored_codes))

        Example:
            >>> docs = client.batch_embed_and_compress(["cat", "dog", "fish"])
            >>> results = client.search("pet", docs, top_k=2)
            >>> results[0]["index"]  # index of best match
        """
        if not stored_codes:
            return []

        # Embed the query
        query_embedding = self._invoke_embedding(query_text, model_id)
        q_arr = np.array(query_embedding, dtype=np.float32)
        dim = len(q_arr)
        q = self._get_quantizer(dim)

        # Score against all stored codes
        scores = np.empty(len(stored_codes), dtype=np.float32)
        for i, entry in enumerate(stored_codes):
            code = entry["code"]
            if isinstance(code, bytes):
                code = np.frombuffer(code, dtype=np.uint8)
            scores[i] = q.inner_product(code, q_arr)

        # Top-K selection
        k = min(top_k, len(stored_codes))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [
            {"index": int(idx), "score": float(scores[idx])}
            for idx in top_idx
        ]

    def decompress(self, entry: Dict[str, Any]) -> np.ndarray:
        """Decompress a single stored code back to float32 vector.

        Args:
            entry: Compressed result dict from embed_and_compress

        Returns:
            float32 numpy array of shape (dim,)
        """
        code = entry["code"]
        if isinstance(code, bytes):
            code = np.frombuffer(code, dtype=np.uint8)
        dim = entry["dim"]
        q = self._get_quantizer(dim)
        return q.decode(code)

    @property
    def bits(self) -> int:
        """Quantization bit-width."""
        return self._bits

    def __repr__(self) -> str:
        return f"BitPolarBedrockClient(bits={self._bits}, seed={self._seed})"
