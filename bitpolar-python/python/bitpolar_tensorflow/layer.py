"""TensorFlow/Keras integration for BitPolar vector quantization.

Provides custom Keras layers for compressing tensors and embeddings
using BitPolar's near-optimal quantization, plus functional APIs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "TensorFlow is required. Install with: pip install tensorflow>=2.12"
    )

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar is required. Install with: pip install bitpolar")


def _validate_bits(bits: int) -> None:
    """Validate quantization bit-width is in the supported range."""
    if not (3 <= bits <= 8):
        raise ValueError(f"bits must be 3-8, got {bits}")


# Module-level quantizer cache keyed by (dim, bits, projections, seed)
_quantizer_cache: Dict[Tuple[int, int, int, int], "_bp.TurboQuantizer"] = {}


def _get_quantizer(
    dim: int,
    bits: int = 4,
    projections: Optional[int] = None,
    seed: int = 42,
) -> "_bp.TurboQuantizer":
    """Get or create a cached TurboQuantizer."""
    proj = projections or max(dim // 4, 1)
    key = (dim, bits, proj, seed)
    if key not in _quantizer_cache:
        _quantizer_cache[key] = _bp.TurboQuantizer(
            dim=dim, bits=bits, projections=proj, seed=seed
        )
    return _quantizer_cache[key]


def compress_tensor(
    tensor: tf.Tensor,
    bits: int = 4,
    projections: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Compress a TensorFlow tensor using BitPolar quantization.

    Args:
        tensor: float32 tensor of shape (dim,) or (n, dim)
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed

    Returns:
        numpy object array of uint8 code arrays.
        Single code for 1D input, array of codes for 2D input.

    Example:
        >>> import tensorflow as tf
        >>> t = tf.random.normal((100, 384))
        >>> codes = compress_tensor(t, bits=4)
        >>> codes.shape
        (100,)
    """
    _validate_bits(bits)
    data = tensor.numpy().astype(np.float32)

    single = data.ndim == 1
    if single:
        data = data.reshape(1, -1)

    if data.ndim != 2:
        raise ValueError(f"Expected 1D or 2D tensor, got {data.ndim}D")

    n, dim = data.shape
    q = _get_quantizer(dim, bits, projections, seed)

    codes = np.empty(n, dtype=object)
    for i in range(n):
        codes[i] = q.encode(np.ascontiguousarray(data[i]))

    if single:
        return codes[0]
    return codes


def decompress_tensor(
    codes: np.ndarray,
    dim: int,
    bits: int = 4,
    projections: Optional[int] = None,
    seed: int = 42,
) -> tf.Tensor:
    """Decompress BitPolar codes back to a TensorFlow tensor.

    Args:
        codes: numpy object array of compressed codes or single uint8 array
        dim: Original vector dimension
        bits: Quantization precision used during compression
        projections: Number of QJL projections used during compression
        seed: Random seed used during compression

    Returns:
        tf.Tensor of shape (n, dim) or (dim,) for single code

    Example:
        >>> codes = compress_tensor(tf.random.normal((10, 384)), bits=4)
        >>> restored = decompress_tensor(codes, dim=384, bits=4)
        >>> restored.shape
        TensorShape([10, 384])
    """
    _validate_bits(bits)
    q = _get_quantizer(dim, bits, projections, seed)

    # Single code
    if isinstance(codes, np.ndarray) and codes.dtype == np.uint8:
        vec = q.decode(codes)
        return tf.constant(vec, dtype=tf.float32)

    codes_arr = np.asarray(codes)
    if codes_arr.ndim == 0:
        vec = q.decode(codes_arr.item())
        return tf.constant(vec, dtype=tf.float32)

    n = len(codes_arr)
    result = np.empty((n, dim), dtype=np.float32)
    for i in range(n):
        result[i] = q.decode(codes_arr[i])

    return tf.constant(result, dtype=tf.float32)


class BitPolarLayer(tf.keras.layers.Layer):
    """Keras layer that applies BitPolar quantization to inputs.

    Compresses input tensors through BitPolar's near-optimal quantization
    and immediately decompresses, simulating quantization noise during
    training. At inference, the quantized representation is used.

    Args:
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed

    Example:
        >>> layer = BitPolarLayer(bits=4)
        >>> out = layer(tf.random.normal((32, 384)))
        >>> out.shape
        TensorShape([32, 384])
    """

    def __init__(
        self,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        _validate_bits(bits)
        self._bits = bits
        self._projections = projections
        self._seed = seed
        self._quantizer: Optional["_bp.TurboQuantizer"] = None
        self._dim: Optional[int] = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Initialize the quantizer based on the input dimension."""
        dim = int(input_shape[-1])
        self._dim = dim
        proj = self._projections or max(dim // 4, 1)
        self._quantizer = _bp.TurboQuantizer(
            dim=dim, bits=self._bits, projections=proj, seed=self._seed
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compress and decompress inputs through BitPolar quantization.

        Args:
            inputs: float32 tensor of shape (batch, dim) or (dim,)

        Returns:
            Quantized tensor with same shape as inputs
        """
        assert self._quantizer is not None, "Layer not built. Call build() first."

        single = inputs.shape.ndims == 1
        if single:
            inputs = tf.expand_dims(inputs, 0)

        data = inputs.numpy().astype(np.float32)
        n, dim = data.shape

        result = np.empty_like(data)
        for i in range(n):
            code = self._quantizer.encode(np.ascontiguousarray(data[i]))
            result[i] = self._quantizer.decode(code)

        output = tf.constant(result, dtype=tf.float32)
        if single:
            output = tf.squeeze(output, axis=0)
        return output

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer configuration."""
        config = super().get_config()
        config.update({
            "bits": self._bits,
            "projections": self._projections,
            "seed": self._seed,
        })
        return config


class BitPolarEmbedding(tf.keras.layers.Layer):
    """Keras layer that stores embeddings in BitPolar-compressed form.

    Drop-in replacement for tf.keras.layers.Embedding that stores vectors
    using BitPolar quantization and supports approximate inner product
    scoring without full decompression.

    Args:
        num_embeddings: Size of the embedding vocabulary
        embedding_dim: Dimension of each embedding vector
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed

    Example:
        >>> layer = BitPolarEmbedding(num_embeddings=10000, embedding_dim=384)
        >>> layer.build([])
        >>> layer.set_embeddings(tf.random.normal((10000, 384)))
        >>> out = layer(tf.constant([0, 1, 2]))
        >>> out.shape
        TensorShape([3, 384])
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        _validate_bits(bits)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._bits = bits
        self._projections = projections
        self._seed = seed
        self._quantizer: Optional["_bp.TurboQuantizer"] = None
        self._codes: List[np.ndarray] = []

    def build(self, input_shape: Any) -> None:
        """Initialize the quantizer."""
        proj = self._projections or max(self.embedding_dim // 4, 1)
        self._quantizer = _bp.TurboQuantizer(
            dim=self.embedding_dim,
            bits=self._bits,
            projections=proj,
            seed=self._seed,
        )
        # Initialize with zeros (call set_embeddings to populate)
        if not self._codes:
            data = np.zeros(
                (self.num_embeddings, self.embedding_dim), dtype=np.float32
            )
            self._codes = [
                self._quantizer.encode(data[i]) for i in range(self.num_embeddings)
            ]
        super().build(input_shape)

    def set_embeddings(self, embeddings: tf.Tensor) -> None:
        """Load embedding vectors into compressed storage.

        Args:
            embeddings: float32 tensor of shape (num_embeddings, embedding_dim)
        """
        data = embeddings.numpy().astype(np.float32)
        if data.shape != (self.num_embeddings, self.embedding_dim):
            raise ValueError(
                f"Expected shape ({self.num_embeddings}, {self.embedding_dim}), "
                f"got {data.shape}"
            )

        assert self._quantizer is not None, "Layer not built."
        self._codes = [
            self._quantizer.encode(np.ascontiguousarray(data[i]))
            for i in range(self.num_embeddings)
        ]

    def call(self, indices: tf.Tensor) -> tf.Tensor:
        """Look up embeddings by index, decompressing on-the-fly.

        Args:
            indices: Integer tensor of embedding indices

        Returns:
            float32 tensor of shape (*indices.shape, embedding_dim)
        """
        assert self._quantizer is not None, "Layer not built."

        idx_flat = indices.numpy().reshape(-1).tolist()
        result = np.empty((len(idx_flat), self.embedding_dim), dtype=np.float32)

        for i, idx in enumerate(idx_flat):
            if idx < 0 or idx >= self.num_embeddings:
                raise IndexError(
                    f"Embedding index {idx} out of range [0, {self.num_embeddings})"
                )
            result[i] = self._quantizer.decode(self._codes[idx])

        output = tf.constant(result, dtype=tf.float32)
        output_shape = list(indices.shape) + [self.embedding_dim]
        return tf.reshape(output, output_shape)

    def score(self, indices: tf.Tensor, query: tf.Tensor) -> tf.Tensor:
        """Score compressed embeddings against a query via inner product.

        Args:
            indices: Integer tensor of embedding indices
            query: float32 tensor of shape (embedding_dim,)

        Returns:
            float32 tensor of scores, same shape as indices
        """
        assert self._quantizer is not None, "Layer not built."

        idx_flat = indices.numpy().reshape(-1).tolist()
        q_np = query.numpy().astype(np.float32)

        scores = np.empty(len(idx_flat), dtype=np.float32)
        for i, idx in enumerate(idx_flat):
            scores[i] = self._quantizer.inner_product(self._codes[idx], q_np)

        return tf.reshape(
            tf.constant(scores, dtype=tf.float32), indices.shape
        )

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer configuration."""
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "bits": self._bits,
            "projections": self._projections,
            "seed": self._seed,
        })
        return config
