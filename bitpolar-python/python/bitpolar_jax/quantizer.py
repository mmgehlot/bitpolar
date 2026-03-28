"""JAX/Flax integration for BitPolar vector quantization.

Provides functional APIs for compressing/decompressing JAX arrays and a
Flax linen Module for embedding compression in neural network architectures.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError(
        "JAX is required. Install with: pip install jax jaxlib"
    )

try:
    import flax.linen as flax_nn
except ImportError:
    flax_nn = None  # type: ignore[assignment]

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar is required. Install with: pip install bitpolar")


def _validate_bits(bits: int) -> None:
    """Validate quantization bit-width is in the supported range."""
    if not (3 <= bits <= 8):
        raise ValueError(f"bits must be 3-8, got {bits}")


# Module-level cache for quantizers keyed by (dim, bits, projections, seed)
_quantizer_cache: dict[Tuple[int, int, int, int], "_bp.TurboQuantizer"] = {}


def _get_quantizer(
    dim: int,
    bits: int = 4,
    projections: Optional[int] = None,
    seed: int = 42,
) -> "_bp.TurboQuantizer":
    """Get or create a cached TurboQuantizer instance."""
    proj = projections or max(dim // 4, 1)
    key = (dim, bits, proj, seed)
    if key not in _quantizer_cache:
        _quantizer_cache[key] = _bp.TurboQuantizer(
            dim=dim, bits=bits, projections=proj, seed=seed
        )
    return _quantizer_cache[key]


def compress(
    array: "jnp.ndarray",
    bits: int = 4,
    projections: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Compress a JAX array using BitPolar quantization.

    Accepts 1D (single vector) or 2D (batch of vectors) input and returns
    compressed codes as a numpy object array of uint8 arrays.

    Args:
        array: JAX array of shape (dim,) or (n, dim)
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed for deterministic compression

    Returns:
        numpy object array where each element is a uint8 code array.
        Shape (n,) for batch input, scalar for single vector.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((100, 384))
        >>> codes = compress(x, bits=4)
        >>> codes.shape
        (100,)
    """
    _validate_bits(bits)

    # Convert JAX array to numpy float32
    data = np.asarray(array, dtype=np.float32)

    single = data.ndim == 1
    if single:
        data = data.reshape(1, -1)

    if data.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {data.ndim}D")

    n, dim = data.shape
    q = _get_quantizer(dim, bits, projections, seed)

    codes = np.empty(n, dtype=object)
    for i in range(n):
        codes[i] = q.encode(np.ascontiguousarray(data[i]))

    if single:
        return codes[0]
    return codes


def decompress(
    codes: np.ndarray,
    dim: int,
    bits: int = 4,
    projections: Optional[int] = None,
    seed: int = 42,
) -> "jnp.ndarray":
    """Decompress BitPolar codes back to a JAX array.

    Args:
        codes: numpy object array of compressed codes (from :func:`compress`),
               or a single uint8 code array
        dim: Original vector dimension
        bits: Quantization precision used during compression
        projections: Number of QJL projections used during compression
        seed: Random seed used during compression

    Returns:
        JAX array of shape (n, dim) or (dim,) for single code

    Example:
        >>> codes = compress(jnp.ones((10, 384)), bits=4)
        >>> reconstructed = decompress(codes, dim=384, bits=4)
        >>> reconstructed.shape
        (10, 384)
    """
    _validate_bits(bits)
    q = _get_quantizer(dim, bits, projections, seed)

    # Handle single code vs batch
    if isinstance(codes, np.ndarray) and codes.dtype == np.uint8:
        # Single code
        vec = q.decode(codes)
        return jnp.array(vec, dtype=jnp.float32)

    # Object array of codes
    codes_arr = np.asarray(codes)
    if codes_arr.ndim == 0:
        # Single code wrapped in 0-d object array
        vec = q.decode(codes_arr.item())
        return jnp.array(vec, dtype=jnp.float32)

    n = len(codes_arr)
    result = np.empty((n, dim), dtype=np.float32)
    for i in range(n):
        result[i] = q.decode(codes_arr[i])

    return jnp.array(result, dtype=jnp.float32)


def inner_product(
    code: np.ndarray,
    query: "jnp.ndarray",
    bits: int = 4,
    projections: Optional[int] = None,
    seed: int = 42,
) -> float:
    """Compute approximate inner product between a compressed code and a JAX query.

    Args:
        code: uint8 compressed code from :func:`compress`
        query: JAX array of shape (dim,)
        bits: Quantization precision used during compression
        projections: Number of QJL projections used during compression
        seed: Random seed used during compression

    Returns:
        Approximate inner product as a Python float

    Example:
        >>> x = jnp.ones(384)
        >>> code = compress(x, bits=4)
        >>> score = inner_product(code, x, bits=4)
    """
    _validate_bits(bits)
    q_np = np.asarray(query, dtype=np.float32)

    if q_np.ndim != 1:
        raise ValueError(f"Expected 1D query, got {q_np.ndim}D")

    dim = q_np.shape[0]
    q = _get_quantizer(dim, bits, projections, seed)
    return float(q.inner_product(code, q_np))


if flax_nn is not None:

    class BitPolarFlaxModule(flax_nn.Module):
        """Flax linen Module that compresses input vectors using BitPolar.

        Applies BitPolar quantization as a layer in a Flax neural network.
        In training mode, passes through a straight-through estimator.
        At inference, compresses and immediately decompresses to simulate
        quantization noise.

        Attributes:
            bits: Quantization precision (3-8)
            projections: Number of QJL projections (default: dim/4)
            seed: Random seed
            passthrough_training: If True, skip quantization during training

        Example:
            >>> import jax, jax.numpy as jnp
            >>> from bitpolar_jax import BitPolarFlaxModule
            >>> module = BitPolarFlaxModule(bits=4)
            >>> params = module.init(jax.random.PRNGKey(0), jnp.ones((1, 384)))
            >>> out = module.apply(params, jnp.ones((1, 384)))
        """

        bits: int = 4
        projections: Optional[int] = None
        seed: int = 42
        passthrough_training: bool = True

        @flax_nn.compact
        def __call__(
            self,
            x: "jnp.ndarray",
            training: bool = False,
        ) -> "jnp.ndarray":
            """Compress and decompress input to simulate quantization.

            Args:
                x: Input JAX array of shape (batch, dim) or (dim,)
                training: If True and passthrough_training is True, return
                          input unchanged (straight-through estimator)

            Returns:
                Quantized (compressed then decompressed) JAX array,
                same shape as input
            """
            if training and self.passthrough_training:
                return x

            single = x.ndim == 1
            if single:
                x = x.reshape(1, -1)

            dim = x.shape[-1]
            codes = compress(
                x, bits=self.bits, projections=self.projections, seed=self.seed
            )
            result = decompress(
                codes, dim=dim, bits=self.bits,
                projections=self.projections, seed=self.seed,
            )

            if single:
                result = result.reshape(-1)

            return result

else:
    # Flax not installed — provide a helpful error
    class BitPolarFlaxModule:  # type: ignore[no-redef]
        """Placeholder — flax is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "Flax is required for BitPolarFlaxModule. "
                "Install with: pip install flax"
            )
