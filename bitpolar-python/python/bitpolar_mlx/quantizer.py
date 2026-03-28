"""Apple MLX quantizer using BitPolar compression.

Compresses MLX and numpy arrays using BitPolar quantization for
efficient on-device inference on Apple Silicon.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")

# Optional MLX import — works with plain numpy too
try:
    import mlx.core as mx

    _HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    _HAS_MLX = False


def _to_numpy(array: Union[np.ndarray, "mx.array"]) -> np.ndarray:
    """Convert an MLX array or numpy array to contiguous float32 numpy."""
    if _HAS_MLX and isinstance(array, mx.array):
        return np.array(array, dtype=np.float32)
    return np.ascontiguousarray(array, dtype=np.float32)


class BitPolarMLXQuantizer:
    """BitPolar quantizer for Apple MLX arrays.

    Compresses MLX or numpy arrays using BitPolar quantization.
    Handles automatic conversion between mlx.core.array and numpy.

    Args:
        bits: Quantization precision (3-8, default 4)
        seed: Random seed for deterministic compression
        projections: Number of QJL projections (default: dim // 4)
    """

    def __init__(
        self,
        bits: int = 4,
        seed: int = 42,
        projections: Optional[int] = None,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")

        self._bits = bits
        self._seed = seed
        self._projections = projections
        self._quantizers: dict[int, _bp.TurboQuantizer] = {}

    def _get_quantizer(self, dim: int) -> _bp.TurboQuantizer:
        """Return a cached quantizer for the given dimension."""
        if dim not in self._quantizers:
            proj = self._projections or max(dim // 4, 1)
            self._quantizers[dim] = _bp.TurboQuantizer(
                dim=dim,
                bits=self._bits,
                projections=proj,
                seed=self._seed,
            )
        return self._quantizers[dim]

    def compress(
        self, array: Union[np.ndarray, "mx.array"]
    ) -> np.ndarray:
        """Compress a single vector using BitPolar quantization.

        Args:
            array: Input vector as MLX array or numpy array of shape (dim,).

        Returns:
            Compressed BitPolar code as numpy array.
        """
        vec = _to_numpy(array).ravel()
        quantizer = self._get_quantizer(len(vec))
        return quantizer.encode(vec)

    def decompress(self, code: np.ndarray) -> np.ndarray:
        """Decompress a BitPolar code back to an approximate vector.

        Uses the quantizer's decode method to reconstruct an approximate
        version of the original vector from the compressed code.

        Args:
            code: Compressed BitPolar code from ``compress``.

        Returns:
            Approximate float32 numpy array reconstruction.
        """
        code = np.asarray(code)
        # Find the matching quantizer by checking all cached dims
        for dim, quantizer in self._quantizers.items():
            try:
                return quantizer.decode(code)
            except Exception:
                continue
        raise ValueError(
            "Cannot decompress: no matching quantizer found. "
            "Call compress() first to initialise a quantizer."
        )

    def inner_product(
        self, code: np.ndarray, query: Union[np.ndarray, "mx.array"]
    ) -> float:
        """Compute approximate inner product between a code and query vector.

        Args:
            code: Compressed BitPolar code from ``compress``.
            query: Query vector as MLX array or numpy array of shape (dim,).

        Returns:
            Approximate inner product score.
        """
        query_np = _to_numpy(query).ravel()
        quantizer = self._get_quantizer(len(query_np))
        return float(quantizer.inner_product(code, query_np))

    def compress_batch(
        self, arrays: List[Union[np.ndarray, "mx.array"]]
    ) -> List[np.ndarray]:
        """Compress a batch of vectors.

        Args:
            arrays: List of input vectors, each of shape (dim,).

        Returns:
            List of compressed BitPolar codes.
        """
        if not arrays:
            return []

        # Determine dimension from first array
        first = _to_numpy(arrays[0]).ravel()
        dim = len(first)
        quantizer = self._get_quantizer(dim)

        codes = [quantizer.encode(first)]
        for arr in arrays[1:]:
            vec = _to_numpy(arr).ravel()
            if len(vec) != dim:
                raise ValueError(
                    f"Dimension mismatch: expected {dim}, got {len(vec)}"
                )
            codes.append(quantizer.encode(vec))

        return codes
