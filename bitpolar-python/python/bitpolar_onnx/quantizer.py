"""ONNX Runtime quantizer using BitPolar compression.

Compresses embeddings and model weights for ONNX models,
with support for embedding layer weight compression.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")

try:
    import onnx as _onnx
    from onnx import numpy_helper as _numpy_helper

    _HAS_ONNX = True
except ImportError:
    _onnx = None  # type: ignore[assignment]
    _numpy_helper = None  # type: ignore[assignment]
    _HAS_ONNX = False

try:
    import onnxruntime as _ort

    _HAS_ORT = True
except ImportError:
    _ort = None  # type: ignore[assignment]
    _HAS_ORT = False


class BitPolarONNXQuantizer:
    """BitPolar quantizer for ONNX Runtime models.

    Compresses embedding arrays using BitPolar quantization and
    provides utilities for compressing embedding weights within
    ONNX model files.

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
        self._quantizers: Dict[int, _bp.TurboQuantizer] = {}

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

    def quantize_embeddings(self, input_array: np.ndarray) -> np.ndarray:
        """Compress an embedding array using BitPolar quantization.

        Handles both single vectors and batches of vectors.

        Args:
            input_array: Input array of shape (dim,) for a single vector
                or (n, dim) for a batch.

        Returns:
            Compressed codes as a numpy array. For a single vector,
            returns shape matching the code size. For a batch, returns
            a list of codes packed into an object array.
        """
        arr = np.ascontiguousarray(input_array, dtype=np.float32)

        if arr.ndim == 1:
            quantizer = self._get_quantizer(len(arr))
            return quantizer.encode(arr)

        if arr.ndim == 2:
            n, dim = arr.shape
            quantizer = self._get_quantizer(dim)
            codes = []
            for i in range(n):
                codes.append(quantizer.encode(arr[i]))
            return np.array(codes)

        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        """Dequantize compressed codes back to approximate vectors.

        Args:
            codes: Compressed codes from ``quantize_embeddings``.
                Single code or batch of codes.

        Returns:
            Approximate float32 reconstructions.
        """
        codes = np.asarray(codes)

        if codes.ndim == 1:
            for dim, quantizer in self._quantizers.items():
                try:
                    return quantizer.decode(codes)
                except Exception:
                    continue
            raise ValueError(
                "Cannot dequantize: no matching quantizer. "
                "Call quantize_embeddings() first."
            )

        if codes.ndim == 2:
            results = []
            for i in range(codes.shape[0]):
                results.append(self.dequantize(codes[i]))
            return np.stack(results)

        raise ValueError(f"Expected 1D or 2D codes, got {codes.ndim}D")

    def create_quantize_node(self, dim: int) -> Dict[str, Any]:
        """Create an ONNX-compatible node configuration for quantization.

        Returns a dictionary describing how BitPolar quantization would
        be applied, suitable for custom ONNX graph builders.

        Args:
            dim: Input dimension for the quantization node.

        Returns:
            Dictionary with ONNX node configuration including:
                - op_type: Node operation type
                - domain: Custom domain name
                - attributes: Quantization parameters
                - inputs: Expected input names and shapes
                - outputs: Expected output names and shapes
        """
        proj = self._projections or max(dim // 4, 1)
        code_dim = (dim * self._bits + 7) // 8  # bytes needed

        return {
            "op_type": "BitPolarQuantize",
            "domain": "com.bitpolar",
            "attributes": {
                "bits": self._bits,
                "seed": self._seed,
                "projections": proj,
                "dim": dim,
            },
            "inputs": [
                {"name": "input", "shape": ["batch_size", dim], "dtype": "float32"},
            ],
            "outputs": [
                {
                    "name": "codes",
                    "shape": ["batch_size", code_dim],
                    "dtype": "uint8",
                },
            ],
        }

    def compress_model_embeddings(
        self,
        model_path: str,
        output_path: str,
        embedding_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load an ONNX model, compress embedding weights, and save.

        Finds embedding initializers in the ONNX model (2D float tensors),
        compresses each row using BitPolar, and replaces the original
        weights with compressed codes. The compressed model must use a
        custom runtime or dequantization step for inference.

        Args:
            model_path: Path to the input ONNX model file.
            output_path: Path to save the compressed ONNX model.
            embedding_names: Optional list of initializer names to
                compress. If None, compresses all 2D float32 initializers
                with names containing "embed" (case-insensitive).

        Returns:
            Dictionary with compression statistics:
                - compressed_tensors: Number of tensors compressed
                - tensor_names: List of compressed tensor names
                - original_bytes: Total original size in bytes
                - compressed_bytes: Total compressed size in bytes
                - compression_ratio: Overall compression ratio

        Raises:
            ImportError: If the onnx package is not installed.
            FileNotFoundError: If the model file does not exist.
        """
        if not _HAS_ONNX:
            raise ImportError(
                "onnx required for model compression. "
                "Install with: pip install onnx"
            )

        model = _onnx.load(model_path)
        stats: Dict[str, Any] = {
            "compressed_tensors": 0,
            "tensor_names": [],
            "original_bytes": 0,
            "compressed_bytes": 0,
            "compression_ratio": 0.0,
        }

        for initializer in model.graph.initializer:
            name = initializer.name

            # Filter: only 2D float tensors
            arr = _numpy_helper.to_array(initializer)
            if arr.ndim != 2:
                continue
            if arr.dtype not in (np.float32, np.float16):
                continue

            # Filter by name if specified, otherwise look for embedding layers
            if embedding_names is not None:
                if name not in embedding_names:
                    continue
            else:
                if "embed" not in name.lower():
                    continue

            arr = arr.astype(np.float32)
            n_rows, dim = arr.shape

            quantizer = self._get_quantizer(dim)
            compressed_rows = []
            for i in range(n_rows):
                code = quantizer.encode(arr[i])
                compressed_rows.append(code)

            # Stack compressed codes into a 2D array
            compressed = np.array(compressed_rows)

            # Replace initializer with compressed data
            new_tensor = _numpy_helper.from_array(compressed, name=name)
            initializer.CopyFrom(new_tensor)

            original_size = arr.nbytes
            compressed_size = compressed.nbytes

            stats["compressed_tensors"] += 1
            stats["tensor_names"].append(name)
            stats["original_bytes"] += original_size
            stats["compressed_bytes"] += compressed_size

        if stats["compressed_bytes"] > 0:
            stats["compression_ratio"] = round(
                stats["original_bytes"] / stats["compressed_bytes"], 2
            )

        _onnx.save(model, output_path)
        return stats
