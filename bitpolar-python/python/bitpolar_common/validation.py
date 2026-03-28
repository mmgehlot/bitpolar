"""Common validation utilities for BitPolar integrations.

Provides reusable validation functions used across all integration
packages. Import these instead of duplicating validation logic.

Usage:
    from bitpolar_common.validation import validate_bits, validate_vector
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence, Union

import numpy as np


def validate_bits(bits: int) -> None:
    """Validate quantization bit-width is in the supported range.

    Args:
        bits: Quantization precision to validate

    Raises:
        ValueError: If bits is not in [3, 8]
    """
    if not (3 <= bits <= 8):
        raise ValueError(f"bits must be 3-8, got {bits}")


def validate_dim(dim: int) -> None:
    """Validate vector dimension is positive.

    Args:
        dim: Dimension to validate

    Raises:
        ValueError: If dim <= 0
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")


def validate_vector(
    vector: Any,
    expected_dim: Optional[int] = None,
    name: str = "vector",
) -> np.ndarray:
    """Validate and convert a vector to float32 numpy array.

    Args:
        vector: Input vector (list, numpy array, etc.)
        expected_dim: Expected dimension (None to skip check)
        name: Name for error messages

    Returns:
        Validated float32 numpy array

    Raises:
        ValueError: If vector is empty or wrong dimension
    """
    if vector is None:
        raise ValueError(f"{name} cannot be None")

    vec = np.asarray(vector, dtype=np.float32)

    if vec.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if vec.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {vec.ndim}D")

    if expected_dim is not None and len(vec) != expected_dim:
        raise ValueError(
            f"{name} dimension {len(vec)} != expected {expected_dim}"
        )

    return vec


def validate_query(
    query: Any,
    expected_dim: Optional[int] = None,
) -> np.ndarray:
    """Validate and convert a query vector to float32 numpy array.

    Convenience wrapper around validate_vector with name="query".
    """
    return validate_vector(query, expected_dim=expected_dim, name="query")


def ensure_float32(array: np.ndarray) -> np.ndarray:
    """Ensure array is float32, converting if necessary.

    Args:
        array: Input numpy array

    Returns:
        float32 numpy array (may be a copy or the original)
    """
    if array.dtype != np.float32:
        return array.astype(np.float32)
    return array


def validate_embedding_matrix(
    embeddings: np.ndarray,
    name: str = "embeddings",
) -> np.ndarray:
    """Validate a 2D embedding matrix.

    Args:
        embeddings: Input matrix
        name: Name for error messages

    Returns:
        Validated float32 2D numpy array

    Raises:
        ValueError: If matrix is empty or not 2D
    """
    if embeddings.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {embeddings.ndim}D")
    if embeddings.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    return ensure_float32(embeddings)


_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def sanitize_table_name(name: str) -> str:
    """Validate and return a safe SQL table name.

    Only allows alphanumeric characters and underscores,
    starting with a letter or underscore.

    Args:
        name: Table name to validate

    Returns:
        The validated table name (unchanged)

    Raises:
        ValueError: If name contains invalid characters
    """
    if not _TABLE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid table name: {name!r}. "
            "Only alphanumeric and underscore allowed, "
            "must start with letter or underscore."
        )
    return name
