"""Shared utilities for BitPolar integrations.

Provides common validation, initialization, and helper functions
used across all 42+ integration packages to reduce duplication.
"""

from bitpolar_common.validation import (
    validate_bits,
    validate_dim,
    validate_vector,
    validate_query,
    ensure_float32,
    sanitize_table_name,
)

__all__ = [
    "validate_bits",
    "validate_dim",
    "validate_vector",
    "validate_query",
    "ensure_float32",
    "sanitize_table_name",
]
