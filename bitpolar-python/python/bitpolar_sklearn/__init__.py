"""BitPolar scikit-learn Integration — sklearn-compatible transformers.

Provides TransformerMixin-compatible classes for compressing feature matrices
and performing compressed nearest-neighbor search in sklearn pipelines.

Usage:
    >>> from bitpolar_sklearn import BitPolarTransformer
    >>> transformer = BitPolarTransformer(bits=4)
    >>> compressed = transformer.fit_transform(X)
"""

from bitpolar_sklearn.transformer import (
    BitPolarSearchTransformer,
    BitPolarTransformer,
)

__all__ = [
    "BitPolarTransformer",
    "BitPolarSearchTransformer",
]
__version__ = "0.3.1"
