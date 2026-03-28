"""BitPolar DSPy Integration — retriever module for DSPy pipelines.

Usage:
    >>> from bitpolar_dspy import BitPolarRM
    >>> texts = ["passage 1", "passage 2"]
    >>> embeddings = np.random.randn(2, 128).astype(np.float32)
    >>> rm = BitPolarRM(texts=texts, embeddings=embeddings, bits=4)
    >>> result = rm.forward([0.1]*128, k=5)
"""

from bitpolar_dspy.retriever import BitPolarRM

__all__ = ["BitPolarRM"]
__version__ = "0.3.0"
