"""BitPolar LangGraph Integration — compressed memory checkpointer.

Provides a LangGraph-compatible checkpoint saver that compresses
embedding-like float arrays in checkpoint state using BitPolar
quantization for memory-efficient graph execution.

Usage:
    >>> from bitpolar_langgraph import BitPolarCheckpointer
    >>> checkpointer = BitPolarCheckpointer(bits=4)
    >>> graph = StateGraph(...).compile(checkpointer=checkpointer)
"""

from bitpolar_langgraph.checkpointer import BitPolarCheckpointer

__all__ = ["BitPolarCheckpointer"]
__version__ = "0.2.0"
