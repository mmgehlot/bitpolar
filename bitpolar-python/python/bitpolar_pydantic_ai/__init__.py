"""BitPolar PydanticAI Integration — typed tool functions for PydanticAI agents.

Usage:
    >>> from bitpolar_pydantic_ai import compress, search, add_vector, stats
    >>> from bitpolar_pydantic_ai import CompressInput, SearchInput, AddVectorInput
    >>> result = compress(CompressInput(vector=[0.1]*128, bits=4))
"""

from bitpolar_pydantic_ai.tools import (
    BitPolarToolServer,
    CompressInput,
    SearchInput,
    AddVectorInput,
    compress,
    search,
    add_vector,
    stats,
)

__all__ = [
    "BitPolarToolServer",
    "CompressInput",
    "SearchInput",
    "AddVectorInput",
    "compress",
    "search",
    "add_vector",
    "stats",
]
__version__ = "0.3.3"
