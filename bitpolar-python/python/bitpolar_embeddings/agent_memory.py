"""Agent memory backend using BitPolar compression.

Provides compressed episodic memory for AI agents. Each memory is a
text+embedding pair stored with BitPolar compression, enabling 4-8x
more memories in the same RAM budget.

Works standalone or as a LangChain memory backend.

Usage:
    >>> from bitpolar_embeddings.agent_memory import CompressedMemoryStore
    >>>
    >>> memory = CompressedMemoryStore(dim=384, bits=4, max_memories=10000)
    >>> memory.add("The user's name is Alice", embedding_vector)
    >>> relevant = memory.recall(query_vector, top_k=5)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


@dataclass
class Memory:
    """A single memory entry with text, compressed embedding, and metadata."""
    text: str
    code: np.ndarray  # compressed embedding (uint8)
    timestamp: float
    metadata: dict = field(default_factory=dict)
    access_count: int = 0


class CompressedMemoryStore:
    """Compressed episodic memory for AI agents.

    Stores text memories with BitPolar-compressed embeddings for efficient
    similarity search. Supports time-decay weighting, access frequency
    tracking, and configurable memory limits with LRU eviction.

    Args:
        dim: Embedding dimension
        bits: Quantization precision (3-8)
        max_memories: Maximum number of memories before eviction
        seed: Random seed for deterministic compression
        decay_factor: Time decay factor (0 = no decay, 1 = strong decay)
    """

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        max_memories: int = 10000,
        seed: int = 42,
        decay_factor: float = 0.0,
    ):
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._bits = bits
        self._max = max_memories
        self._decay = decay_factor

        proj = max(dim // 4, 1)
        self._quantizer = _bp.TurboQuantizer(
            dim=dim, bits=bits, projections=proj, seed=seed
        )
        self._memories: list[Memory] = []

    def add(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> int:
        """Add a memory with its embedding.

        Args:
            text: The memory text content
            embedding: float32 embedding vector
            metadata: Optional metadata dict

        Returns:
            Index of the stored memory
        """
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)

        code = self._quantizer.encode(embedding)
        mem = Memory(
            text=text,
            code=code,
            timestamp=time.time(),
            metadata=metadata or {},
            access_count=1,
        )

        # Evict least-accessed before adding to stay within capacity
        while len(self._memories) >= self._max:
            self._memories.sort(key=lambda m: m.access_count)
            self._memories.pop(0)

        self._memories.append(mem)
        return len(self._memories) - 1

    def recall(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        time_weight: bool = True,
    ) -> list[dict]:
        """Recall memories most relevant to a query.

        Args:
            query_embedding: float32 query vector
            top_k: Number of memories to return
            time_weight: Whether to apply time-decay weighting

        Returns:
            List of dicts with 'text', 'score', 'timestamp', 'metadata'
        """
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        if len(query_embedding) != self._dim:
            raise ValueError(f"Query dimension {len(query_embedding)} != configured dim {self._dim}")

        now = time.time()
        scored: list[tuple[int, float]] = []

        for i, mem in enumerate(self._memories):
            # Approximate inner product on compressed embedding
            score = self._quantizer.inner_product(mem.code, query_embedding)

            # Optional time decay: recent memories score higher
            if time_weight and self._decay > 0:
                age_hours = (now - mem.timestamp) / 3600
                decay = np.exp(-self._decay * age_hours)
                score *= decay

            scored.append((i, score))

        # Sort by score descending, take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        results = []
        for idx, score in scored:
            mem = self._memories[idx]
            mem.access_count += 1  # Track access frequency
            results.append({
                "text": mem.text,
                "score": float(score),
                "timestamp": mem.timestamp,
                "metadata": mem.metadata,
                "access_count": mem.access_count,
            })

        return results

    def forget(self, index: int) -> bool:
        """Remove a specific memory by index."""
        if 0 <= index < len(self._memories):
            self._memories.pop(index)
            return True
        return False

    def clear(self):
        """Remove all memories."""
        self._memories.clear()

    @property
    def size(self) -> int:
        """Number of stored memories."""
        return len(self._memories)

    @property
    def memory_bytes(self) -> int:
        """Total bytes used by compressed memories."""
        return sum(len(m.code) for m in self._memories)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"CompressedMemoryStore(size={self.size}, dim={self._dim}, "
            f"bits={self._bits}, bytes={self.memory_bytes:,})"
        )
