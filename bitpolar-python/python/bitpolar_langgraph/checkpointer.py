"""LangGraph compressed memory checkpointer using BitPolar quantization.

Provides a checkpoint saver that automatically compresses embedding-like
float arrays in checkpoint state, reducing memory footprint for long-running
graph executions with large embedding state.
"""

from __future__ import annotations

import base64
import copy
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False

    class BaseCheckpointSaver:  # type: ignore[no-redef]
        """Stub when langgraph is not installed."""

        pass

    CheckpointTuple = Any  # type: ignore[misc,assignment]
    Checkpoint = Dict[str, Any]  # type: ignore[misc,assignment]
    CheckpointMetadata = Dict[str, Any]  # type: ignore[misc,assignment]


# Threshold for treating a list/array as an embedding (min length of float array)
_EMBEDDING_MIN_DIM = 32


class BitPolarCheckpointer(BaseCheckpointSaver):
    """LangGraph checkpoint saver with BitPolar embedding compression.

    Automatically detects and compresses embedding-like float arrays in
    checkpoint state before storing them in memory. Decompression is
    transparent on retrieval.

    Args:
        bits: Quantization precision (3-8). Default 4 for balanced
            compression/accuracy.
        seed: Random seed for reproducible quantization.
    """

    def __init__(self, bits: int = 4, seed: int = 42) -> None:
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        if _HAS_LANGGRAPH:
            super().__init__()
        self._bits = bits
        self._seed = seed
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._quantizer_dim: Optional[int] = None
        self._storage: dict[Tuple[str, str], dict[str, Any]] = {}
        self._writes_storage: dict[Tuple[str, str, str], list[tuple[str, Any]]] = {}

    def _ensure_quantizer(self, dim: int) -> _bp.TurboQuantizer:
        """Lazily initialize or reinitialize the quantizer for the given dimension.

        Args:
            dim: Vector dimension.

        Returns:
            The initialized TurboQuantizer instance.
        """
        if self._quantizer is None or self._quantizer_dim != dim:
            proj = max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self._bits, projections=proj, seed=self._seed
            )
            self._quantizer_dim = dim
        return self._quantizer

    def _is_embedding_like(self, value: Any) -> bool:
        """Check if a value looks like a float embedding array.

        Args:
            value: Any Python value from checkpoint state.

        Returns:
            True if the value is an embedding-like float array.
        """
        if isinstance(value, np.ndarray):
            return (
                value.ndim == 1
                and value.dtype in (np.float32, np.float64)
                and len(value) >= _EMBEDDING_MIN_DIM
            )
        if isinstance(value, (list, tuple)) and len(value) >= _EMBEDDING_MIN_DIM:
            return all(isinstance(v, (int, float)) for v in value)
        return False

    def _compress_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compress embedding-like values in a checkpoint state dict.

        Args:
            state: The checkpoint state dictionary.

        Returns:
            A new dict with embeddings replaced by compressed representations.
        """
        compressed = {}
        for key, value in state.items():
            if self._is_embedding_like(value):
                vec = np.asarray(value, dtype=np.float32)
                quantizer = self._ensure_quantizer(len(vec))
                code = quantizer.encode(vec)
                code_b64 = base64.b64encode(bytes(code)).decode("ascii")
                compressed[key] = {
                    "__bp_compressed__": True,
                    "code": code_b64,
                    "dim": len(vec),
                    "bits": self._bits,
                }
            elif isinstance(value, dict):
                compressed[key] = self._compress_state(value)
            else:
                compressed[key] = copy.deepcopy(value)
        return compressed

    def _decompress_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Decompress BitPolar-compressed values back to numpy arrays.

        Args:
            state: The compressed checkpoint state dictionary.

        Returns:
            A new dict with compressed values restored as numpy arrays.
        """
        decompressed = {}
        for key, value in state.items():
            if isinstance(value, dict) and value.get("__bp_compressed__"):
                code = np.frombuffer(
                    base64.b64decode(value["code"]), dtype=np.uint8
                ).copy()
                dim = value["dim"]
                quantizer = self._ensure_quantizer(dim)
                reconstructed = quantizer.decode(code)
                decompressed[key] = reconstructed
            elif isinstance(value, dict):
                decompressed[key] = self._decompress_state(value)
            else:
                decompressed[key] = copy.deepcopy(value)
        return decompressed

    def _get_thread_checkpoint_id(
        self, config: dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        """Extract thread_id and checkpoint_id from a LangGraph config.

        Args:
            config: LangGraph configuration dict.

        Returns:
            Tuple of (thread_id, checkpoint_id or None).
        """
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "default")
        checkpoint_id = configurable.get("checkpoint_id")
        return thread_id, checkpoint_id

    def get_tuple(self, config: dict[str, Any]) -> Optional[CheckpointTuple]:
        """Retrieve a checkpoint by config.

        Looks up the checkpoint by (thread_id, checkpoint_id). If no
        checkpoint_id is specified, returns the latest checkpoint for
        the thread.

        Args:
            config: LangGraph config with configurable.thread_id and
                optional configurable.checkpoint_id.

        Returns:
            CheckpointTuple with decompressed state, or None if not found.
        """
        thread_id, checkpoint_id = self._get_thread_checkpoint_id(config)

        if checkpoint_id is not None:
            key = (thread_id, checkpoint_id)
            entry = self._storage.get(key)
        else:
            # Find the latest checkpoint for this thread
            thread_entries = [
                (k, v) for k, v in self._storage.items() if k[0] == thread_id
            ]
            if not thread_entries:
                return None
            key, entry = max(thread_entries, key=lambda x: x[1].get("ts", 0))

        if entry is None:
            return None

        checkpoint = self._decompress_state(entry["checkpoint"])
        metadata = entry.get("metadata", {})

        result_config = {
            "configurable": {
                "thread_id": key[0],
                "checkpoint_id": key[1],
            }
        }

        parent_config = None
        if entry.get("parent_checkpoint_id"):
            parent_config = {
                "configurable": {
                    "thread_id": key[0],
                    "checkpoint_id": entry["parent_checkpoint_id"],
                }
            }

        if _HAS_LANGGRAPH:
            return CheckpointTuple(
                config=result_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
            )

        return {
            "config": result_config,
            "checkpoint": checkpoint,
            "metadata": metadata,
            "parent_config": parent_config,
        }

    def list(
        self,
        config: dict[str, Any],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread.

        Args:
            config: LangGraph config with configurable.thread_id.
            filter: Optional metadata filter dict.
            before: Optional config to list checkpoints before.
            limit: Maximum number of checkpoints to return.

        Yields:
            CheckpointTuple instances in reverse chronological order.
        """
        thread_id, _ = self._get_thread_checkpoint_id(config)

        thread_entries = [
            (k, v) for k, v in self._storage.items() if k[0] == thread_id
        ]
        thread_entries.sort(key=lambda x: x[1].get("ts", 0), reverse=True)

        if before is not None:
            before_ts = None
            before_id = before.get("configurable", {}).get("checkpoint_id")
            if before_id:
                before_key = (thread_id, before_id)
                before_entry = self._storage.get(before_key)
                if before_entry:
                    before_ts = before_entry.get("ts", 0)
            if before_ts is not None:
                thread_entries = [
                    (k, v) for k, v in thread_entries if v.get("ts", 0) < before_ts
                ]

        if filter:
            filtered = []
            for k, v in thread_entries:
                meta = v.get("metadata", {})
                if all(meta.get(fk) == fv for fk, fv in filter.items()):
                    filtered.append((k, v))
            thread_entries = filtered

        count = 0
        for key, entry in thread_entries:
            if limit is not None and count >= limit:
                break

            checkpoint = self._decompress_state(entry["checkpoint"])
            metadata = entry.get("metadata", {})
            result_config = {
                "configurable": {
                    "thread_id": key[0],
                    "checkpoint_id": key[1],
                }
            }

            parent_config = None
            if entry.get("parent_checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": key[0],
                        "checkpoint_id": entry["parent_checkpoint_id"],
                    }
                }

            if _HAS_LANGGRAPH:
                yield CheckpointTuple(
                    config=result_config,
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                )
            else:
                yield {
                    "config": result_config,
                    "checkpoint": checkpoint,
                    "metadata": metadata,
                    "parent_config": parent_config,
                }
            count += 1

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Save a checkpoint with compressed embeddings.

        Detects embedding-like float arrays in the checkpoint state and
        compresses them using BitPolar before storing.

        Args:
            config: LangGraph config with configurable.thread_id.
            checkpoint: The checkpoint state dict to store.
            metadata: Checkpoint metadata.
            new_versions: Optional channel version updates.

        Returns:
            Updated config with the new checkpoint_id.
        """
        thread_id, parent_checkpoint_id = self._get_thread_checkpoint_id(config)
        checkpoint_id = checkpoint.get("id", str(uuid.uuid4()))

        compressed = self._compress_state(checkpoint)

        key = (thread_id, checkpoint_id)
        self._storage[key] = {
            "checkpoint": compressed,
            "metadata": metadata or {},
            "parent_checkpoint_id": parent_checkpoint_id,
            "new_versions": new_versions or {},
            "ts": time.time_ns(),
        }

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Save intermediate writes for a checkpoint.

        Compresses any embedding-like values in the writes before storing.

        Args:
            config: LangGraph config with configurable.thread_id and
                configurable.checkpoint_id.
            writes: Sequence of (channel, value) tuples to store.
            task_id: The task ID these writes belong to.
        """
        thread_id, checkpoint_id = self._get_thread_checkpoint_id(config)
        if checkpoint_id is None:
            checkpoint_id = "pending"

        key = (thread_id, checkpoint_id, task_id)

        compressed_writes: list[tuple[str, Any]] = []
        for channel, value in writes:
            if isinstance(value, dict):
                compressed_writes.append((channel, self._compress_state(value)))
            elif self._is_embedding_like(value):
                vec = np.asarray(value, dtype=np.float32)
                quantizer = self._ensure_quantizer(len(vec))
                code = quantizer.encode(vec)
                code_b64 = base64.b64encode(bytes(code)).decode("ascii")
                compressed_writes.append(
                    (
                        channel,
                        {
                            "__bp_compressed__": True,
                            "code": code_b64,
                            "dim": len(vec),
                            "bits": self._bits,
                        },
                    )
                )
            else:
                compressed_writes.append((channel, copy.deepcopy(value)))

        if key in self._writes_storage:
            self._writes_storage[key].extend(compressed_writes)
        else:
            self._writes_storage[key] = compressed_writes
