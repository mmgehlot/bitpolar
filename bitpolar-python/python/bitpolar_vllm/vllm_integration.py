"""vLLM integration — registers BitPolar as a quantization backend.

This module hooks into vLLM's quantization registration system.
If vLLM is not installed, this module is a no-op.

Usage:
    Simply importing bitpolar_vllm triggers registration:

    >>> import bitpolar_vllm  # auto-registers
    >>> from vllm import LLM
    >>> llm = LLM("meta-llama/Llama-3-8B", quantization="bitpolar")

For manual registration:
    >>> from bitpolar_vllm.vllm_integration import register_bitpolar_quantization
    >>> register_bitpolar_quantization()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

import threading

_LOCK = threading.Lock()
_REGISTERED = False


def register_bitpolar_quantization() -> bool:
    """Register BitPolar as a vLLM quantization method.

    Returns True if registration succeeded, False if vLLM is not available.
    Safe to call multiple times (idempotent).
    """
    global _REGISTERED
    with _LOCK:
        if _REGISTERED:
            return True
        return _register_impl()


def _register_impl() -> bool:
    """Internal registration (called under lock)."""
    global _REGISTERED
    try:
        from vllm.model_executor.layers.quantization import (
            QUANTIZATION_METHODS,
        )
    except ImportError:
        logger.debug(
            "vLLM not installed — BitPolar quantization registration skipped"
        )
        return False

    # Register our config class under the name "bitpolar"
    if "bitpolar" not in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS["bitpolar"] = _get_config_class()
        logger.info("Registered BitPolar quantization with vLLM")

    _REGISTERED = True
    return True


def _get_config_class():
    """Lazily create the config class to avoid import errors when vLLM
    is not installed."""
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )
    import torch

    class BitPolarQuantConfig(QuantizationConfig):
        """BitPolar KV cache quantization configuration for vLLM.

        Compresses Key-Value caches to 3-8 bits using near-optimal
        polar coordinate quantization + QJL residual correction.
        """

        def __init__(self, bits: int = 4, seed: int = 42):
            self.bits = bits
            self.seed = seed

        def get_name(self) -> str:
            return "bitpolar"

        def get_supported_act_dtypes(self) -> list:
            return [torch.float16, torch.bfloat16, torch.float32]

        def get_min_capability(self) -> int:
            # Minimum CUDA compute capability (Ampere+)
            return 80

        def get_quant_method(self, layer, prefix: str):
            """Return quantization method for the given layer.

            Only applies to attention layers for KV cache compression.
            Other layers pass through unchanged.
            """
            # Import here to avoid circular dependency
            from torch.nn import Module

            layer_name = type(layer).__name__
            if "Attention" in layer_name:
                return BitPolarKVMethod(self)
            return None  # No quantization for non-attention layers

        @classmethod
        def from_config(cls, config: dict) -> "BitPolarQuantConfig":
            return cls(
                bits=config.get("bits", 4),
                seed=config.get("seed", 42),
            )

    class BitPolarKVMethod:
        """Quantization method applied to attention KV caches."""

        def __init__(self, config: BitPolarQuantConfig):
            self.config = config
            self._quantizer = None

        def _ensure_quantizer(self, head_dim: int, num_heads: int):
            if self._quantizer is None:
                from bitpolar_vllm.quantizer import KVCacheQuantizer

                self._quantizer = KVCacheQuantizer(
                    head_dim=head_dim,
                    bits=self.config.bits,
                    num_heads=num_heads,
                    seed=self.config.seed,
                )

        def create_weights(self, layer):
            """Called during model loading to set up quantization state."""
            pass  # BitPolar is data-oblivious — no pre-trained weights needed

        def apply(self, layer, key, value):
            """Quantize K/V tensors during inference.

            Called by vLLM's attention layer to compress KV entries
            before storing them in the cache.
            """
            import torch
            import numpy as np

            # Get dimensions from the tensor shape
            # key shape: (batch, num_heads, seq_len, head_dim) or similar
            if key.dim() == 4:
                batch, num_heads, seq_len, head_dim = key.shape
            elif key.dim() == 3:
                batch, num_heads, head_dim = key.shape
                seq_len = 1
            else:
                # Unsupported shape — pass through unmodified
                return key, value

            self._ensure_quantizer(head_dim, num_heads)

            # NOTE: Full vLLM integration requires a custom cache manager
            # that stores BitPolar compressed codes instead of float tensors.
            # This is experimental and currently passes through unchanged.
            # For working KV cache compression, use BitPolarDynamicCache
            # from bitpolar_vllm.dynamic_cache instead.
            logger.warning(
                "BitPolar vLLM apply() is experimental — K/V passed through "
                "uncompressed. Use BitPolarDynamicCache for working compression."
            )
            return key, value

    return BitPolarQuantConfig
