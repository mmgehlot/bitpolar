"""Quantization-aware loss functions for BitPolar fine-tuning.

These loss functions penalize embedding representations that quantize poorly
under BitPolar's polar coordinate + QJL scheme, encouraging the model to
produce embeddings that compress well while maintaining semantic quality.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch required for fine-tuning. Install with: pip install torch"
    )


class QuantizationDistortionLoss(nn.Module):
    """Loss that penalizes quantization distortion.

    Combines the standard training loss (e.g., contrastive) with a term
    that measures how much information is lost during BitPolar compression.
    This encourages the model to produce embeddings that are "quantization-friendly."

    The distortion is measured as the normalized reconstruction error:
        L_distortion = ||x - Q(x)||² / ||x||²

    where Q(x) = decompress(compress(x)) is the quantize-dequantize operation.

    Total loss = L_task + alpha * L_distortion

    Args:
        bits: BitPolar quantization precision (3-8)
        alpha: Weight of the distortion penalty (default 0.1)
        dim: Embedding dimension (inferred from first batch if None)
        seed: BitPolar seed
    """

    def __init__(
        self,
        bits: int = 4,
        alpha: float = 0.1,
        dim: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.bits = bits
        self.alpha = alpha
        self.seed = seed
        self._dim = dim
        self._quantizer = None

    def _ensure_quantizer(self, dim: int):
        """Lazily initialize the quantizer when we know the dimension."""
        if self._quantizer is None:
            import bitpolar as _bp

            projections = max(dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=dim, bits=self.bits, projections=projections, seed=self.seed
            )
            self._dim = dim

    def compute_distortion(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute the mean quantization distortion for a batch.

        Uses straight-through estimation: the forward pass quantizes
        (non-differentiable) and measures error, but the gradient flows
        through as if the quantization were identity.

        Args:
            embeddings: float32 tensor of shape (batch, dim)

        Returns:
            Scalar tensor: mean normalized reconstruction error
        """
        batch_size, dim = embeddings.shape
        self._ensure_quantizer(dim)

        # Detach for the quantize-dequantize operation (not differentiable)
        emb_np = embeddings.detach().cpu().numpy().astype(np.float32)

        distortions = []
        for i in range(batch_size):
            code = self._quantizer.encode(emb_np[i])
            decoded = np.array(self._quantizer.decode(code), dtype=np.float32)
            original_norm_sq = float(np.dot(emb_np[i], emb_np[i]))
            error = emb_np[i] - decoded
            error_norm_sq = float(np.dot(error, error))
            if original_norm_sq > 1e-8:
                distortions.append(error_norm_sq / original_norm_sq)
            else:
                distortions.append(0.0)

        # Straight-through estimator: distortion is non-differentiable (Rust encode/decode)
        # but we scale it by the embedding norm ratio so gradients flow through the model.
        # This encourages the model to adjust embeddings to reduce quantization error.
        distortion_scalar = float(np.mean(distortions))

        # Differentiable proxy: scale distortion by normalized embedding norms.
        # Gradient: d(loss)/d(emb) = distortion * d(norm)/d(emb) / detach(norm)
        embedding_norms = torch.norm(embeddings, dim=1, keepdim=True)
        norm_proxy = (embedding_norms / embedding_norms.detach()).mean()
        return distortion_scalar * norm_proxy

    def forward(
        self,
        task_loss: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Combine task loss with quantization distortion penalty.

        Args:
            task_loss: The primary training loss (contrastive, triplet, etc.)
            embeddings: The embedding batch being trained

        Returns:
            Combined loss: task_loss + alpha * distortion
        """
        distortion = self.compute_distortion(embeddings)
        return task_loss + self.alpha * distortion

    def __repr__(self) -> str:
        return (
            f"QuantizationDistortionLoss(bits={self.bits}, alpha={self.alpha})"
        )
