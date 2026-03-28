"""BitPolar Quantization-Aware Fine-Tuning SDK.

Train embedding models that are optimized for BitPolar's quantization grid,
improving recall by 2-5% compared to unoptimized models.

Usage:
    >>> from bitpolar_finetune import QuantizationAwareTrainer
    >>>
    >>> trainer = QuantizationAwareTrainer(
    ...     model="sentence-transformers/all-MiniLM-L6-v2",
    ...     bits=4,
    ...     epochs=3,
    ... )
    >>> trainer.fit(training_pairs)
    >>> trainer.save("models/bitpolar-optimized-minilm")
"""

from bitpolar_finetune.trainer import QuantizationAwareTrainer
from bitpolar_finetune.loss import QuantizationDistortionLoss

__all__ = ["QuantizationAwareTrainer", "QuantizationDistortionLoss"]
__version__ = "0.3.3"
