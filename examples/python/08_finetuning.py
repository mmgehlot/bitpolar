"""BitPolar Quantization-Aware Fine-Tuning.

Fine-tune an embedding model to produce BitPolar-friendly representations,
improving recall by 2-5% compared to unoptimized models.

Prerequisites:
    pip install bitpolar numpy sentence-transformers torch

Usage:
    python examples/python/08_finetuning.py
"""

from bitpolar_finetune import QuantizationAwareTrainer

print("=== BitPolar Quantization-Aware Fine-Tuning ===\n")

try:
    # Create trainer
    trainer = QuantizationAwareTrainer(
        model="sentence-transformers/all-MiniLM-L6-v2",
        bits=4,
        alpha=0.1,      # Distortion penalty weight
        epochs=1,        # Just 1 epoch for demo
        batch_size=16,
    )
    print(f"Trainer: {trainer}")

    # Training data: (sentence_a, sentence_b, similarity_score)
    training_data = [
        ("A dog plays in the park", "A puppy runs in the garden", 0.9),
        ("The weather is sunny", "It's a bright day outside", 0.8),
        ("I love programming", "Coding is my passion", 0.85),
        ("The cat sleeps on the bed", "A kitten rests on the couch", 0.7),
        ("Python is a programming language", "JavaScript runs in browsers", 0.3),
        ("The sky is blue", "Water is transparent", 0.1),
    ] * 10  # Repeat for minimum batch size

    # Fine-tune
    print("\nStarting fine-tuning...")
    metrics = trainer.fit(training_data)
    print(f"Training complete: {metrics}")

    # Evaluate compression quality
    eval_data = training_data[:6]
    eval_metrics = trainer.evaluate(eval_data, k_values=[1, 3])
    print(f"\nEvaluation:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save optimized model
    trainer.save("/tmp/bitpolar-optimized-minilm")
    print(f"\nModel saved to /tmp/bitpolar-optimized-minilm")

except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install sentence-transformers torch")
