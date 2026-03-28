"""BitPolar ML Framework Integrations — PyTorch, JAX, TensorFlow, scikit-learn.

Demonstrates compression patterns for each framework using mock data.
No actual framework imports required — patterns shown with numpy equivalents.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/25_ml_frameworks.py
"""

import numpy as np
import bitpolar

DIM = 256
q = bitpolar.TurboQuantizer(dim=DIM, bits=4, projections=32, seed=42)

# =============================================================================
# PyTorch: Quantize Embedding Layer
# =============================================================================
print("=== PyTorch — Quantize Embedding Table ===\n")

num_embeddings = 50
embeddings = np.random.randn(num_embeddings, DIM).astype(np.float32)

compressed = [q.encode(embeddings[i]) for i in range(num_embeddings)]
orig_bytes = embeddings.nbytes
comp_bytes = sum(len(c) for c in compressed)
print(f"Embedding table: {num_embeddings} x {DIM}")
print(f"Original: {orig_bytes:,}B | Compressed: {comp_bytes:,}B | Ratio: {orig_bytes/comp_bytes:.1f}x")

decoded = q.decode(compressed[0])
error = np.linalg.norm(embeddings[0] - decoded) / np.linalg.norm(embeddings[0])
print(f"Reconstruction error (row 0): {error:.4f}")

# Pattern: torch.nn.Embedding -> BitPolar compressed lookup
# class BitPolarEmbedding(torch.nn.Module):
#     def __init__(self, weight, bits=4):
#         self.q = bitpolar.TurboQuantizer(dim=weight.shape[1], bits=bits)
#         self.codes = [self.q.encode(w.numpy()) for w in weight]
#     def forward(self, indices):
#         return torch.stack([torch.from_numpy(self.q.decode(self.codes[i])) for i in indices])

# =============================================================================
# JAX: Compress/Decompress Arrays
# =============================================================================
print("\n=== JAX — Compress/Decompress Arrays ===\n")

jax_array = np.random.randn(DIM).astype(np.float32)
code = q.encode(jax_array)
restored = q.decode(code)
score = q.inner_product(code, jax_array)
exact = float(np.dot(jax_array, jax_array))
print(f"JAX array: {jax_array.nbytes}B -> {len(code)}B compressed")
print(f"Inner product: estimated={score:.4f}, exact={exact:.4f}")

# Pattern: jax.numpy array -> compress -> store -> decompress -> jax.numpy
# compressed = q.encode(jnp.array(vec).to_py())
# restored = jnp.array(q.decode(compressed))

# =============================================================================
# TensorFlow: BitPolar Layer
# =============================================================================
print("\n=== TensorFlow — BitPolar Compression Layer ===\n")

batch = np.random.randn(8, DIM).astype(np.float32)
codes = [q.encode(batch[i]) for i in range(len(batch))]
decoded_batch = np.array([q.decode(c) for c in codes])
mse = np.mean((batch - decoded_batch) ** 2)
print(f"Batch shape: {batch.shape}, MSE after compress/decompress: {mse:.6f}")

# Pattern: tf.keras.layers.Layer subclass
# class BitPolarLayer(tf.keras.layers.Layer):
#     def call(self, inputs):
#         codes = [self.q.encode(x.numpy()) for x in inputs]
#         return tf.stack([tf.constant(self.q.decode(c)) for c in codes])

# =============================================================================
# scikit-learn: Pipeline with Compressed Features
# =============================================================================
print("\n=== scikit-learn — Pipeline with Compressed Features ===\n")

X = np.random.randn(100, DIM).astype(np.float32)
X_compressed = np.array([q.decode(q.encode(X[i])) for i in range(len(X))])
corr = np.mean([np.dot(X[i], X_compressed[i]) / (np.linalg.norm(X[i]) * np.linalg.norm(X_compressed[i]))
                 for i in range(len(X))])
print(f"Dataset: {len(X)} samples x {DIM} features")
print(f"Mean cosine similarity (original vs compressed): {corr:.4f}")

# Pattern: sklearn.base.TransformerMixin
# class BitPolarTransformer(TransformerMixin):
#     def transform(self, X):
#         return np.array([self.q.decode(self.q.encode(x)) for x in X])
# pipe = Pipeline([("compress", BitPolarTransformer()), ("clf", LogisticRegression())])
