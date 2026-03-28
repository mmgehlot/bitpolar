"""BitPolar + scikit-learn Pipeline.

Integrates BitPolar as sklearn-compatible transformers for use
in Pipeline, GridSearchCV, and other sklearn workflows.

Prerequisites:
    pip install bitpolar scikit-learn numpy

Usage:
    python examples/python/22_sklearn_pipeline.py
"""

import numpy as np
from bitpolar_sklearn import BitPolarTransformer, BitPolarSearchTransformer

print("=== BitPolar + scikit-learn Pipeline ===\n")

# --- BitPolarTransformer (compress/decompress) -------------------------------
print("--- BitPolarTransformer ---")
transformer = BitPolarTransformer(bits=4, seed=42)

X = np.random.randn(200, 384).astype(np.float32)
transformer.fit(X)  # learns dim, prepares quantizer
X_compressed = transformer.transform(X)
print(f"Input:      {X.shape}, {X.nbytes:,}B")
print(f"Compressed: {X_compressed.shape}, {X_compressed.nbytes:,}B")
print(f"Ratio:      {X.nbytes / X_compressed.nbytes:.1f}x")

# Inverse transform (decompress)
X_reconstructed = transformer.inverse_transform(X_compressed)
error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)
print(f"Reconstruction error: {error:.4f}")

# --- In a sklearn Pipeline ---------------------------------------------------
print("\n--- sklearn Pipeline ---")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("bitpolar", BitPolarTransformer(bits=4, seed=42)),
])

X_train = np.random.randn(500, 128).astype(np.float32)
X_test = np.random.randn(50, 128).astype(np.float32)

pipe.fit(X_train)
X_out = pipe.transform(X_test)
print(f"Pipeline: {X_test.shape} -> scaled -> compressed -> {X_out.shape}")

# --- BitPolarSearchTransformer -----------------------------------------------
print("\n--- BitPolarSearchTransformer ---")
search = BitPolarSearchTransformer(bits=4, seed=42, top_k=5)

corpus = np.random.randn(1000, 128).astype(np.float32)
queries = np.random.randn(3, 128).astype(np.float32)

search.fit(corpus)  # builds compressed index
results = search.transform(queries)  # returns (n_queries, top_k) neighbor IDs
print(f"Corpus: {corpus.shape[0]} vectors")
print(f"Queries: {queries.shape[0]}")
print(f"Results shape: {results.shape}")
for i in range(len(queries)):
    print(f"  Query {i} neighbors: {results[i].tolist()}")

print(f"\nIndex memory: {search.memory_bytes():,}B vs raw: {corpus.nbytes:,}B")
