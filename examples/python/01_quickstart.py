"""BitPolar Python Quick Start — compress and search vectors in 10 lines.

Prerequisites:
    pip install bitpolar numpy

Usage:
    python examples/python/01_quickstart.py
"""

import numpy as np
import bitpolar

# Create quantizer — no training needed, just 4 integers
q = bitpolar.TurboQuantizer(dim=128, bits=4, projections=32, seed=42)
print(f"Quantizer: dim={q.dim}, code_size={q.code_size_bytes}B")

# Encode a vector
vector = np.random.randn(128).astype(np.float32)
code = q.encode(vector)
print(f"Encoded: {len(vector)*4}B → {len(code)}B ({len(vector)*4/len(code):.1f}x compression)")

# Decode back to approximate vector
decoded = q.decode(code)
error = np.linalg.norm(vector - decoded) / np.linalg.norm(vector)
print(f"Reconstruction error: {error:.4f}")

# Estimate inner product without decompressing
score = q.inner_product(code, vector)
exact = float(np.dot(vector, vector))
print(f"Inner product: estimated={score:.4f}, exact={exact:.4f}")
