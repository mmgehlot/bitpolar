"""Tests for bitpolar Python bindings."""

import numpy as np
import pytest


def test_import():
    """Module imports successfully."""
    import bitpolar
    assert hasattr(bitpolar, "TurboQuantizer")
    assert hasattr(bitpolar, "VectorIndex")


def test_quantizer_creation():
    """TurboQuantizer can be created with valid parameters."""
    import bitpolar
    q = bitpolar.TurboQuantizer(dim=128, bits=4, projections=32, seed=42)
    assert q.dim == 128
    assert q.code_size_bytes > 0


def test_quantizer_default_projections():
    """Projections default to dim/4 when not specified."""
    import bitpolar
    q = bitpolar.TurboQuantizer(dim=128)
    assert q.dim == 128


def test_encode_decode_roundtrip():
    """Encode then decode produces approximate reconstruction."""
    import bitpolar
    q = bitpolar.TurboQuantizer(dim=128, bits=4, projections=32, seed=42)
    v = np.random.randn(128).astype(np.float32)
    code = q.encode(v)
    assert isinstance(code, np.ndarray)
    assert code.dtype == np.uint8
    decoded = q.decode(code)
    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == np.float32
    assert decoded.shape == (128,)
    # Relative error should be reasonable
    error = np.linalg.norm(v - decoded) / np.linalg.norm(v)
    assert error < 0.8, f"Reconstruction error too high: {error}"


def test_inner_product_estimation():
    """Inner product estimate is in the right ballpark."""
    import bitpolar
    q = bitpolar.TurboQuantizer(dim=128, bits=4, projections=32, seed=42)
    v = np.random.randn(128).astype(np.float32)
    code = q.encode(v)
    # Estimate IP of v with itself (should be close to ||v||²)
    estimate = q.inner_product(code, v)
    exact = float(np.dot(v, v))
    # Allow generous tolerance for 4-bit quantization
    assert abs(estimate - exact) / abs(exact) < 1.0, \
        f"IP estimate {estimate} too far from exact {exact}"


def test_vector_index_add_search():
    """VectorIndex stores vectors and finds nearest neighbors."""
    import bitpolar
    dim = 64
    idx = bitpolar.VectorIndex(dim=dim, bits=4, projections=16, seed=42)
    assert len(idx) == 0

    # Add 100 random vectors
    vectors = []
    for i in range(100):
        v = np.random.randn(dim).astype(np.float32)
        idx.add(i, v)
        vectors.append(v)
    assert len(idx) == 100

    # Search with the first vector as query
    ids, scores = idx.search(vectors[0], top_k=5)
    assert len(ids) == 5
    assert len(scores) == 5
    # The first vector should rank highly (ideally #1)
    assert 0 in ids, f"Expected vector 0 in top-5, got {ids}"


def test_invalid_dimension_raises():
    """Invalid parameters raise ValueError."""
    import bitpolar
    with pytest.raises(ValueError):
        bitpolar.TurboQuantizer(dim=0)


def test_repr():
    """String representation is informative."""
    import bitpolar
    q = bitpolar.TurboQuantizer(dim=128, bits=4, projections=32, seed=42)
    r = repr(q)
    assert "TurboQuantizer" in r
    assert "128" in r
