//! # bitpolar (Python)
//!
//! Python bindings for BitPolar vector quantization via PyO3.
//!
//! ## Usage
//!
//! ```python
//! import numpy as np
//! import bitpolar
//!
//! # Create a quantizer — no training needed
//! q = bitpolar.TurboQuantizer(dim=128, bits=4, projections=32, seed=42)
//!
//! # Encode a vector
//! v = np.random.randn(128).astype(np.float32)
//! code = q.encode(v)
//!
//! # Decode back to approximate vector
//! decoded = q.decode(code)
//!
//! # Build a search index
//! idx = bitpolar.VectorIndex(dim=128, bits=4, projections=32, seed=42)
//! for i in range(1000):
//!     idx.add(i, np.random.randn(128).astype(np.float32))
//! ids, scores = idx.search(np.random.randn(128).astype(np.float32), top_k=10)
//! ```

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use bitpolar_core::traits::{VectorQuantizer, SerializableCode};

/// Convert a BitPolar error to a Python ValueError.
fn to_py_err(e: bitpolar_core::TurboQuantError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// TurboQuantizer: two-stage vector quantization (Polar + QJL).
///
/// Compresses f32 vectors to compact codes with provably unbiased
/// inner product estimates. No training or calibration data needed.
///
/// Args:
///     dim: Vector dimension (e.g., 128, 768, 1536)
///     bits: Quantization precision (3-8, default 4)
///     projections: QJL projections (typically dim/4)
///     seed: Random seed for deterministic rotation/projection
#[pyclass]
pub struct TurboQuantizer {
    inner: bitpolar_core::TurboQuantizer,
    /// Cached code size in bytes (computed once at construction).
    cached_code_size: usize,
}

#[pymethods]
impl TurboQuantizer {
    /// Create a new TurboQuantizer.
    #[new]
    #[pyo3(signature = (dim, bits=4, projections=None, seed=42))]
    fn new(dim: usize, bits: u8, projections: Option<usize>, seed: u64) -> PyResult<Self> {
        let proj = projections.unwrap_or((dim / 4).max(1));
        let inner = bitpolar_core::TurboQuantizer::new(dim, bits, proj, seed)
            .map_err(to_py_err)?;
        // Cache code size by doing one dummy encode at construction
        let dummy = vec![0.0_f32; dim];
        let cached_code_size = if let Ok(code) = inner.encode(&dummy) {
            code.to_compact_bytes().len()
        } else {
            0
        };
        Ok(TurboQuantizer { inner, cached_code_size })
    }

    /// Encode a vector to compact bytes.
    ///
    /// Args:
    ///     vector: numpy float32 array of length `dim`
    ///
    /// Returns:
    ///     numpy uint8 array of compressed code
    fn encode<'py>(
        &self,
        py: Python<'py>,
        vector: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Py<PyArray1<u8>>> {
        let view = vector.as_slice()?;
        let code = self.inner.encode(view).map_err(to_py_err)?;
        let bytes = code.to_compact_bytes();
        Ok(bytes.into_pyarray(py).unbind())
    }

    /// Decode a compressed code back to an approximate vector.
    ///
    /// Args:
    ///     code: numpy uint8 array (from encode)
    ///
    /// Returns:
    ///     numpy float32 array of length `dim`
    fn decode<'py>(
        &self,
        py: Python<'py>,
        code: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let bytes = code.as_slice()?;
        let turbo_code = bitpolar_core::TurboCode::from_compact_bytes(bytes)
            .map_err(to_py_err)?;
        let decoded = self.inner.decode(&turbo_code);
        Ok(decoded.into_pyarray(py).unbind())
    }

    /// Estimate inner product between a compressed code and a query vector.
    ///
    /// Much faster than decompressing and computing full dot product.
    fn inner_product(
        &self,
        code: PyReadonlyArray1<'_, u8>,
        query: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<f32> {
        let code_bytes = code.as_slice()?;
        let query_slice = query.as_slice()?;
        let turbo_code = bitpolar_core::TurboCode::from_compact_bytes(code_bytes)
            .map_err(to_py_err)?;
        self.inner
            .inner_product_estimate(&turbo_code, query_slice)
            .map_err(to_py_err)
    }

    /// Return the vector dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Return the compressed code size in bytes (cached at construction).
    #[getter]
    fn code_size_bytes(&self) -> usize {
        self.cached_code_size
    }

    fn __repr__(&self) -> String {
        format!(
            "TurboQuantizer(dim={}, code_size={}B)",
            self.inner.dim(),
            self.code_size_bytes()
        )
    }
}

/// VectorIndex: approximate nearest neighbor search using BitPolar compression.
///
/// Stores vectors as compressed codes and searches by estimated inner product.
///
/// Args:
///     dim: Vector dimension
///     bits: Quantization precision (3-8, default 4)
///     projections: QJL projections (default dim/4)
///     seed: Random seed
#[pyclass]
pub struct VectorIndex {
    quantizer: bitpolar_core::TurboQuantizer,
    codes: Vec<Vec<u8>>,
    ids: Vec<u64>,
}

#[pymethods]
impl VectorIndex {
    #[new]
    #[pyo3(signature = (dim, bits=4, projections=None, seed=42))]
    fn new(dim: usize, bits: u8, projections: Option<usize>, seed: u64) -> PyResult<Self> {
        let proj = projections.unwrap_or((dim / 4).max(1));
        let quantizer = bitpolar_core::TurboQuantizer::new(dim, bits, proj, seed)
            .map_err(to_py_err)?;
        Ok(VectorIndex {
            quantizer,
            codes: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// Add a vector to the index.
    fn add(&mut self, id: u64, vector: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let view = vector.as_slice()?;
        let code = self.quantizer.encode(view).map_err(to_py_err)?;
        self.codes.push(code.to_compact_bytes());
        self.ids.push(id);
        Ok(())
    }

    /// Search for top-k nearest vectors by inner product.
    ///
    /// Returns:
    ///     Tuple of (ids: numpy int64 array, scores: numpy float32 array)
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        top_k: usize,
    ) -> PyResult<(Py<PyArray1<u64>>, Py<PyArray1<f32>>)> {
        let query_slice = query.as_slice()?;

        // Score all vectors
        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(self.codes.len());
        for (i, code_bytes) in self.codes.iter().enumerate() {
            let code = bitpolar_core::TurboCode::from_compact_bytes(code_bytes)
                .map_err(to_py_err)?;
            let score = self.quantizer
                .inner_product_estimate(&code, query_slice)
                .map_err(to_py_err)?;
            scored.push((self.ids[i], score));
        }

        // Sort descending by score, take top-k
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let ids: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
        let scores: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();

        Ok((
            ids.into_pyarray(py).unbind(),
            scores.into_pyarray(py).unbind(),
        ))
    }

    fn __len__(&self) -> usize {
        self.codes.len()
    }

    fn __repr__(&self) -> String {
        format!("VectorIndex(size={}, dim={})", self.codes.len(), self.quantizer.dim())
    }
}

/// Python module initialization.
#[pymodule]
fn bitpolar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TurboQuantizer>()?;
    m.add_class::<VectorIndex>()?;
    Ok(())
}
