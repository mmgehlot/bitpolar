//! Node.js bindings for BitPolar vector quantization via NAPI-RS.
//!
//! Provides `TurboQuantizer` and `VectorIndex` classes for JavaScript/TypeScript.
//!
//! Usage:
//! ```javascript
//! const { TurboQuantizer, VectorIndex } = require('bitpolar');
//!
//! const q = new TurboQuantizer(128, 4, 32, 42);
//! const code = q.encode(new Float32Array(128).fill(0.1));
//! const decoded = q.decode(code);
//! ```

use napi::bindgen_prelude::*;
use napi_derive::napi;

use bitpolar::traits::{SerializableCode, VectorQuantizer};

/// TurboQuantizer — two-stage vector quantization (Polar + QJL).
/// No training needed. Deterministic from (dim, bits, projections, seed).
#[napi]
pub struct TurboQuantizer {
    inner: bitpolar::TurboQuantizer,
    cached_code_size: usize,
}

#[napi]
impl TurboQuantizer {
    /// Create a new quantizer.
    /// @param dim - Vector dimension
    /// @param bits - Quantization precision (3-8)
    /// @param projections - QJL projections (typically dim/4)
    /// @param seed - Random seed for deterministic behavior
    #[napi(constructor)]
    pub fn new(dim: u32, bits: u32, projections: u32, seed: i64) -> Result<Self> {
        let inner = bitpolar::TurboQuantizer::new(
            dim as usize,
            bits as u8,
            projections as usize,
            seed as u64,
        )
        .map_err(|e| Error::from_reason(e.to_string()))?;

        // Cache code size
        let dummy = vec![0.0_f32; dim as usize];
        let cached_code_size = inner
            .encode(&dummy)
            .map(|c| c.to_compact_bytes().len())
            .unwrap_or(0);

        Ok(Self {
            inner,
            cached_code_size,
        })
    }

    /// Encode a vector to compressed bytes.
    /// @param vector - Float32Array of length `dim`
    /// @returns Uint8Array of compressed code
    #[napi]
    pub fn encode(&self, vector: Float32Array) -> Result<Uint8Array> {
        let code = self
            .inner
            .encode(vector.as_ref())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let bytes = code.to_compact_bytes();
        Ok(Uint8Array::new(bytes))
    }

    /// Decode a compressed code back to an approximate vector.
    /// @param code - Uint8Array from encode()
    /// @returns Float32Array of length `dim`
    #[napi]
    pub fn decode(&self, code: Uint8Array) -> Result<Float32Array> {
        let turbo_code = bitpolar::TurboCode::from_compact_bytes(code.as_ref())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let decoded = self.inner.decode(&turbo_code);
        Ok(Float32Array::new(decoded))
    }

    /// Estimate inner product between a compressed code and a query.
    /// @param code - Uint8Array from encode()
    /// @param query - Float32Array query vector
    /// @returns Approximate inner product score
    #[napi]
    pub fn inner_product(&self, code: Uint8Array, query: Float32Array) -> Result<f64> {
        let turbo_code = bitpolar::TurboCode::from_compact_bytes(code.as_ref())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let score = self
            .inner
            .inner_product_estimate(&turbo_code, query.as_ref())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(score as f64)
    }

    /// Vector dimension.
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.inner.dim() as u32
    }

    /// Compressed code size in bytes.
    #[napi(getter)]
    pub fn code_size_bytes(&self) -> u32 {
        self.cached_code_size as u32
    }
}

/// VectorIndex — in-memory compressed vector search index.
#[napi]
pub struct VectorIndex {
    quantizer: bitpolar::TurboQuantizer,
    codes: Vec<Vec<u8>>,
    ids: Vec<u32>,
}

#[napi]
impl VectorIndex {
    /// Create a new vector index.
    #[napi(constructor)]
    pub fn new(dim: u32, bits: u32, projections: u32, seed: i64) -> Result<Self> {
        let quantizer = bitpolar::TurboQuantizer::new(
            dim as usize,
            bits as u8,
            projections as usize,
            seed as u64,
        )
        .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Self {
            quantizer,
            codes: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// Add a vector to the index.
    #[napi]
    pub fn add(&mut self, id: u32, vector: Float32Array) -> Result<()> {
        let code = self
            .quantizer
            .encode(vector.as_ref())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        self.codes.push(code.to_compact_bytes());
        self.ids.push(id);
        Ok(())
    }

    /// Search for top-k nearest vectors by inner product.
    /// @returns Array of IDs sorted by descending score
    #[napi]
    pub fn search(&self, query: Float32Array, top_k: u32) -> Result<Vec<u32>> {
        if query.as_ref().len() != self.quantizer.dim() {
            return Err(Error::from_reason(format!(
                "Query dimension {} != index dimension {}",
                query.as_ref().len(),
                self.quantizer.dim()
            )));
        }
        let mut scored: Vec<(u32, f32)> = Vec::with_capacity(self.codes.len());
        for (i, code_bytes) in self.codes.iter().enumerate() {
            let code = bitpolar::TurboCode::from_compact_bytes(code_bytes)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            let score = self
                .quantizer
                .inner_product_estimate(&code, query.as_ref())
                .map_err(|e| Error::from_reason(e.to_string()))?;
            scored.push((self.ids[i], score));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k as usize);
        Ok(scored.into_iter().map(|(id, _)| id).collect())
    }

    /// Number of indexed vectors.
    #[napi(getter)]
    pub fn len(&self) -> u32 {
        self.codes.len() as u32
    }
}
