//! # bitpolar-wasm
//!
//! WebAssembly bindings for BitPolar vector quantization.
//! Enables browser-side vector search with near-optimal compression.
//!
//! ## Usage (JavaScript)
//!
//! ```javascript
//! import init, { WasmQuantizer, WasmVectorIndex } from 'bitpolar-wasm';
//!
//! await init();
//!
//! // Create quantizer
//! const q = new WasmQuantizer(128, 4, 32, 42n);
//!
//! // Encode/decode vectors
//! const vector = new Float32Array(128).fill(0.1);
//! const code = q.encode(vector);
//! const decoded = q.decode(code);
//!
//! // Build and search an index
//! const index = new WasmVectorIndex(128, 4, 32, 42n);
//! index.add(0, vector);
//! const results = index.search(vector, 5);
//! ```

use bitpolar::traits::VectorQuantizer;
use wasm_bindgen::prelude::*;

/// A WASM-compatible wrapper around BitPolar's TurboQuantizer.
///
/// Provides encode, decode, and inner product estimation in the browser.
#[wasm_bindgen]
pub struct WasmQuantizer {
    /// The underlying TurboQuantizer instance.
    inner: bitpolar::TurboQuantizer,
    /// Cached code size in bytes (computed once at construction).
    cached_code_size: usize,
}

#[wasm_bindgen]
impl WasmQuantizer {
    /// Create a new quantizer.
    ///
    /// # Arguments
    /// - `dim` — Vector dimension (e.g., 128, 768, 1536)
    /// - `bits` — Quantization precision (3-8)
    /// - `projections` — Number of QJL projections (typically dim/4)
    /// - `seed` — Random seed for deterministic rotation/projection
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim: usize,
        bits: u8,
        projections: usize,
        seed: u64,
    ) -> Result<WasmQuantizer, JsError> {
        let inner = bitpolar::TurboQuantizer::new(dim, bits, projections, seed)
            .map_err(|e| JsError::new(&e.to_string()))?;
        // Cache code size by doing one dummy encode
        let dummy = vec![0.0_f32; dim];
        let cached_code_size = if let Ok(code) = inner.encode(&dummy) {
            use bitpolar::traits::SerializableCode;
            code.to_compact_bytes().len()
        } else {
            0
        };
        Ok(WasmQuantizer {
            inner,
            cached_code_size,
        })
    }

    /// Encode a vector to compressed bytes.
    ///
    /// Input: Float32Array of length `dim`.
    /// Output: Uint8Array of compressed code.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>, JsError> {
        if vector.is_empty() {
            return Err(JsError::new("vector cannot be empty"));
        }
        let code = self
            .inner
            .encode(vector)
            .map_err(|e| JsError::new(&e.to_string()))?;
        use bitpolar::traits::SerializableCode;
        Ok(code.to_compact_bytes())
    }

    /// Decode a compressed code back to an approximate vector.
    ///
    /// Input: Uint8Array (from encode).
    /// Output: Float32Array of length `dim`.
    pub fn decode(&self, code_bytes: &[u8]) -> Result<Vec<f32>, JsError> {
        if code_bytes.is_empty() {
            return Err(JsError::new("code bytes cannot be empty"));
        }
        use bitpolar::traits::SerializableCode;
        let code = bitpolar::TurboCode::from_compact_bytes(code_bytes)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(self.inner.decode(&code))
    }

    /// Estimate inner product between two compressed codes.
    ///
    /// Much faster than decompressing both vectors and computing dot product.
    pub fn inner_product(&self, code_a: &[u8], code_b_query: &[f32]) -> Result<f32, JsError> {
        use bitpolar::traits::SerializableCode;
        let code = bitpolar::TurboCode::from_compact_bytes(code_a)
            .map_err(|e| JsError::new(&e.to_string()))?;
        self.inner
            .inner_product_estimate(&code, code_b_query)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Return the vector dimension.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Return the compressed code size in bytes.
    pub fn code_size_bytes(&self) -> usize {
        self.cached_code_size
    }
}

/// A WASM-compatible vector search index.
///
/// Stores compressed vectors and supports approximate nearest neighbor search.
#[wasm_bindgen]
pub struct WasmVectorIndex {
    /// The quantizer used for encoding/estimation.
    quantizer: bitpolar::TurboQuantizer,
    /// Stored compressed codes as compact bytes.
    codes: Vec<Vec<u8>>,
    /// Original IDs corresponding to each code.
    ids: Vec<u32>,
}

#[wasm_bindgen]
impl WasmVectorIndex {
    /// Create a new vector index.
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim: usize,
        bits: u8,
        projections: usize,
        seed: u64,
    ) -> Result<WasmVectorIndex, JsError> {
        let quantizer = bitpolar::TurboQuantizer::new(dim, bits, projections, seed)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmVectorIndex {
            quantizer,
            codes: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, id: u32, vector: &[f32]) -> Result<(), JsError> {
        let code = self
            .quantizer
            .encode(vector)
            .map_err(|e| JsError::new(&e.to_string()))?;
        use bitpolar::traits::SerializableCode;
        self.codes.push(code.to_compact_bytes());
        self.ids.push(id);
        Ok(())
    }

    /// Search for the top-k nearest vectors by inner product.
    ///
    /// Returns IDs of the top-k results sorted by descending score.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<u32>, JsError> {
        if query.is_empty() {
            return Err(JsError::new("query cannot be empty"));
        }
        use bitpolar::traits::SerializableCode;

        let mut scored: Vec<(u32, f32)> = Vec::with_capacity(self.codes.len());
        for (i, code_bytes) in self.codes.iter().enumerate() {
            let code = bitpolar::TurboCode::from_compact_bytes(code_bytes)
                .map_err(|e| JsError::new(&e.to_string()))?;
            let score = self
                .quantizer
                .inner_product_estimate(&code, query)
                .map_err(|e| JsError::new(&e.to_string()))?;
            scored.push((self.ids[i], score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored.iter().map(|(id, _)| *id).collect())
    }

    /// Search and return both IDs and scores as flat interleaved arrays.
    ///
    /// Returns `[id0, id1, ...]` as IDs and scores as a separate float list.
    /// On the JS side, call `search()` for IDs-only or use this for ranked results
    /// with scores.
    ///
    /// Returns interleaved `[id0, score_bits0, id1, score_bits1, ...]` where
    /// `score_bits` is the f32 score as u32 bits. Decode on JS side:
    /// ```js
    /// const ids = [], scores = [];
    /// for (let i = 0; i < result.length; i += 2) {
    ///     ids.push(result[i]);
    ///     scores.push(new Float32Array(new Uint32Array([result[i+1]]).buffer)[0]);
    /// }
    /// ```
    pub fn search_with_scores(&self, query: &[f32], top_k: usize) -> Result<Vec<u32>, JsError> {
        use bitpolar::traits::SerializableCode;

        let mut scored: Vec<(u32, f32)> = Vec::with_capacity(self.codes.len());
        for (i, code_bytes) in self.codes.iter().enumerate() {
            let code = bitpolar::TurboCode::from_compact_bytes(code_bytes)
                .map_err(|e| JsError::new(&e.to_string()))?;
            let score = self
                .quantizer
                .inner_product_estimate(&code, query)
                .map_err(|e| JsError::new(&e.to_string()))?;
            scored.push((self.ids[i], score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let mut result = Vec::with_capacity(scored.len() * 2);
        for (id, score) in &scored {
            result.push(*id);
            result.push(score.to_bits());
        }
        Ok(result)
    }

    /// Return the number of indexed vectors.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Return whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}
