//! Oversampled approximate nearest-neighbor search.
//!
//! `OversampledSearch` stores compressed codes alongside the original f32
//! vectors. At query time it:
//!
//! 1. Computes an approximate inner product score for every stored code
//!    (fast — compressed arithmetic).
//! 2. Selects the top `k * oversample_factor` candidates.
//! 3. Re-ranks those candidates with exact f32 inner products against the
//!    original vectors (slow, but few candidates).
//! 4. Returns the true top-k by exact score.
//!
//! This two-phase strategy recovers recall that would otherwise be lost when
//! quantizing to very low bit widths.
//!
//! # Example
//!
//! ```rust
//! use bitpolar::TurboQuantizer;
//! use bitpolar::search::OversampledSearch;
//!
//! let q = TurboQuantizer::new(16, 4, 8, 42).unwrap();
//! let mut index = OversampledSearch::new(q, 4);
//!
//! for i in 0..20_u32 {
//!     let v: Vec<f32> = (0..16).map(|j| ((i * 16 + j) as f32 * 0.1).sin()).collect();
//!     index.add(v).unwrap();
//! }
//!
//! let query: Vec<f32> = vec![1.0_f32; 16];
//! let results = index.search(&query, 3).unwrap();
//! assert_eq!(results.len(), 3);
//! // Results are (index, score) pairs sorted by descending exact IP.
//! for (idx, score) in &results {
//!     assert!(*idx < 20);
//!     let _ = score;
//! }
//! ```

use crate::error::{Result, TurboQuantError};
use crate::traits::VectorQuantizer;

// ---------------------------------------------------------------------------
// OversampledSearch
// ---------------------------------------------------------------------------

/// Approximate nearest-neighbor index with oversampled re-ranking.
///
/// Generic over any `Q: VectorQuantizer` whose code type is `Clone`.
///
/// # Type Parameters
/// - `Q` — a quantizer implementing [`VectorQuantizer`]
///
/// # Memory
///
/// Stores one compressed code **and** one full f32 vector per indexed item.
/// Use a lower `oversample_factor` (e.g., 2) for memory-constrained settings.
#[derive(Debug)]
pub struct OversampledSearch<Q>
where
    Q: VectorQuantizer,
    Q::Code: Clone,
{
    quantizer: Q,
    codes: Vec<Q::Code>,
    originals: Vec<Vec<f32>>,
    oversample_factor: usize,
}

impl<Q> OversampledSearch<Q>
where
    Q: VectorQuantizer,
    Q::Code: Clone,
{
    /// Create a new empty index.
    ///
    /// # Arguments
    /// - `quantizer` — the quantizer used to compress stored vectors
    /// - `oversample_factor` — how many times more candidates to fetch in the
    ///   approximate phase before exact re-ranking (must be ≥ 1; 4 is a good
    ///   default)
    ///
    /// # Panics
    ///
    /// Panics if `oversample_factor == 0`.
    pub fn new(quantizer: Q, oversample_factor: usize) -> Self {
        assert!(oversample_factor > 0, "oversample_factor must be >= 1");
        Self {
            quantizer,
            codes: Vec::new(),
            originals: Vec::new(),
            oversample_factor,
        }
    }

    /// Number of vectors currently indexed.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Returns `true` if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Add a vector to the index and return its 0-based index.
    ///
    /// The vector is compressed with the quantizer; the original f32 values are
    /// also retained for exact re-ranking.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
    pub fn add(&mut self, vector: Vec<f32>) -> Result<usize> {
        let code = self.quantizer.encode(&vector)?;
        let idx = self.codes.len();
        self.codes.push(code);
        self.originals.push(vector);
        Ok(idx)
    }

    /// Find the `k` nearest vectors to `query` by inner product (larger = closer).
    ///
    /// Returns a `Vec<(index, exact_inner_product)>` sorted by descending exact
    /// inner product, with at most `k` entries.
    ///
    /// If the index contains fewer than `k` vectors, all stored vectors are
    /// returned.
    ///
    /// # Errors
    /// - `DimensionMismatch`, `NonFiniteInput`
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }
        let k = k.min(self.len());
        if k == 0 {
            return Ok(Vec::new());
        }

        // ------------------------------------------------------------------
        // Phase 1: approximate inner product on all compressed codes.
        // ------------------------------------------------------------------
        let mut approx_scores: Vec<(usize, f32)> = self
            .codes
            .iter()
            .enumerate()
            .map(|(i, code)| -> Result<(usize, f32)> {
                let score = self.quantizer.inner_product_estimate(code, query)?;
                Ok((i, score))
            })
            .collect::<Result<Vec<_>>>()?;

        // ------------------------------------------------------------------
        // Phase 2: keep the top k * oversample_factor candidates.
        // ------------------------------------------------------------------
        let candidates = (k * self.oversample_factor).min(self.len());
        if candidates == 0 {
            return Ok(vec![]);
        }
        // Partial sort: bring the top `candidates` entries to the front.
        approx_scores.select_nth_unstable_by(candidates - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal)
        });
        approx_scores.truncate(candidates);

        // ------------------------------------------------------------------
        // Phase 3: exact f32 inner product on the original vectors.
        // ------------------------------------------------------------------
        let mut exact_scores: Vec<(usize, f32)> = approx_scores
            .iter()
            .map(|&(i, _)| {
                let exact = Self::exact_inner_product(&self.originals[i], query)?;
                Ok((i, exact))
            })
            .collect::<Result<Vec<_>>>()?;

        // ------------------------------------------------------------------
        // Phase 4: sort by descending exact score and return top k.
        // ------------------------------------------------------------------
        exact_scores.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal)
        });
        exact_scores.truncate(k);

        Ok(exact_scores)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    #[inline]
    fn exact_inner_product(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(TurboQuantError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TurboQuantizer;

    fn make_index(dim: usize, oversample: usize) -> OversampledSearch<TurboQuantizer> {
        let q = TurboQuantizer::new(dim, 4, dim.max(4), 42).unwrap();
        OversampledSearch::new(q, oversample)
    }

    fn make_vector(dim: usize, seed: usize) -> Vec<f32> {
        (0..dim).map(|i| ((seed * dim + i) as f32 * 0.37).sin()).collect()
    }

    #[test]
    fn test_new_empty() {
        let index = make_index(16, 4);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_add_returns_sequential_indices() {
        let mut index = make_index(16, 4);
        for i in 0..5 {
            let idx = index.add(make_vector(16, i)).unwrap();
            assert_eq!(idx, i);
        }
        assert_eq!(index.len(), 5);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_search_empty_returns_empty() {
        let index = make_index(16, 4);
        let query = make_vector(16, 99);
        let results = index.search(&query, 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_k_zero_returns_empty() {
        let mut index = make_index(16, 4);
        index.add(make_vector(16, 0)).unwrap();
        let results = index.search(&make_vector(16, 99), 0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_returns_k_results() {
        let mut index = make_index(16, 4);
        for i in 0..20 {
            index.add(make_vector(16, i)).unwrap();
        }
        let query = make_vector(16, 99);
        let results = index.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_search_fewer_than_k_returns_all() {
        let mut index = make_index(16, 4);
        for i in 0..3 {
            index.add(make_vector(16, i)).unwrap();
        }
        let query = make_vector(16, 99);
        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_results_sorted_by_descending_score() {
        let mut index = make_index(32, 4);
        for i in 0..20 {
            index.add(make_vector(32, i)).unwrap();
        }
        let query = make_vector(32, 42);
        let results = index.search(&query, 5).unwrap();
        for window in results.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "results not sorted: {} < {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn test_search_indices_in_range() {
        let n = 15usize;
        let mut index = make_index(16, 4);
        for i in 0..n {
            index.add(make_vector(16, i)).unwrap();
        }
        let query = make_vector(16, 99);
        let results = index.search(&query, 5).unwrap();
        for (idx, _) in &results {
            assert!(*idx < n, "index {idx} out of range {n}");
        }
    }

    #[test]
    fn test_recall_self_is_top_result() {
        // Insert a distinctive query vector and verify it is recalled as top-1.
        let dim = 32usize;
        let mut index = make_index(dim, 8);
        // Add noise vectors.
        for i in 0..50 {
            index.add(make_vector(dim, i + 1)).unwrap();
        }
        // Add the query vector itself at a known index.
        let query: Vec<f32> = (0..dim).map(|j| j as f32).collect();
        let query_idx = index.add(query.clone()).unwrap();

        let results = index.search(&query, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].0, query_idx,
            "query vector should be its own nearest neighbour"
        );
    }

    #[test]
    fn test_add_wrong_dimension() {
        let mut index = make_index(16, 4);
        let bad = vec![0.0_f32; 8];
        assert!(index.add(bad).is_err());
        assert_eq!(index.len(), 0, "failed add should not increment len");
    }

    #[test]
    fn test_search_single_vector() {
        let mut index = make_index(16, 4);
        index.add(make_vector(16, 0)).unwrap();
        let query = make_vector(16, 1);
        let results = index.search(&query, 3).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_oversample_factor_one() {
        let q = TurboQuantizer::new(16, 4, 8, 42).unwrap();
        let mut index = OversampledSearch::new(q, 1);
        for i in 0..10 {
            index.add(make_vector(16, i)).unwrap();
        }
        let results = index.search(&make_vector(16, 99), 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_k_larger_than_n_returns_all() {
        // k > indexed vectors → returns all vectors.
        let mut index = make_index(16, 4);
        for i in 0..5 {
            index.add(make_vector(16, i)).unwrap();
        }
        let results = index.search(&make_vector(16, 99), 100).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_search_exact_scores_are_finite() {
        let mut index = make_index(16, 4);
        for i in 0..10 {
            index.add(make_vector(16, i)).unwrap();
        }
        let query = make_vector(16, 42);
        let results = index.search(&query, 5).unwrap();
        for (_, score) in &results {
            assert!(score.is_finite(), "score must be finite, got {score}");
        }
    }

    #[test]
    fn test_add_non_finite_rejected() {
        let mut index = make_index(16, 4);
        let mut bad = make_vector(16, 0);
        bad[3] = f32::NAN;
        assert!(index.add(bad).is_err());
        assert_eq!(index.len(), 0, "failed add should not grow the index");
    }

    #[test]
    #[should_panic(expected = "oversample_factor must be >= 1")]
    fn test_zero_oversample_factor_panics() {
        let q = TurboQuantizer::new(16, 4, 8, 42).unwrap();
        let _ = OversampledSearch::new(q, 0);
    }
}
