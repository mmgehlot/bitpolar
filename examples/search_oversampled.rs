//! Oversampled search — approximate scoring + exact re-ranking.
//!
//! Demonstrates the two-phase search pattern that recovers recall:
//! 1. Score all compressed vectors (fast, approximate)
//! 2. Re-rank top candidates using exact dot product (slow, precise)
//!
//! Run: `cargo run --example search_oversampled`

use bitpolar::traits::VectorQuantizer;
use bitpolar::TurboQuantizer;

fn main() {
    let dim = 128;
    let n = 5000;
    let top_k = 10;
    let oversample = 3; // Retrieve 3x candidates before re-ranking

    let q = TurboQuantizer::new(dim, 4, 32, 42).unwrap();

    // Generate vectors
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.001).sin())
                .collect()
        })
        .collect();
    let codes: Vec<_> = vectors.iter().map(|v| q.encode(v).unwrap()).collect();

    let query = &vectors[0];

    // Phase 1: Approximate scoring on compressed vectors
    let mut approx_scores: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, code)| (i, q.inner_product_estimate(code, query).unwrap()))
        .collect();
    approx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let candidates: Vec<usize> = approx_scores
        .iter()
        .take(top_k * oversample)
        .map(|(i, _)| *i)
        .collect();

    // Phase 2: Exact re-ranking on original vectors
    let mut exact_scores: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&i| {
            let exact: f32 = vectors[i]
                .iter()
                .zip(query.iter())
                .map(|(a, b)| a * b)
                .sum();
            (i, exact)
        })
        .collect();
    exact_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!(
        "Oversampled search ({}x oversample, top-{}):",
        oversample, top_k
    );
    for (id, score) in exact_scores.iter().take(top_k) {
        println!("  Vector {}: exact score = {:.6}", id, score);
    }
}
