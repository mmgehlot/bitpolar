//! Brute-force search on compressed vectors.
//!
//! Demonstrates the simplest search pattern: encode all vectors,
//! then scan all codes to find the top-k by inner product.
//!
//! Run: `cargo run --example search_brute_force`

use bitpolar::traits::VectorQuantizer;
use bitpolar::TurboQuantizer;

fn main() {
    let dim = 128;
    let q = TurboQuantizer::new(dim, 4, 32, 42).unwrap();

    // Generate 1000 random vectors
    let vectors: Vec<Vec<f32>> = (0..1000)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.001).sin())
                .collect()
        })
        .collect();

    // Encode all vectors
    let codes: Vec<_> = vectors.iter().map(|v| q.encode(v).unwrap()).collect();

    // Query: find top-5 most similar to vectors[0]
    let query = &vectors[0];
    let mut scores: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, code)| (i, q.inner_product_estimate(code, query).unwrap()))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 5 results for query vector 0:");
    for (id, score) in scores.iter().take(5) {
        println!("  Vector {}: score = {:.6}", id, score);
    }
}
