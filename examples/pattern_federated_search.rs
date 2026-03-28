//! Federated search pattern — multiple nodes quantize independently.
//!
//! When all nodes use the same (dim, bits, projections, seed), their
//! quantizers are identical. Compressed codes are directly comparable
//! without coordination, enabling federated vector search.
//!
//! Run: `cargo run --example federated_search`

use bitpolar::TurboQuantizer;
use bitpolar::traits::{VectorQuantizer, SerializableCode};

fn main() {
    // Shared parameters — distributed to all nodes via config
    let dim = 128;
    let bits = 4;
    let projections = 32;
    let seed = 42;

    // Simulate 3 independent nodes
    println!("=== Federated Search: 3 Independent Nodes ===\n");

    // Node A: encodes vectors 0-99
    let node_a = TurboQuantizer::new(dim, bits, projections, seed).unwrap();
    let vectors_a: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.001).sin()).collect())
        .collect();
    let codes_a: Vec<Vec<u8>> = vectors_a
        .iter()
        .map(|v| node_a.encode(v).unwrap().to_compact_bytes())
        .collect();
    println!("Node A: encoded {} vectors", codes_a.len());

    // Node B: encodes vectors 100-199 (completely independent)
    let node_b = TurboQuantizer::new(dim, bits, projections, seed).unwrap();
    let vectors_b: Vec<Vec<f32>> = (100..200)
        .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.001).sin()).collect())
        .collect();
    let codes_b: Vec<Vec<u8>> = vectors_b
        .iter()
        .map(|v| node_b.encode(v).unwrap().to_compact_bytes())
        .collect();
    println!("Node B: encoded {} vectors", codes_b.len());

    // Aggregator: collects codes from all nodes, searches centrally
    let aggregator = TurboQuantizer::new(dim, bits, projections, seed).unwrap();
    let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.01).cos()).collect();

    // Search across all nodes' codes
    let mut all_scores: Vec<(usize, f32)> = Vec::new();

    for (i, code_bytes) in codes_a.iter().enumerate() {
        let code = bitpolar::TurboCode::from_compact_bytes(code_bytes).unwrap();
        let score = aggregator.inner_product_estimate(&code, &query).unwrap();
        all_scores.push((i, score));
    }
    for (i, code_bytes) in codes_b.iter().enumerate() {
        let code = bitpolar::TurboCode::from_compact_bytes(code_bytes).unwrap();
        let score = aggregator.inner_product_estimate(&code, &query).unwrap();
        all_scores.push((100 + i, score));
    }

    // Top-5 results across all nodes
    all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n=== Top 5 Results (Federated) ===");
    for (id, score) in all_scores.iter().take(5) {
        let node = if *id < 100 { "A" } else { "B" };
        println!("  Vector {} (Node {}): score = {:.6}", id, node, score);
    }

    // Verify: same params → identical quantizers (codes are comparable)
    let test_vec = vec![1.0_f32; dim];
    let code_a = node_a.encode(&test_vec).unwrap().to_compact_bytes();
    let code_b = node_b.encode(&test_vec).unwrap().to_compact_bytes();
    assert_eq!(code_a, code_b, "Same params must produce identical codes");
    println!("\nFederated consistency verified: identical quantizers across nodes");
}
