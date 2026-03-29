//! Example: KV cache compression for transformer attention.
//!
//! Demonstrates compressing 100 tokens of attention key/value vectors for a
//! single attention head, computing approximate attention scores for a query,
//! and finding the top-5 attended positions.
//!
//! Run with:
//!   cargo run -p bitpolar --example kv_cache

use bitpolar::{KvCacheCompressor, KvCacheConfig};

fn main() {
    // 1. Configure the compressor for a 64-dimensional attention head.
    //    bits=4, projections=32, seed=99 (deterministic).
    let config = KvCacheConfig {
        head_dim: 64,
        bits: 4,
        projections: 32,
        seed: 99,
    };

    let mut cache = KvCacheCompressor::new(&config).expect("failed to create KV cache");

    println!(
        "KvCacheCompressor: head_dim={} bits={} projections={}",
        config.head_dim, config.bits, config.projections
    );

    // 2. Push 100 tokens (key/value pairs) into the cache.
    //    Keys and values are generated deterministically from token index.
    let num_tokens = 100usize;
    for t in 0..num_tokens {
        let key: Vec<f32> = (0..config.head_dim)
            .map(|i| ((t * config.head_dim + i) as f32 * 0.01).sin())
            .collect();
        let val: Vec<f32> = (0..config.head_dim)
            .map(|i| ((t * config.head_dim + i) as f32 * 0.01).cos())
            .collect();
        cache.push(&key, &val).expect("push failed");
    }

    println!("Stored {} tokens.", cache.len());

    // 3. Print compression ratio.
    println!("Compression ratio: {:.2}x", cache.compression_ratio());

    // 4. Compute attention scores for a query vector.
    //    Scores are pre-softmax: score_t = IP(query, key_t) / sqrt(head_dim).
    let query: Vec<f32> = (0..config.head_dim)
        .map(|i| (i as f32 * 0.02).sin())
        .collect();
    let scores = cache
        .attention_scores(&query)
        .expect("attention_scores failed");
    assert_eq!(scores.len(), num_tokens);

    // 5. Find top-5 attended positions.
    let mut scored: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    scored.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    println!("\nTop-5 attended positions:");
    println!("{:<10} {}", "Position", "Score");
    println!("{}", "-".repeat(22));
    for (rank, (pos, score)) in scored.iter().take(5).enumerate() {
        println!("#{:<9} t={:<6} score={:.4}", rank + 1, pos, score);
    }

    // 6. Decode the value at the top-attended position.
    let (top_pos, top_score) = scored[0];
    let values = cache.decode_values();
    println!(
        "\nDecoded value at top position (t={}, score={:.4}), first 4 dims: {:?}",
        top_pos,
        top_score,
        &values[top_pos][..4]
    );
}
