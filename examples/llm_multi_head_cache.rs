//! Multi-head KV cache compression for transformer inference.
//!
//! Compresses Key-Value caches across all attention heads with
//! per-head seed isolation. Simulates a Llama-style model.
//!
//! Run: `cargo run --example llm_multi_head_cache`

use bitpolar::kv_cache::{KvCacheConfig, MultiHeadKvCache};

fn main() {
    // Config: 4 heads, 64 head_dim (small for example speed)
    let config = KvCacheConfig {
        head_dim: 64,
        bits: 4,
        projections: 16,
        seed: 42,
    };
    let num_heads = 4;
    let mut cache = MultiHeadKvCache::new(num_heads, &config).unwrap();

    println!(
        "=== Multi-Head KV Cache ({} heads × {} dim, {}-bit) ===\n",
        num_heads, config.head_dim, config.bits
    );

    // Simulate 50 tokens
    for token in 0..50 {
        // Generate dummy K/V vectors for all heads
        let keys: Vec<Vec<f32>> = (0..num_heads)
            .map(|h| {
                (0..config.head_dim)
                    .map(|d| ((token * num_heads + h) as f32 * 0.01 + d as f32 * 0.001).sin())
                    .collect()
            })
            .collect();
        let values = keys.clone();

        // Convert to &[&[f32]] for the API
        let key_refs: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let val_refs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();

        cache.push_token(&key_refs, &val_refs).unwrap();
    }

    println!("Tokens cached: 50");
    println!("Cache length: {} tokens", cache.len());
}
