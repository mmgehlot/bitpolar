//! Example: approximate vector search using TurboQuantizer.
//!
//! Demonstrates building a compressed vector database of 1000 vectors,
//! querying with a probe vector, and retrieving the top-5 results by
//! estimated inner product.
//!
//! Run with:
//!   cargo run -p bitpolar --example vector_search

use bitpolar::{TurboCode, TurboQuantizer};

fn main() {
    // 1. Create a quantizer — fully defined by 4 integers, no training needed.
    //    dim=128, bits=4, projections=32, seed=42 (deterministic)
    let dim = 128usize;
    let q = TurboQuantizer::new(dim, 4, 32, 42).expect("failed to create quantizer");

    println!(
        "TurboQuantizer: dim={} bits={} projections={}",
        q.dim(),
        q.bits(),
        q.projections()
    );

    // 2. Build a compressed database of 1000 vectors with a deterministic seed.
    //    Each vector is generated from a sine function indexed by (i, j) so
    //    the data is reproducible and does not require an external RNG crate.
    let db: Vec<TurboCode> = (0..1000_usize)
        .map(|i| {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.01).sin())
                .collect();
            q.encode(&v).expect("encode failed")
        })
        .collect();

    println!("Encoded {} vectors.", db.len());

    // 3. Print compression statistics.
    let stats = q.batch_stats(&db);
    println!(
        "Compression: {:.2}x ratio, {:.2} bits/value ({} -> {} bytes total)",
        stats.compression_ratio, stats.bits_per_value, stats.original_bytes, stats.compressed_bytes,
    );

    // 4. Build a query vector (deterministic, different pattern from the database).
    let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.02).cos()).collect();

    // 5. Score every database entry against the query using IP estimation.
    //    No decoding required — scoring runs directly on compressed codes.
    let mut scored: Vec<(usize, f32)> = db
        .iter()
        .enumerate()
        .map(|(i, code)| {
            let score = q.inner_product_estimate(code, &query).unwrap_or(f32::MIN);
            (i, score)
        })
        .collect();

    // 6. Sort descending by score and take the top 5.
    scored.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    println!("\nTop-5 results by estimated inner product:");
    println!("{:<8} {}", "Index", "Score");
    println!("{}", "-".repeat(20));
    for (rank, (idx, score)) in scored.iter().take(5).enumerate() {
        println!("#{:<7} idx={:<6} score={:.4}", rank + 1, idx, score);
    }

    // 7. Decode the top result back to an approximate f32 vector.
    let (top_idx, top_score) = scored[0];
    let reconstructed = q.decode(&db[top_idx]);
    println!(
        "\nDecoded top result (index={}, score={:.4}), first 4 dims: {:?}",
        top_idx,
        top_score,
        &reconstructed[..4]
    );
}
