//! Tiered storage — Hot/Warm/Cold bit-width tiers.
//!
//! Demonstrates access-frequency-aware compression: frequently accessed
//! vectors get more bits (higher accuracy), cold vectors get fewer bits
//! (maximum compression).
//!
//! Run: `cargo run --example integration_tiered_storage`

use bitpolar::tiered::{Tier, TieredQuantization};

fn main() {
    let dim = 128;
    let projections = 32;
    let tq = TieredQuantization::new(dim, projections, 42).unwrap();

    let vector = vec![0.1_f32; dim];

    println!("=== Tiered Storage (Hot/Warm/Cold) ===\n");

    for tier in [Tier::Hot, Tier::Warm, Tier::Cold] {
        let code = tq.encode(&vector, tier).unwrap();
        let decoded = tq.decode(&code);

        // Compute reconstruction error
        let error: f32 = vector
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        println!(
            "{:5} — reconstruction error = {:.6}",
            match tier {
                Tier::Hot => "Hot",
                Tier::Warm => "Warm",
                Tier::Cold => "Cold",
            },
            error
        );
    }
}
