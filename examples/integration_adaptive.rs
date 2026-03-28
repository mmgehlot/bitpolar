//! Adaptive bit-width with promote/demote.
//!
//! Demonstrates per-vector precision selection and tier transitions:
//! Cold vectors can be promoted to Warm/Hot when accessed frequently,
//! and demoted back when they become cold again.
//!
//! Run: `cargo run --example integration_adaptive`

use bitpolar::adaptive::AdaptiveQuantizer;
use bitpolar::tiered::Tier;

fn main() {
    let aq = AdaptiveQuantizer::builder(128, 42)
        .hot_bits(8)
        .warm_bits(4)
        .cold_bits(3)
        .build()
        .unwrap();

    let vector: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();

    println!("=== Adaptive Quantization (Promote/Demote) ===\n");

    // Start as Cold (maximum compression)
    let code = aq.encode_adaptive(&vector, Tier::Cold).unwrap();
    println!("Initial: {:?} tier", code.tier());

    // Promote Cold → Warm (more precision)
    let promoted = aq.promote(&code).unwrap();
    println!("Promoted: {:?} tier", promoted.tier());

    // Promote Warm → Hot (maximum precision)
    let hot = aq.promote(&promoted).unwrap();
    println!("Promoted: {:?} tier", hot.tier());

    // Demote Hot → Warm → Cold
    let demoted = aq.demote(&hot).unwrap();
    println!("Demoted: {:?} tier", demoted.tier());

    let cold_again = aq.demote(&demoted).unwrap();
    println!("Demoted: {:?} tier", cold_again.tier());

    // Compare reconstruction quality across tiers
    println!("\nReconstruction quality by tier:");
    for tier in [Tier::Hot, Tier::Warm, Tier::Cold] {
        let c = aq.encode_adaptive(&vector, tier).unwrap();
        let decoded = aq.decode_adaptive(&c);
        let error: f32 = vector
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        println!("  {:?}: error = {:.6}", tier, error);
    }
}
