//! Property-based tests using proptest.
//!
//! Invariants that must hold for every input covered by the strategy.
//! Only the public API is used — no access to `pub(crate)` fields.

use bitpolar::{TurboCode, TurboQuantizer};
use proptest::prelude::*;

// ============================================================================
// Helpers
// ============================================================================

fn make_quantizer(dim_mult: usize, seed: u64) -> TurboQuantizer {
    let dim = dim_mult * 2;
    let projections = (dim / 4).max(1);
    TurboQuantizer::new(dim, 4, projections, seed).unwrap()
}

fn make_vec(dim: usize, trial: u64) -> Vec<f32> {
    (0..dim)
        .map(|j| (trial as f64 * 1.3 + j as f64 * 0.7).sin() as f32)
        .collect()
}

// ============================================================================
// Property 1 — Determinism
//
// Encoding the same vector twice always produces identical compact bytes.
// ============================================================================

proptest! {
    #[test]
    fn turbo_encode_deterministic(
        dim_mult in 1usize..8,
        seed in 0u64..1000,
        trial in 0u64..500,
    ) {
        let q = make_quantizer(dim_mult, seed);
        let v = make_vec(q.dim(), trial);

        let bytes1 = q.encode(&v).unwrap().to_compact_bytes();
        let bytes2 = q.encode(&v).unwrap().to_compact_bytes();

        prop_assert_eq!(bytes1, bytes2, "compact bytes must be identical on repeated encodes");
    }
}

// ============================================================================
// Property 2 — Polar Norm Preservation
//
// Encode then decode: the decoded vector's norm should be within 15 % of the
// original.
// ============================================================================

proptest! {
    #[test]
    fn polar_norm_preserved(
        dim_mult in 1usize..8,
        trial in 0u64..500,
    ) {
        let dim = dim_mult * 2;
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, 4, projections, 42).unwrap();
        let v = make_vec(dim, trial);

        let original_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assume!(original_norm > 1e-4);

        let code = q.encode(&v).unwrap();
        let decoded = q.decode(&code);
        let decoded_norm: f32 = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();

        let ratio = decoded_norm / original_norm;
        prop_assert!(
            (ratio - 1.0).abs() < 0.15,
            "norm ratio={:.4} outside [0.85, 1.15]",
            ratio
        );
    }
}

// ============================================================================
// Property 3 — Dimension Mismatch Always Rejected
// ============================================================================

proptest! {
    #[test]
    fn dimension_mismatch_rejected(
        dim_mult in 2usize..8,
        offset in 1usize..4,
    ) {
        let q = make_quantizer(dim_mult, 42);
        let wrong_dim = (q.dim().saturating_sub(offset)).max(1);
        prop_assume!(wrong_dim != q.dim());

        let v = make_vec(wrong_dim, 0);
        let result = q.encode(&v);
        prop_assert!(
            matches!(result, Err(bitpolar::TurboQuantError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {:?}",
            result
        );
    }
}

// ============================================================================
// Property 4 — Seed Independence
//
// Two quantizers with different seeds produce different compact bytes for the
// residual sketch (verified via full code bytes differing).
// Requires `std` feature for full QR rotation; the no_std diagonal fallback
// has too few degrees of freedom at small dimensions.
// ============================================================================

#[cfg(feature = "std")]
proptest! {
    #[test]
    fn seed_independence(
        dim_mult in 1usize..6,
        seed_a in 0u64..500,
        seed_b in 500u64..1000,
        trial in 0u64..200,
    ) {
        let dim = dim_mult * 2;
        let projections = (dim / 4).max(4);
        let qa = TurboQuantizer::new(dim, 4, projections, seed_a).unwrap();
        let qb = TurboQuantizer::new(dim, 4, projections, seed_b).unwrap();
        let v = make_vec(dim, trial);

        let bytes_a = qa.encode(&v).unwrap().to_compact_bytes();
        let bytes_b = qb.encode(&v).unwrap().to_compact_bytes();

        // Different seeds → different rotation AND projection matrices →
        // (virtually certainly) different encoded bytes.
        prop_assert_ne!(bytes_a, bytes_b,
            "different seeds should (almost surely) produce different codes");
    }
}

// ============================================================================
// Property 5 — Compact Bytes Roundtrip
// ============================================================================

proptest! {
    #[test]
    fn compact_bytes_roundtrip(
        dim_mult in 1usize..8,
        seed in 0u64..500,
        trial in 0u64..300,
    ) {
        let q = make_quantizer(dim_mult, seed);
        let v = make_vec(q.dim(), trial);

        let code = q.encode(&v).unwrap();
        let bytes = code.to_compact_bytes();
        let restored = TurboCode::from_compact_bytes(&bytes).unwrap();
        let bytes2 = restored.to_compact_bytes();

        prop_assert_eq!(bytes, bytes2,
            "roundtrip must be identity: to_compact_bytes → from_compact_bytes → to_compact_bytes");
    }
}

// ============================================================================
// Property 6 — L2 Self Distance Is Finite And Non-Negative
// ============================================================================

proptest! {
    #[test]
    fn l2_self_distance_finite_and_non_negative(
        dim_mult in 1usize..8,
        trial in 0u64..200,
    ) {
        let dim = dim_mult * 2;
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, 6, projections, 42).unwrap();
        let v = make_vec(dim, trial);

        let code = q.encode(&v).unwrap();
        let dist = q.l2_distance_estimate(&code, &v).unwrap();

        prop_assert!(dist >= 0.0, "L2 distance must be non-negative, got {dist}");
        prop_assert!(dist.is_finite(), "L2 distance must be finite, got {dist}");

        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!(
            dist <= v_norm * 3.0 + 1.0,
            "L2 dist={dist:.4} unreasonably large vs vector norm={v_norm:.4}"
        );
    }
}

// ============================================================================
// Property 7 — IP Sign Consistency: IP(v, v) > 0 for non-zero v
// ============================================================================

proptest! {
    #[test]
    fn ip_sign_consistency(
        dim_mult in 1usize..8,
        trial in 0u64..300,
    ) {
        let dim = dim_mult * 2;
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, 4, projections, 42).unwrap();
        let v = make_vec(dim, trial);

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assume!(norm > 1e-3);

        let code = q.encode(&v).unwrap();
        let est = q.inner_product_estimate(&code, &v).unwrap();

        prop_assert!(
            est > 0.0,
            "IP(v, v) must be positive for non-zero v; got {est:.6}"
        );
    }
}

// ============================================================================
// Property 8 — Size bytes > 0 for any valid encode
// ============================================================================

proptest! {
    #[test]
    fn code_size_bytes_is_positive(
        dim_mult in 1usize..8,
        trial in 0u64..200,
    ) {
        let q = make_quantizer(dim_mult, 42);
        let v = make_vec(q.dim(), trial);
        let code = q.encode(&v).unwrap();
        prop_assert!(code.size_bytes() > 0, "code must occupy at least 1 byte");
        // Compact bytes include a version header + both payloads.
        let compact = code.to_compact_bytes();
        prop_assert!(!compact.is_empty(), "compact bytes must be non-empty");
    }
}
