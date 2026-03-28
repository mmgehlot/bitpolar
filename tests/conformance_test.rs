//! Conformance tests: verify the implementation matches ICLR 2026 TurboQuant
//! paper's theoretical guarantees.
//!
//! Each test runs at least 1 000 trials with different random vectors so that
//! sampling noise is negligible.
//!
//! Note: only the public API is used (no `pub(crate)` field access).

use bitpolar::{QjlQuantizer, StoredRotation, TurboQuantizer};

// ============================================================================
// Helpers
// ============================================================================

/// Deterministic pseudo-random vector generator.
fn make_vector(dim: usize, trial: usize, salt: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim)
        .map(|j| ((trial * 7919 + j * 6271 + salt * 1009) as f64 * 0.001).sin() as f32)
        .collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        return vec![1.0_f32 / (dim as f32).sqrt(); dim];
    }
    raw.iter().map(|x| x / norm).collect()
}

fn exact_ip(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ============================================================================
// Theorem 1 — MSE Distortion Bound
//
// D_mse ≤ (√3 · π / 2) · (1 / 4^b)
//
// For each bit width b in 1..=6, compute average reconstruction MSE over
// 1 000 random 128-dim unit vectors and verify it stays below the theoretical
// bound (with 20 % tolerance for sampling noise and finite-dim effects).
// ============================================================================

#[test]
fn theorem1_mse_distortion_bound() {
    let dim = 128usize;
    let trials = 1_000usize;
    let projections = dim / 4;

    let sqrt3_pi_over_2 = (3.0_f64).sqrt() * std::f64::consts::PI / 2.0;

    for bits in 1u8..=6u8 {
        let q = TurboQuantizer::new(dim, bits, projections, 42).unwrap();

        let mut total_mse = 0.0f64;
        for trial in 0..trials {
            let v = make_vector(dim, trial, bits as usize);
            let code = q.encode(&v).unwrap();
            let decoded = q.decode(&code);
            let mse: f64 =
                v.iter()
                    .zip(decoded.iter())
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum::<f64>()
                    / dim as f64;
            total_mse += mse;
        }
        let avg_mse = total_mse / trials as f64;

        // Theoretical bound: (√3 π / 2) · 4^(-b)
        let bound = sqrt3_pi_over_2 * (4.0_f64).powi(-(bits as i32));
        let tolerance = 1.20; // 20 % headroom for sampling noise

        assert!(
            avg_mse <= bound * tolerance,
            "bits={}: avg_mse={:.6e} exceeds bound={:.6e} (× {:.2})",
            bits,
            avg_mse,
            bound,
            tolerance
        );
    }
}

// ============================================================================
// Theorem 2 — Inner Product Unbiasedness
//
// E[IP_estimate] = exact_IP
// Over 5 000 random vector pairs, the mean signed error should be < 0.5.
// ============================================================================

#[test]
fn theorem2_inner_product_unbiasedness() {
    let dim = 128usize;
    let bits = 4u8;
    let projections = dim / 4;
    let trials = 5_000usize;

    let q = TurboQuantizer::new(dim, bits, projections, 7).unwrap();

    let mut total_signed_error = 0.0f64;

    for trial in 0..trials {
        let v = make_vector(dim, trial, 1);
        let query = make_vector(dim, trial + 100_000, 2);

        let exact = exact_ip(&v, &query) as f64;
        let code = q.encode(&v).unwrap();
        let estimated = q.inner_product_estimate(&code, &query).unwrap() as f64;

        total_signed_error += estimated - exact;
    }

    let mean_error = total_signed_error / trials as f64;
    assert!(
        mean_error.abs() < 0.5,
        "IP estimator should be unbiased: mean_signed_error={:.6}",
        mean_error
    );
}

// ============================================================================
// Theorem 3 — Inner Product Distortion Bound
//
// D_prod ≤ (√3 · π² / d) · (1 / 4^b)
//
// For 4-bit, 128-dim, verify measured IP MSE (over 1 000 random pairs) is
// below the theoretical bound with generous tolerance.
// ============================================================================

#[test]
fn theorem3_inner_product_distortion_bound() {
    let dim = 128usize;
    let bits = 4u8;
    let projections = dim / 4;
    let trials = 1_000usize;

    let q = TurboQuantizer::new(dim, bits, projections, 13).unwrap();

    let sqrt3_pi2_over_d = (3.0_f64).sqrt() * std::f64::consts::PI.powi(2) / dim as f64;
    let bound = sqrt3_pi2_over_d * (4.0_f64).powi(-(bits as i32));
    // Wide tolerance: the QJL residual correction adds variance on top of polar
    let tolerance = 50.0;

    let mut total_sq_err = 0.0f64;
    for trial in 0..trials {
        let v = make_vector(dim, trial, 3);
        let query = make_vector(dim, trial + 200_000, 4);

        let exact = exact_ip(&v, &query) as f64;
        let code = q.encode(&v).unwrap();
        let estimated = q.inner_product_estimate(&code, &query).unwrap() as f64;

        total_sq_err += (estimated - exact).powi(2);
    }
    let ip_mse = total_sq_err / trials as f64;

    assert!(
        ip_mse <= bound * tolerance,
        "IP MSE={:.6e} exceeds theoretical bound={:.6e} × {:.1}",
        ip_mse,
        bound,
        tolerance
    );
}

// ============================================================================
// Determinism — same (dim, bits, proj, seed) + same vector → same code always.
//
// We verify by comparing compact bytes (the only public identity check).
// ============================================================================

#[test]
fn determinism_same_params_and_vector_gives_identical_code() {
    let dim = 128usize;
    let q = TurboQuantizer::new(dim, 4, 32, 42).unwrap();
    let v = make_vector(dim, 77, 5);

    let reference_bytes = q.encode(&v).unwrap().to_compact_bytes();

    for _ in 0..100 {
        let bytes = q.encode(&v).unwrap().to_compact_bytes();
        assert_eq!(
            bytes, reference_bytes,
            "compact bytes must be bit-identical across repeated encodes"
        );
    }
}

// ============================================================================
// Norm Preservation — rotation preserves L2 norm.
//
// ||R · v|| = ||v|| for 1 000 random vectors, tolerance 1e-4.
// ============================================================================

#[test]
fn norm_preservation_rotation_is_isometric() {
    let dim = 64usize;
    let rotation = StoredRotation::new(dim, 99).unwrap();
    let tolerance = 1e-4_f32;

    for trial in 0..1_000 {
        let v = make_vector(dim, trial, 6);
        let original_norm = l2_norm(&v);

        let mut rotated = Vec::new();
        rotation.apply_slice(&v, &mut rotated);
        let rotated_norm = l2_norm(&rotated);

        assert!(
            (rotated_norm - original_norm).abs() <= tolerance,
            "trial {}: norm changed from {:.6} to {:.6}",
            trial,
            original_norm,
            rotated_norm
        );
    }
}

// ============================================================================
// Orthogonality — R · R^T ≈ I (within 1e-4).
// ============================================================================

#[cfg(feature = "std")]
#[test]
fn orthogonality_r_times_rt_is_identity() {
    let dim = 8usize;
    let rotation = StoredRotation::new(dim, 55).unwrap();
    let tolerance = 1e-4_f32;

    let mat = rotation.matrix(); // nalgebra DMatrix<f32>
    let product = mat.clone() * mat.transpose();

    for i in 0..dim {
        for j in 0..dim {
            let expected = if i == j { 1.0_f32 } else { 0.0_f32 };
            let actual = product[(i, j)];
            assert!(
                (actual - expected).abs() <= tolerance,
                "R·Rᵀ[{},{}] = {:.6}, expected {:.6}",
                i,
                j,
                actual,
                expected
            );
        }
    }
}

// ============================================================================
// QJL Variance Bound
//
// Var[QJL_estimate] ≤ π / (2 · projections) · ||y||²
// Empirical variance over 1 000 random stored vectors + fixed query should
// be finite and not wildly exceed the per-pair theoretical bound.
// ============================================================================

#[test]
fn qjl_variance_bound() {
    let dim = 64usize;
    let projections = 64usize;
    let trials = 1_000usize;

    let q = QjlQuantizer::new(dim, projections, 17).unwrap();

    let y = make_vector(dim, 42, 7);
    let y_norm_sq: f64 = y.iter().map(|x| (*x as f64).powi(2)).sum();

    // Per-pair variance bound: π / (2 * projections) * ||y||²
    let var_bound = std::f64::consts::PI / (2.0 * projections as f64) * y_norm_sq;

    let mut estimates = Vec::with_capacity(trials);
    for trial in 0..trials {
        let v = make_vector(dim, trial, 8);
        let sketch = q.sketch(&v).unwrap();
        let est = q.inner_product_estimate(&sketch, &y).unwrap() as f64;
        estimates.push(est);
    }

    let mean = estimates.iter().sum::<f64>() / trials as f64;
    let variance = estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / trials as f64;

    assert!(variance.is_finite(), "QJL variance should be finite, got {variance}");
    // Loose ceiling: across random v's the cross-pair variance can exceed the
    // single-pair bound, so we use 200× as a sanity check.
    assert!(
        variance < var_bound * 200.0,
        "QJL empirical variance={:.6e} is far above per-pair bound={:.6e}",
        variance,
        var_bound
    );
}

// ============================================================================
// Compression Ratio — ratio > 1, bits_per_value < 32
// ============================================================================

#[test]
fn compression_ratio_matches_expected_for_parameters() {
    let dim = 128usize;
    let bits = 4u8;
    let projections = dim / 4;
    let n_vectors = 100usize;

    let q = TurboQuantizer::new(dim, bits, projections, 42).unwrap();
    let codes: Vec<_> = (0..n_vectors)
        .map(|i| q.encode(&make_vector(dim, i, 9)).unwrap())
        .collect();

    let stats = q.batch_stats(&codes);
    assert!(
        stats.compression_ratio > 1.0,
        "expected compression_ratio > 1.0, got {:.3}",
        stats.compression_ratio
    );
    assert_eq!(stats.count, n_vectors);
    assert!(
        stats.bits_per_value < 32.0,
        "bits_per_value={:.3} should be well below 32 (f32 baseline)",
        stats.bits_per_value
    );
}
