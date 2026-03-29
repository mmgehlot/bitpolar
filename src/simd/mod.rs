//! SIMD-accelerated kernels (NEON on aarch64, AVX2 on x86_64).
//!
//! This module is only compiled when the `simd` feature is enabled.
//! All public items in this module use unsafe code gated by the
//! crate-level `#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]`.
//!
//! # Runtime Dispatch
//!
//! Both `matvec_multiply` and `dot_product` use compile-time target
//! detection to select an implementation:
//!
//! | Platform   | Implementation                   |
//! |------------|----------------------------------|
//! | `aarch64`  | NEON 128-bit float32x4 intrinsics |
//! | `x86_64`   | AVX2 (detected at runtime)       |
//! | fallback   | Scalar with 4× loop unrolling    |
//!
//! The fallback path is always present and used on other targets or when
//! AVX2 is unavailable at runtime on x86_64.

// ============================================================================
// aarch64 NEON implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON matrix-vector multiply: `out = matrix × vector`.
    ///
    /// `matrix` is row-major with `dim` columns per row and `out.len()` rows.
    ///
    /// # Safety
    /// `matrix.len() == out.len() * dim`, `vector.len() == dim`.
    ///
    /// # Panics
    /// Panics in debug mode on dimension mismatch; UB in release if violated.
    #[inline]
    pub(super) unsafe fn matvec(matrix: &[f32], vector: &[f32], dim: usize, out: &mut [f32]) {
        debug_assert!(
            matrix.len() >= out.len() * dim,
            "matrix too short for matvec"
        );
        debug_assert!(vector.len() >= dim, "vector too short for matvec");
        let rows = out.len();
        for i in 0..rows {
            let row = &matrix[i * dim..(i + 1) * dim];
            out[i] = dot(row, vector, dim);
        }
    }

    /// NEON dot product of two f32 slices of length `dim`.
    ///
    /// # Safety
    /// `a.len() >= dim`, `b.len() >= dim`.
    #[inline]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32], dim: usize) -> f32 {
        debug_assert!(a.len() >= dim, "a too short for dot");
        debug_assert!(b.len() >= dim, "b too short for dot");
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;

        // Process 4 elements at a time with NEON float32x4
        while i + 4 <= dim {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            acc = vmlaq_f32(acc, va, vb);
            i += 4;
        }

        // Horizontal add the 4 lanes
        let sum4 = vaddvq_f32(acc);

        // Scalar tail
        let mut tail = 0.0f32;
        while i < dim {
            tail += a[i] * b[i];
            i += 1;
        }
        sum4 + tail
    }
}

// ============================================================================
// x86_64 AVX2 implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2 matrix-vector multiply: `out = matrix × vector`.
    ///
    /// # Safety
    /// `matrix.len() == out.len() * dim`, `vector.len() == dim`.
    /// The caller must have verified AVX2 support via `is_x86_feature_detected!("avx2")`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn matvec(matrix: &[f32], vector: &[f32], dim: usize, out: &mut [f32]) {
        debug_assert!(
            matrix.len() >= out.len() * dim,
            "matrix too short for matvec"
        );
        debug_assert!(vector.len() >= dim, "vector too short for matvec");
        let rows = out.len();
        for i in 0..rows {
            let row = &matrix[i * dim..(i + 1) * dim];
            out[i] = dot(row, vector, dim);
        }
    }

    /// AVX2 dot product of two f32 slices of length `dim`.
    ///
    /// # Safety
    /// `a.len() >= dim`, `b.len() >= dim`.
    /// The caller must have verified AVX2 support.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32], dim: usize) -> f32 {
        debug_assert!(a.len() >= dim, "a too short for dot");
        debug_assert!(b.len() >= dim, "b too short for dot");
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;

        // Process 8 elements at a time with 256-bit YMM registers
        while i + 8 <= dim {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            // Fused multiply-add: acc = acc + va * vb
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }

        // Horizontal reduction of 8 lanes → scalar
        // Step 1: add upper 128 bits to lower 128 bits
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let sum128 = _mm_add_ps(lo, hi);

        // Step 2: reduce 4 lanes → 2 → 1
        let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
        let sums = _mm_add_ps(sum128, shuf); // [0+1, ?, 2+3, ?]
        let shuf2 = _mm_movehl_ps(shuf, sums); // [2+3, ?, ?, ?]
        let total = _mm_add_ss(sums, shuf2);
        let mut scalar = _mm_cvtss_f32(total);

        // Scalar tail for remaining elements
        while i < dim {
            scalar += a[i] * b[i];
            i += 1;
        }
        scalar
    }
}

// ============================================================================
// Scalar fallback with 4x loop unrolling
// ============================================================================

mod scalar {
    /// Scalar matrix-vector multiply with 4× loop unrolling.
    ///
    /// `out = matrix × vector` where `matrix` is `rows × dim` row-major.
    #[inline]
    pub(super) fn matvec(matrix: &[f32], vector: &[f32], dim: usize, out: &mut [f32]) {
        let rows = out.len();
        for i in 0..rows {
            let row = &matrix[i * dim..(i + 1) * dim];
            out[i] = dot(row, vector);
        }
    }

    /// Scalar dot product with 4× manual loop unrolling.
    ///
    /// Helps the compiler auto-vectorize on platforms without explicit intrinsics.
    #[inline]
    pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;
        let mut i = 0usize;

        while i + 4 <= n {
            acc0 += a[i] * b[i];
            acc1 += a[i + 1] * b[i + 1];
            acc2 += a[i + 2] * b[i + 2];
            acc3 += a[i + 3] * b[i + 3];
            i += 4;
        }
        let mut sum = acc0 + acc1 + acc2 + acc3;
        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Compute `out = matrix × vector` using the best available SIMD kernel.
///
/// `matrix` is a row-major flat slice with `dim` columns. The number of rows
/// equals `out.len()`. `vector` must have length `dim`.
///
/// Dispatch order:
/// 1. NEON on `aarch64`
/// 2. AVX2 on `x86_64` if detected at runtime
/// 3. Scalar with 4× loop unrolling (all other platforms / fallback)
///
/// # Panics
///
/// Panics if `matrix.len() != out.len() * dim` or `vector.len() != dim`.
#[inline]
pub fn matvec_multiply(matrix: &[f32], vector: &[f32], dim: usize, out: &mut [f32]) {
    let rows = out.len();
    assert_eq!(
        matrix.len(),
        rows * dim,
        "matrix must be rows×dim = {rows}×{dim} = {}; got {}",
        rows * dim,
        matrix.len()
    );
    assert_eq!(
        vector.len(),
        dim,
        "vector length must be dim={dim}; got {}",
        vector.len()
    );
    if dim == 0 || rows == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: lengths verified above; NEON is always available on aarch64.
        unsafe { neon::matvec(matrix, vector, dim, out) }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: lengths verified above; AVX2+FMA support confirmed.
            unsafe { avx2::matvec(matrix, vector, dim, out) }
            return;
        }
    }

    // Fallback scalar path (also covers non-aarch64/non-x86_64 targets and
    // the compile-time dead-code path on aarch64 below the early return).
    scalar::matvec(matrix, vector, dim, out);
}

/// Compute the dot product `a · b` using the best available SIMD kernel.
///
/// `a` and `b` must have the same length. If lengths differ, only
/// `min(a.len(), b.len())` elements are used.
///
/// Dispatch order:
/// 1. NEON on `aarch64`
/// 2. AVX2 on `x86_64` if detected at runtime
/// 3. Scalar with 4× loop unrolling (all other platforms / fallback)
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let dim = a.len().min(b.len());
    if dim == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: dim == a.len().min(b.len()) so bounds are safe.
        return unsafe { neon::dot(a, b, dim) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: dim verified above.
            return unsafe { avx2::dot(a, b, dim) };
        }
    }

    scalar::dot(a, b)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn scalar_matvec(matrix: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
        let rows = matrix.len() / dim;
        (0..rows)
            .map(|i| {
                let row = &matrix[i * dim..(i + 1) * dim];
                scalar_dot(row, vector)
            })
            .collect()
    }

    #[test]
    fn test_dot_product_matches_scalar() {
        let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32 * 0.7).sin()).collect();
        let expected = scalar_dot(&a, &b);
        let got = dot_product(&a, &b);
        assert!(
            (got - expected).abs() < 1e-3,
            "dot_product mismatch: expected={expected}, got={got}"
        );
    }

    #[test]
    fn test_dot_product_empty() {
        assert_eq!(dot_product(&[], &[]), 0.0);
    }

    #[test]
    fn test_dot_product_single() {
        assert!((dot_product(&[3.0], &[4.0]) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_all_ones() {
        let a = vec![1.0f32; 128];
        let b = vec![1.0f32; 128];
        let result = dot_product(&a, &b);
        assert!((result - 128.0).abs() < 1e-4);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        // Alternating +1/-1 should cancel to near zero
        let a: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let b: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { 1.0 })
            .collect();
        let expected = scalar_dot(&a, &b);
        let got = dot_product(&a, &b);
        assert!((got - expected).abs() < 1e-4);
    }

    #[test]
    fn test_matvec_multiply_matches_scalar() {
        let dim = 16usize;
        let rows = 8usize;
        let matrix: Vec<f32> = (0..(rows * dim)).map(|i| (i as f32 * 0.1).sin()).collect();
        let vector: Vec<f32> = (0..dim).map(|i| i as f32 * 0.05).collect();

        let expected = scalar_matvec(&matrix, &vector, dim);
        let mut got = vec![0.0f32; rows];
        matvec_multiply(&matrix, &vector, dim, &mut got);

        for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (e - g).abs() < 1e-3,
                "matvec mismatch at row {i}: expected={e}, got={g}"
            );
        }
    }

    #[test]
    fn test_matvec_multiply_identity() {
        // 4×4 identity matrix × [1,2,3,4] should give [1,2,3,4]
        let dim = 4usize;
        #[rustfmt::skip]
        let matrix = vec![
            1.0_f32, 0.0, 0.0, 0.0,
            0.0,     1.0, 0.0, 0.0,
            0.0,     0.0, 1.0, 0.0,
            0.0,     0.0, 0.0, 1.0,
        ];
        let vector = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        matvec_multiply(&matrix, &vector, dim, &mut out);
        for (i, (&o, &v)) in out.iter().zip(vector.iter()).enumerate() {
            assert!(
                (o - v).abs() < 1e-6,
                "identity multiply failed at index {i}"
            );
        }
    }

    #[test]
    fn test_matvec_multiply_zero_vector() {
        let dim = 8usize;
        let matrix: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let vector = vec![0.0f32; dim];
        let mut out = vec![99.0f32; 2];
        matvec_multiply(&matrix, &vector, dim, &mut out);
        for &v in &out {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_matvec_multiply_larger() {
        let dim = 128usize;
        let rows = 32usize;
        let matrix: Vec<f32> = (0..(rows * dim))
            .map(|i| ((i as f32) * 0.01).cos())
            .collect();
        let vector: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.03).sin()).collect();

        let expected = scalar_matvec(&matrix, &vector, dim);
        let mut got = vec![0.0f32; rows];
        matvec_multiply(&matrix, &vector, dim, &mut got);

        for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (e - g).abs() < 1e-2,
                "large matvec mismatch at row {i}: expected={e}, got={g}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_dot_matches_scalar() {
        let a: Vec<f32> = (1..=32).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (1..=32).map(|i| (i as f32).recip()).collect();
        let expected = scalar_dot(&a, &b);
        let got = unsafe { super::neon::dot(&a, &b, 32) };
        assert!((got - expected).abs() < 1e-4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_dot_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return; // Skip if AVX2 not available
        }
        let a: Vec<f32> = (1..=32).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (1..=32).map(|i| (i as f32).recip()).collect();
        let expected = scalar_dot(&a, &b);
        let got = unsafe { super::avx2::dot(&a, &b, 32) };
        assert!((got - expected).abs() < 1e-4);
    }

    #[test]
    fn test_scalar_dot_unrolled() {
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect(); // non-multiple of 4
        let b: Vec<f32> = (0..17).map(|i| i as f32 * 2.0).collect();
        let expected = scalar_dot(&a, &b);
        let got = scalar::dot(&a, &b);
        assert!((got - expected).abs() < 1e-4);
    }
}
