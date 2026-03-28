//! Walsh-Hadamard Transform (WHT) rotation strategy.
//!
//! Provides an O(d log d) alternative to the Haar QR rotation in O(d²).
//! Uses a randomized WHT (random sign-flip diagonal + normalized WHT) for
//! near-Haar random projection properties, satisfying the JL lemma guarantees
//! needed by TurboQuant's polar encoding stage.
//!
//! # Memory
//!
//! The WHT requires only O(d) memory for the sign vector, compared to O(d²)
//! for the full Haar rotation matrix. For d=768:
//! - Haar QR: 768² × 4 = 2.3 MB
//! - WHT: 768 × 4 = 3 KB  (770× smaller)
//!
//! # Dimension Requirements
//!
//! The WHT requires power-of-2 dimensions internally. Non-power-of-2 inputs
//! are zero-padded to the next power of 2, transformed, then truncated.
//!
//! **Important**: For non-power-of-2 dimensions, the round-trip
//! `inverse(forward(x))` is **approximately** (not exactly) equal to `x`.
//! The truncation loses spectral energy in the padded positions. For
//! power-of-2 dimensions, the round-trip is exact.
//!
//! In practice, the quality loss from truncation is small and does not
//! significantly affect quantization recall. For best results, use
//! power-of-2 dimensions (128, 256, 512, 1024).

use crate::error::{Result, TurboQuantError};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

/// A randomized Walsh-Hadamard Transform rotation.
///
/// Applies a random sign-flip diagonal matrix followed by a normalized WHT.
/// This produces near-Haar random projections with O(d) memory and O(d log d)
/// computation, compared to O(d²) for both metrics with the full Haar QR.
///
/// Deterministic: same `(dim, seed)` always produces the same transform.
///
/// # Example
///
/// ```rust
/// use bitpolar::wht::WhtRotation;
///
/// let wht = WhtRotation::new(128, 42).unwrap();
/// let v = vec![1.0_f32; 128];
/// let rotated = wht.forward(&v);
/// let recovered = wht.inverse(&rotated);
/// // recovered ≈ v (within floating-point precision)
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(serde::Serialize, serde::Deserialize))]
pub struct WhtRotation {
    /// Original vector dimension (may not be power of 2)
    dim: usize,
    /// Padded dimension (next power of 2 >= dim)
    padded_dim: usize,
    /// Random sign vector (+1.0 or -1.0), length = padded_dim.
    /// Deterministic from seed via ChaCha8Rng.
    signs: Vec<f32>,
    /// RNG seed for reproducibility
    seed: u64,
}

impl WhtRotation {
    /// Create a new randomized WHT rotation for the given dimension and seed.
    ///
    /// # Errors
    /// - `ZeroDimension` if `dim == 0`
    pub fn new(dim: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(TurboQuantError::ZeroDimension);
        }

        let padded_dim = dim.next_power_of_two();

        // Generate deterministic random sign vector
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..padded_dim)
            .map(|_| if rng.gen::<bool>() { 1.0_f32 } else { -1.0_f32 })
            .collect();

        Ok(Self { dim, padded_dim, signs, seed })
    }

    /// Return the original (unpadded) dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the seed used to generate this rotation.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Approximate heap bytes consumed by this rotation.
    pub fn size_bytes(&self) -> usize {
        self.signs.len() * core::mem::size_of::<f32>()
    }

    /// Apply the forward randomized WHT: D * H * x (sign-flip then WHT).
    ///
    /// Returns a new vector of length `self.dim()`.
    pub fn forward(&self, vector: &[f32]) -> Vec<f32> {
        assert_eq!(vector.len(), self.dim, "WhtRotation::forward: expected {} dims, got {}", self.dim, vector.len());

        // Zero-pad to power of 2
        let mut buf = vec![0.0_f32; self.padded_dim];
        buf[..self.dim].copy_from_slice(vector);

        // Step 1: Random sign flip (D * x)
        for (x, &s) in buf.iter_mut().zip(self.signs.iter()) {
            *x *= s;
        }

        // Step 2: Normalized in-place WHT
        fwht_normalized(&mut buf);

        // Truncate back to original dimension
        buf.truncate(self.dim);
        buf
    }

    /// Apply the inverse randomized WHT: H^{-1} * D^{-1} * x.
    ///
    /// Since the normalized WHT is self-inverse (H = H^{-1}) and the sign-flip
    /// diagonal is also self-inverse (D = D^{-1}), the inverse is: WHT then
    /// sign-flip. Note the order is reversed from the forward transform.
    pub fn inverse(&self, vector: &[f32]) -> Vec<f32> {
        assert_eq!(vector.len(), self.dim, "WhtRotation::inverse: expected {} dims, got {}", self.dim, vector.len());

        // Zero-pad to power of 2
        let mut buf = vec![0.0_f32; self.padded_dim];
        buf[..self.dim].copy_from_slice(vector);

        // Step 1: Normalized WHT (self-inverse)
        fwht_normalized(&mut buf);

        // Step 2: Sign flip (self-inverse)
        for (x, &s) in buf.iter_mut().zip(self.signs.iter()) {
            *x *= s;
        }

        // Truncate back to original dimension
        buf.truncate(self.dim);
        buf
    }

    /// Apply forward WHT in-place, writing result to `out`.
    /// Matches `StoredRotation::apply_slice` interface.
    #[inline]
    pub fn apply_slice(&self, slice: &[f32], out: &mut Vec<f32>) {
        let result = self.forward(slice);
        *out = result;
    }

    /// Apply inverse WHT in-place, writing result to `out`.
    /// Matches `StoredRotation::apply_inverse_slice` interface.
    #[inline]
    pub fn apply_inverse_slice(&self, slice: &[f32], out: &mut Vec<f32>) {
        let result = self.inverse(slice);
        *out = result;
    }
}

// ---------------------------------------------------------------------------
// Core WHT algorithms
// ---------------------------------------------------------------------------

/// Fast Walsh-Hadamard Transform, in-place, O(n log n).
///
/// `data` length MUST be a power of 2. This is a precondition enforced by
/// `WhtRotation::new` which pads to the next power of 2.
///
/// Uses the standard iterative butterfly decomposition: at each level,
/// pairs of elements are combined with addition and subtraction.
pub(crate) fn fwht_in_place(data: &mut [f32]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "fwht_in_place: length {} must be power of 2", n);

    let mut half = 1;
    while half < n {
        let mut i = 0;
        while i < n {
            for j in i..i + half {
                let a = data[j];
                let b = data[j + half];
                data[j] = a + b;           // butterfly: add
                data[j + half] = a - b;    // butterfly: subtract
            }
            i += half * 2;
        }
        half *= 2;
    }
}

/// Normalized Fast Walsh-Hadamard Transform.
///
/// Multiplies by `1/sqrt(n)` after the raw WHT to make the transform
/// orthonormal: `||WHT(x)||₂ = ||x||₂`. The normalized WHT is its own
/// inverse: `WHT(WHT(x)) = x`.
pub(crate) fn fwht_normalized(data: &mut [f32]) {
    fwht_in_place(data);
    let scale = 1.0 / crate::compat::math::sqrtf(data.len() as f32);
    for x in data.iter_mut() {
        *x *= scale;
    }
}

// ---------------------------------------------------------------------------
// RotationStrategy trait implementation
// ---------------------------------------------------------------------------

impl crate::traits::RotationStrategy for WhtRotation {
    fn rotate(&self, vector: &[f32]) -> Vec<f32> {
        self.forward(vector)
    }

    fn rotate_inverse(&self, vector: &[f32]) -> Vec<f32> {
        self.inverse(vector)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

impl PartialEq for WhtRotation {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.seed == other.seed
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_power_of_two() {
        // Known WHT of [1, 0, 0, 0] = [1, 1, 1, 1] (unnormalized)
        let mut data = [1.0_f32, 0.0, 0.0, 0.0];
        fwht_in_place(&mut data);
        assert_eq!(data, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_fwht_length_one() {
        // Single-element WHT is identity
        let mut data = [42.0_f32];
        fwht_in_place(&mut data);
        assert_eq!(data, [42.0]);
    }

    #[test]
    fn test_fwht_all_ones() {
        // WHT of [1, 1, 1, 1] = [4, 0, 0, 0] (unnormalized)
        let mut data = [1.0_f32, 1.0, 1.0, 1.0];
        fwht_in_place(&mut data);
        assert_eq!(data, [4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalized_self_inverse() {
        // Normalized WHT applied twice should return the original vector
        let original = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        fwht_normalized(&mut data);
        fwht_normalized(&mut data);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "self-inverse failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_normalized_preserves_norm() {
        // Orthonormal: ||WHT(x)||₂ = ||x||₂
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let original_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut transformed = data.clone();
        fwht_normalized(&mut transformed);
        let transformed_norm: f32 = transformed.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (original_norm - transformed_norm).abs() < 1e-5,
            "norm not preserved: {} vs {}",
            original_norm,
            transformed_norm
        );
    }

    #[test]
    fn test_wht_rotation_round_trip() {
        let wht = WhtRotation::new(128, 42).unwrap();
        let v: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let rotated = wht.forward(&v);
        let recovered = wht.inverse(&rotated);
        for (a, b) in v.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "round-trip failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_wht_rotation_non_power_of_two_construction() {
        // dim=100 can be constructed (padded to 128 internally)
        let wht = WhtRotation::new(100, 42).unwrap();
        assert_eq!(wht.dim(), 100);
        assert_eq!(wht.padded_dim, 128);

        // Forward transform produces output of correct length
        let v: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let rotated = wht.forward(&v);
        assert_eq!(rotated.len(), 100);
    }

    #[test]
    fn test_wht_power_of_two_exact_round_trip() {
        // Power-of-2 dimensions have exact round-trip (no padding loss)
        for dim in [2, 4, 8, 16, 32, 64, 128, 256] {
            let wht = WhtRotation::new(dim, 42).unwrap();
            let v: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            let rotated = wht.forward(&v);
            let recovered = wht.inverse(&rotated);
            for (a, b) in v.iter().zip(recovered.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "pow2 round-trip failed at dim={}: {} vs {}",
                    dim, a, b
                );
            }
        }
    }

    #[test]
    fn test_wht_deterministic() {
        let wht1 = WhtRotation::new(64, 99).unwrap();
        let wht2 = WhtRotation::new(64, 99).unwrap();
        let v: Vec<f32> = (0..64).map(|i| i as f32).collect();
        assert_eq!(wht1.forward(&v), wht2.forward(&v));
    }

    #[test]
    fn test_wht_different_seeds_differ() {
        let wht1 = WhtRotation::new(64, 1).unwrap();
        let wht2 = WhtRotation::new(64, 2).unwrap();
        let v: Vec<f32> = (0..64).map(|i| i as f32).collect();
        assert_ne!(wht1.forward(&v), wht2.forward(&v));
    }

    #[test]
    fn test_zero_dimension_error() {
        assert!(matches!(WhtRotation::new(0, 42), Err(TurboQuantError::ZeroDimension)));
    }

    #[test]
    fn test_memory_much_smaller_than_haar() {
        let wht = WhtRotation::new(768, 42).unwrap();
        let haar = crate::rotation::StoredRotation::new(768, 42).unwrap();
        // WHT: 1024 × 4 = 4096 bytes (padded to pow2=1024)
        // Haar: 768² × 4 = 2,359,296 bytes
        assert!(wht.size_bytes() < haar.size_bytes() / 100,
            "WHT ({}) should be >100x smaller than Haar ({})",
            wht.size_bytes(), haar.size_bytes()
        );
    }

    #[test]
    fn test_rotation_strategy_trait() {
        use crate::traits::RotationStrategy;
        let wht = WhtRotation::new(64, 42).unwrap();
        let v: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let rotated = wht.rotate(&v);
        let recovered = wht.rotate_inverse(&rotated);
        for (a, b) in v.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "trait round-trip: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_apply_slice_interface() {
        let wht = WhtRotation::new(32, 42).unwrap();
        let v: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut out = Vec::new();
        wht.apply_slice(&v, &mut out);
        assert_eq!(out.len(), 32);
        let mut recovered = Vec::new();
        wht.apply_inverse_slice(&out, &mut recovered);
        for (a, b) in v.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "apply_slice round-trip: {} vs {}", a, b);
        }
    }
}
