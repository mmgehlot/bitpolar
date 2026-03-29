//! Haar-distributed orthogonal rotation matrix.
//!
//! `StoredRotation` pre-computes and stores a random orthogonal matrix generated
//! via QR decomposition of a Gaussian matrix, producing a Haar-distributed
//! (uniformly random) rotation. This spreads energy uniformly across all
//! coordinates before polar encoding, improving quantization quality.

use crate::error::{Result, TurboQuantError};

/// A stored, pre-computed Haar-distributed orthogonal rotation matrix.
///
/// Constructed once from `(dim, seed)` and reused for all encode/decode
/// operations. The matrix is deterministic: same `(dim, seed)` always
/// produces the same rotation.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct StoredRotation {
    /// Vector dimension
    dim: usize,
    /// Random seed used to generate this rotation
    seed: u64,
    /// Flat row-major storage of the dim×dim orthogonal matrix
    data: Vec<f32>,
}

impl StoredRotation {
    /// Create a new Haar-distributed rotation for the given dimension and seed.
    ///
    /// # Errors
    /// - `ZeroDimension` if `dim == 0`
    pub fn new(dim: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(TurboQuantError::ZeroDimension);
        }
        // Generate a Haar-distributed rotation via QR decomposition.
        // We use a simple construction: fill with Gaussian random values,
        // then perform Gram-Schmidt orthogonalisation (or use nalgebra QR).
        let data = Self::generate_rotation(dim, seed);

        Ok(Self { dim, seed, data })
    }

    /// Return the dimension this rotation operates on.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the seed used to generate this rotation.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Apply the rotation `R * v` in-place, writing the result into `out`.
    ///
    /// `slice` must have length `self.dim()`.
    #[inline]
    pub fn apply_slice(&self, slice: &[f32], out: &mut Vec<f32>) {
        let d = self.dim;
        debug_assert_eq!(
            slice.len(),
            d,
            "apply_slice: slice.len() must equal self.dim"
        );
        debug_assert_eq!(
            self.data.len(),
            d * d,
            "apply_slice: data.len() must equal dim*dim"
        );
        // Defensive: clear output to zeros on dimension mismatch (prevents stale data)
        out.clear();
        out.resize(d, 0.0);
        if slice.len() != d {
            return;
        }
        for (i, o) in out.iter_mut().enumerate().take(d) {
            *o = self.data[i * d..i * d + d]
                .iter()
                .zip(slice.iter())
                .map(|(m, v)| m * v)
                .sum();
        }
    }

    /// Apply the inverse rotation `R^T * v` (transpose = inverse for orthogonal).
    ///
    /// `slice` must have length `self.dim()`.
    #[inline]
    pub fn apply_inverse_slice(&self, slice: &[f32], out: &mut Vec<f32>) {
        let d = self.dim;
        debug_assert_eq!(
            slice.len(),
            d,
            "apply_inverse_slice: slice.len() must equal self.dim"
        );
        debug_assert_eq!(
            self.data.len(),
            d * d,
            "apply_inverse_slice: data.len() must equal dim*dim"
        );
        // Defensive: clear output to zeros on dimension mismatch (prevents stale data)
        out.clear();
        out.resize(d, 0.0);
        if slice.len() != d {
            return;
        }
        // Transpose multiplication: out[i] = sum_j data[j*d+i] * slice[j]
        for (i, o) in out.iter_mut().enumerate().take(d) {
            *o = slice
                .iter()
                .enumerate()
                .take(d)
                .map(|(j, &s)| self.data[j * d + i] * s)
                .sum();
        }
    }

    /// Approximate heap bytes consumed by this rotation matrix.
    pub fn size_bytes(&self) -> usize {
        self.data.len() * core::mem::size_of::<f32>()
    }

    /// Return the rotation as a nalgebra `DMatrix<f32>` (requires `std` feature).
    #[cfg(feature = "std")]
    pub fn matrix(&self) -> nalgebra::DMatrix<f32> {
        nalgebra::DMatrix::from_row_slice(self.dim, self.dim, &self.data)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Generate a dim×dim Haar-distributed orthogonal matrix stored row-major.
    ///
    /// Uses a seeded ChaCha RNG for reproducibility. On `std` builds we use
    /// nalgebra's QR decomposition; on `no_std` we fall back to a simple
    /// random-sign diagonal (rotation quality is reduced but still valid).
    fn generate_rotation(dim: usize, seed: u64) -> Vec<f32> {
        #[cfg(feature = "std")]
        {
            use nalgebra::DMatrix;
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rand_distr::StandardNormal;

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let dist = StandardNormal;

            // Fill dim×dim matrix with N(0,1) entries.
            let raw: Vec<f32> = (0..dim * dim)
                .map(|_| {
                    <StandardNormal as rand_distr::Distribution<f64>>::sample(&dist, &mut rng)
                        as f32
                })
                .collect();

            let m = DMatrix::from_row_slice(dim, dim, &raw);
            // QR decomposition gives a Haar-distributed rotation Q.
            let qr = m.qr();
            let q = qr.q();

            // Ensure positive determinant (QR can produce reflections).
            let det_sign = if q.determinant() >= 0.0 {
                1.0_f32
            } else {
                -1.0_f32
            };
            let mut data = vec![0.0_f32; dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    data[i * dim + j] = q[(i, j)] * det_sign;
                }
            }
            data
        }

        #[cfg(not(feature = "std"))]
        {
            // Fallback: random-sign diagonal matrix (not truly Haar but valid).
            use rand::Rng;
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut data = vec![0.0_f32; dim * dim];
            for i in 0..dim {
                let sign: f32 = if rng.gen::<bool>() { 1.0 } else { -1.0 };
                data[i * dim + i] = sign;
            }
            data
        }
    }
}

impl PartialEq for StoredRotation {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.seed == other.seed
    }
}

// ---------------------------------------------------------------------------
// RotationStrategy impl
// ---------------------------------------------------------------------------

impl crate::traits::RotationStrategy for StoredRotation {
    fn rotate(&self, vector: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dim);
        self.apply_slice(vector, &mut out);
        out
    }

    fn rotate_inverse(&self, vector: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dim);
        self.apply_inverse_slice(vector, &mut out);
        out
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_dimension_error() {
        assert!(matches!(
            StoredRotation::new(0, 42),
            Err(TurboQuantError::ZeroDimension)
        ));
    }

    #[test]
    fn test_basic_construction() {
        let r = StoredRotation::new(4, 42).unwrap();
        assert_eq!(r.dim(), 4);
        assert_eq!(r.seed(), 42);
    }

    #[test]
    fn test_partial_eq() {
        let a = StoredRotation::new(4, 42).unwrap();
        let b = StoredRotation::new(4, 42).unwrap();
        let c = StoredRotation::new(4, 99).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_round_trip_identity() {
        let r = StoredRotation::new(4, 7).unwrap();
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut rotated = Vec::new();
        let mut reconstructed = Vec::new();
        r.apply_slice(&v, &mut rotated);
        r.apply_inverse_slice(&rotated, &mut reconstructed);
        for (a, b) in v.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-4, "round-trip failed: {a} vs {b}");
        }
    }
}
