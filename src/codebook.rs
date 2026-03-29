//! Lloyd-Max optimal scalar quantizer for the N(0,1) distribution.
//!
//! Pre-computes centroids and decision boundaries for a B-bit quantizer
//! calibrated to a standard Gaussian. After rotation, each scalar is
//! approximately Gaussian, so this codebook minimises distortion.

use crate::error::{Result, TurboQuantError};

/// A Lloyd-Max codebook optimised for N(0,1) scalars.
///
/// Stores `2^bits` centroids and the `2^bits - 1` decision boundaries
/// that partition the real line into `2^bits` intervals.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub(crate) struct LloydMaxCodebook {
    /// Bit width (1..=16). Retained for diagnostics and serde round-tripping.
    #[allow(dead_code)]
    bits: u8,
    /// Reconstruction centroids (length 2^bits)
    centroids: Vec<f32>,
    /// Decision boundaries (length 2^bits - 1)
    boundaries: Vec<f32>,
}

impl LloydMaxCodebook {
    /// Compute a Lloyd-Max codebook for the given bit width.
    ///
    /// # Errors
    /// - `InvalidBitWidth` if `bits` is 0 or > 16
    pub(crate) fn compute(bits: u8) -> Result<Self> {
        if bits == 0 || bits > 16 {
            return Err(TurboQuantError::InvalidBitWidth(bits));
        }

        let levels = 1usize << bits;

        // Simple uniform initialisation on [-3σ, 3σ], then a few Lloyd-Max
        // iterations to converge to the Gaussian-optimal codebook.
        let sigma = 3.0_f32;
        let step = (2.0 * sigma) / levels as f32;

        // Initial centroids at midpoints of uniform intervals.
        let mut centroids: Vec<f32> = (0..levels)
            .map(|i| -sigma + (i as f32 + 0.5) * step)
            .collect();

        // Lloyd-Max iterations (30 is sufficient for convergence).
        for _ in 0..30 {
            // Update boundaries: midpoints between adjacent centroids.
            let boundaries: Vec<f32> = centroids.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

            // Update centroids: E[X | X in interval] for N(0,1).
            let mut new_centroids = vec![0.0_f32; levels];
            let lo_bounds: Vec<f32> = core::iter::once(f32::NEG_INFINITY)
                .chain(boundaries.iter().copied())
                .collect();
            let hi_bounds: Vec<f32> = boundaries
                .iter()
                .copied()
                .chain(core::iter::once(f32::INFINITY))
                .collect();

            for k in 0..levels {
                new_centroids[k] = gaussian_conditional_mean(lo_bounds[k], hi_bounds[k]);
            }

            centroids = new_centroids;
        }

        // Recompute final boundaries.
        let boundaries: Vec<f32> = centroids.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

        Ok(Self {
            bits,
            centroids,
            boundaries,
        })
    }

    /// Quantize a scalar value to its nearest codebook index.
    ///
    /// Returns an index in `0..2^bits`.
    #[inline]
    pub(crate) fn quantize(&self, value: f32) -> u16 {
        // Binary-search the decision boundaries.
        let idx = self.boundaries.partition_point(|&b| value > b);
        (idx.min(u16::MAX as usize)) as u16
    }

    /// Dequantize a codebook index back to the centroid value.
    #[inline]
    pub(crate) fn dequantize(&self, index: u16) -> f32 {
        let i = (index as usize).min(self.centroids.len().saturating_sub(1));
        self.centroids[i]
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Compute E[X | lo < X ≤ hi] for X ~ N(0,1).
///
/// Uses the formula: (φ(lo) − φ(hi)) / (Φ(hi) − Φ(lo))
/// where φ is the standard Gaussian PDF and Φ is the CDF.
fn gaussian_conditional_mean(lo: f32, hi: f32) -> f32 {
    let phi_lo = if lo.is_finite() {
        standard_normal_pdf(lo)
    } else {
        0.0
    };
    let phi_hi = if hi.is_finite() {
        standard_normal_pdf(hi)
    } else {
        0.0
    };
    let p_lo = if lo.is_finite() {
        standard_normal_cdf(lo)
    } else {
        0.0
    };
    let p_hi = if hi.is_finite() {
        standard_normal_cdf(hi)
    } else {
        1.0
    };
    let denom = p_hi - p_lo;
    if denom.abs() < 1e-12 {
        // Interval probability is negligible; return midpoint.
        let mid = if lo.is_finite() && hi.is_finite() {
            (lo + hi) / 2.0
        } else if lo.is_finite() {
            lo + 3.0
        } else if hi.is_finite() {
            hi - 3.0
        } else {
            0.0
        };
        return mid;
    }
    (phi_lo - phi_hi) / denom
}

/// Standard Gaussian PDF: φ(x) = exp(-x²/2) / sqrt(2π)
#[inline]
fn standard_normal_pdf(x: f32) -> f32 {
    let inv_sqrt_2pi = 0.398_942_3_f32; // 1/sqrt(2π)
    inv_sqrt_2pi * crate::compat::math::expf(-0.5 * x * x)
}

/// Standard Gaussian CDF approximation (Abramowitz & Stegun 26.2.17).
#[inline]
fn standard_normal_cdf(x: f32) -> f32 {
    // Use erfc approximation.
    let t = x / core::f32::consts::SQRT_2;
    0.5 * (1.0 + erf_approx(t))
}

/// Polynomial approximation of erf(x) (maximum error < 1.5e-7).
fn erf_approx(x: f32) -> f32 {
    // Rational approximation for erf valid for all x.
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_72_f32;
    let a3 = 1.421_413_8_f32;
    let a4 = -1.453_152_1_f32;
    let a5 = 1.061_405_4_f32;
    let p = 0.327_591_1_f32;
    let sign = if x < 0.0 { -1.0_f32 } else { 1.0_f32 };
    let x = crate::compat::math::fabsf(x);
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * crate::compat::math::expf(-x * x);
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_bits_error() {
        assert!(matches!(
            LloydMaxCodebook::compute(0),
            Err(TurboQuantError::InvalidBitWidth(0))
        ));
    }

    #[test]
    fn test_too_many_bits_error() {
        assert!(matches!(
            LloydMaxCodebook::compute(17),
            Err(TurboQuantError::InvalidBitWidth(17))
        ));
    }

    #[test]
    fn test_1bit_codebook() {
        let cb = LloydMaxCodebook::compute(1).unwrap();
        assert_eq!(cb.centroids.len(), 2);
        assert_eq!(cb.boundaries.len(), 1);
        // For N(0,1) at 1-bit the boundary should be near 0.
        assert!(
            cb.boundaries[0].abs() < 0.1,
            "boundary={}",
            cb.boundaries[0]
        );
    }

    #[test]
    fn test_quantize_dequantize_round_trip() {
        let cb = LloydMaxCodebook::compute(4).unwrap();
        // Values should round-trip to a nearby centroid.
        for v in [-2.0_f32, -1.0, 0.0, 1.0, 2.0] {
            let idx = cb.quantize(v);
            let recon = cb.dequantize(idx);
            assert!((v - recon).abs() < 1.0, "v={v} recon={recon}");
        }
    }
}
