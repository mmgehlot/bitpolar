//! Online distortion tracking for quality monitoring.
//!
//! [`DistortionTracker`] maintains exponential moving averages of MSE and bias
//! from observed (estimate, ground_truth) pairs. Use it in production to
//! detect quantization quality degradation without offline evaluation.

use crate::stats::DistortionMetrics;

/// Online distortion tracker using exponential moving averages.
///
/// Thread-safe only when access is externally synchronized; the tracker
/// is `Send` but not `Sync` (use `Mutex<DistortionTracker>` in async code).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-support", derive(serde::Serialize, serde::Deserialize))]
pub struct DistortionTracker {
    /// EMA smoothing factor (0 < alpha ≤ 1)
    alpha: f64,
    /// MSE threshold above which `is_healthy()` returns false
    mse_threshold: f64,
    /// Exponential moving average of squared error
    ema_mse: f64,
    /// Exponential moving average of signed error (bias)
    ema_bias: f64,
    /// Total number of observations recorded
    samples: u64,
}

impl DistortionTracker {
    /// Create a new `DistortionTracker`.
    ///
    /// # Arguments
    /// - `alpha` — EMA smoothing factor (0.0 < alpha ≤ 1.0).
    ///   Smaller values give more weight to history (e.g., `0.01` for slow adaptation).
    ///   Use `1.0` for no smoothing (each observation overwrites the state).
    /// - `mse_threshold` — MSE value above which [`is_healthy`](Self::is_healthy) returns `false`.
    pub fn new(alpha: f64, mse_threshold: f64) -> Self {
        let alpha = alpha.clamp(1e-6, 1.0);
        Self { alpha, mse_threshold, ema_mse: 0.0, ema_bias: 0.0, samples: 0 }
    }

    /// Record a single (estimated, ground_truth) pair.
    ///
    /// Updates the EMA of MSE and bias.
    pub fn observe(&mut self, estimated: f64, ground_truth: f64) {
        let error = estimated - ground_truth;
        let sq_error = error * error;

        if self.samples == 0 {
            // Bootstrap: initialise with the first observation.
            self.ema_mse = sq_error;
            self.ema_bias = error;
        } else {
            self.ema_mse = self.alpha * sq_error + (1.0 - self.alpha) * self.ema_mse;
            self.ema_bias = self.alpha * error + (1.0 - self.alpha) * self.ema_bias;
        }
        self.samples += 1;
    }

    /// Returns `true` when the running MSE is below the configured threshold.
    ///
    /// Always returns `true` before any observations have been recorded.
    pub fn is_healthy(&self) -> bool {
        if self.samples == 0 {
            return true;
        }
        self.ema_mse <= self.mse_threshold
    }

    /// Return a snapshot of the current distortion metrics.
    pub fn metrics(&self) -> DistortionMetrics {
        DistortionMetrics {
            mse: self.ema_mse,
            bias: self.ema_bias,
            samples: self.samples,
            healthy: self.is_healthy(),
        }
    }

    /// Reset the tracker to its initial state (zero observations).
    pub fn reset(&mut self) {
        self.ema_mse = 0.0;
        self.ema_bias = 0.0;
        self.samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthy_before_observations() {
        let tracker = DistortionTracker::new(0.1, 1.0);
        assert!(tracker.is_healthy());
        assert_eq!(tracker.metrics().samples, 0);
    }

    #[test]
    fn test_healthy_after_zero_error() {
        let mut tracker = DistortionTracker::new(0.1, 1.0);
        for _ in 0..100 {
            tracker.observe(0.5, 0.5);
        }
        assert!(tracker.is_healthy());
        assert!((tracker.metrics().mse).abs() < 1e-9);
        assert!((tracker.metrics().bias).abs() < 1e-9);
    }

    #[test]
    fn test_unhealthy_after_large_errors() {
        let mut tracker = DistortionTracker::new(0.5, 0.1);
        for _ in 0..50 {
            tracker.observe(10.0, 0.0); // error = 10, sq_error = 100
        }
        assert!(!tracker.is_healthy());
        assert!(tracker.metrics().mse > 0.1);
    }

    #[test]
    fn test_bias_tracking() {
        let mut tracker = DistortionTracker::new(0.5, 1000.0);
        for _ in 0..200 {
            tracker.observe(1.0, 0.0); // constant positive bias
        }
        let m = tracker.metrics();
        assert!(m.bias > 0.5, "bias={}", m.bias);
    }

    #[test]
    fn test_reset() {
        let mut tracker = DistortionTracker::new(0.1, 1.0);
        tracker.observe(5.0, 0.0);
        tracker.reset();
        assert_eq!(tracker.metrics().samples, 0);
        assert!(tracker.is_healthy());
    }

    #[test]
    fn test_samples_count() {
        let mut tracker = DistortionTracker::new(0.1, 1.0);
        for i in 0..42 {
            tracker.observe(i as f64, 0.0);
        }
        assert_eq!(tracker.metrics().samples, 42);
    }
}
