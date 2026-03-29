//! Compression statistics and quality metrics.
//!
//! Provides aggregate statistics for batches of compressed vectors
//! and per-observation quality tracking for production monitoring.

/// Aggregate statistics for a batch of compressed vectors.
///
/// Use [`crate::TurboQuantizer::batch_stats`] to compute these from a
/// collection of codes.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct BatchStats {
    /// Number of vectors in the batch
    pub count: usize,
    /// Total original size in bytes (f32 x dim x count)
    pub original_bytes: usize,
    /// Total compressed size in bytes (sum of code sizes)
    pub compressed_bytes: usize,
    /// Compression ratio: original_bytes / compressed_bytes
    pub compression_ratio: f64,
    /// Average bits per scalar value: (compressed_bytes * 8) / (count * dim)
    pub bits_per_value: f64,
}

/// Quality metrics from the [`crate::distortion::DistortionTracker`].
///
/// These metrics are designed for integration into health check endpoints
/// and monitoring dashboards (Prometheus, Grafana, OpenTelemetry).
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct DistortionMetrics {
    /// Exponential moving average of squared estimation error (MSE).
    ///
    /// Lower is better. A healthy quantizer should have MSE well below
    /// the theoretical upper bound for its bit width.
    pub mse: f64,
    /// Exponential moving average of signed estimation error (bias).
    ///
    /// Should be ~0 for an unbiased estimator. Persistent non-zero bias
    /// indicates a bug or data distribution issue.
    pub bias: f64,
    /// Total number of observations recorded.
    pub samples: u64,
    /// Whether the tracker considers the quantizer healthy.
    ///
    /// `false` when MSE exceeds the configured threshold.
    pub healthy: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_stats_fields() {
        let stats = BatchStats {
            count: 100,
            original_bytes: 51200,
            compressed_bytes: 12800,
            compression_ratio: 4.0,
            bits_per_value: 8.0,
        };
        assert_eq!(stats.count, 100);
        assert!((stats.compression_ratio - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_distortion_metrics_fields() {
        let metrics = DistortionMetrics {
            mse: 0.01,
            bias: -0.001,
            samples: 5000,
            healthy: true,
        };
        assert!(metrics.healthy);
        assert!(metrics.mse < 0.1);
    }
}
