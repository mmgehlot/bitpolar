//! Prometheus-compatible metrics export for BitPolar operations.
//!
//! Provides metric collection for encode/decode/IP latencies, compression
//! ratios, and distortion quality. Metrics are exposed as structured data
//! that can be formatted for Prometheus, OpenTelemetry, or any monitoring system.
//!
//! Enable with the `tracing-support` feature flag.
//!
//! # Usage
//!
//! ```rust,ignore
//! use bitpolar::metrics::MetricsCollector;
//!
//! let metrics = MetricsCollector::new();
//! metrics.record_encode(dim, latency_us);
//! metrics.record_ip_estimate(dim, latency_us);
//!
//! // Export as Prometheus text format
//! let prometheus_text = metrics.to_prometheus();
//! ```

use core::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe metrics collector for BitPolar operations.
///
/// Uses atomic counters for lock-free metric collection.
/// Safe to share across threads via `Arc<MetricsCollector>`.
pub struct MetricsCollector {
    /// Total encode operations
    pub encode_count: AtomicU64,
    /// Total encode microseconds (sum for average calculation)
    pub encode_us_total: AtomicU64,
    /// Total decode operations
    pub decode_count: AtomicU64,
    /// Total decode microseconds
    pub decode_us_total: AtomicU64,
    /// Total inner product estimations
    pub ip_count: AtomicU64,
    /// Total IP estimation microseconds
    pub ip_us_total: AtomicU64,
    /// Total bytes compressed (original)
    pub bytes_original: AtomicU64,
    /// Total bytes after compression
    pub bytes_compressed: AtomicU64,
}

impl MetricsCollector {
    /// Create a new metrics collector with all counters at zero.
    pub fn new() -> Self {
        Self {
            encode_count: AtomicU64::new(0),
            encode_us_total: AtomicU64::new(0),
            decode_count: AtomicU64::new(0),
            decode_us_total: AtomicU64::new(0),
            ip_count: AtomicU64::new(0),
            ip_us_total: AtomicU64::new(0),
            bytes_original: AtomicU64::new(0),
            bytes_compressed: AtomicU64::new(0),
        }
    }

    /// Record an encode operation.
    pub fn record_encode(&self, original_bytes: usize, compressed_bytes: usize, latency_us: u64) {
        self.encode_count.fetch_add(1, Ordering::Relaxed);
        self.encode_us_total
            .fetch_add(latency_us, Ordering::Relaxed);
        self.bytes_original
            .fetch_add(original_bytes as u64, Ordering::Relaxed);
        self.bytes_compressed
            .fetch_add(compressed_bytes as u64, Ordering::Relaxed);
    }

    /// Record a decode operation.
    pub fn record_decode(&self, latency_us: u64) {
        self.decode_count.fetch_add(1, Ordering::Relaxed);
        self.decode_us_total
            .fetch_add(latency_us, Ordering::Relaxed);
    }

    /// Record an inner product estimation.
    pub fn record_ip_estimate(&self, latency_us: u64) {
        self.ip_count.fetch_add(1, Ordering::Relaxed);
        self.ip_us_total.fetch_add(latency_us, Ordering::Relaxed);
    }

    /// Export metrics as Prometheus text exposition format.
    pub fn to_prometheus(&self) -> String {
        let encode_count = self.encode_count.load(Ordering::Relaxed);
        let encode_us = self.encode_us_total.load(Ordering::Relaxed);
        let decode_count = self.decode_count.load(Ordering::Relaxed);
        let _decode_us = self.decode_us_total.load(Ordering::Relaxed);
        let ip_count = self.ip_count.load(Ordering::Relaxed);
        let ip_us = self.ip_us_total.load(Ordering::Relaxed);
        let bytes_orig = self.bytes_original.load(Ordering::Relaxed);
        let bytes_comp = self.bytes_compressed.load(Ordering::Relaxed);

        let mut out = String::new();

        out.push_str("# HELP bitpolar_encode_total Total encode operations\n");
        out.push_str("# TYPE bitpolar_encode_total counter\n");
        out.push_str(&format!("bitpolar_encode_total {}\n", encode_count));

        out.push_str(
            "# HELP bitpolar_encode_duration_us_total Total encode latency in microseconds\n",
        );
        out.push_str("# TYPE bitpolar_encode_duration_us_total counter\n");
        out.push_str(&format!(
            "bitpolar_encode_duration_us_total {}\n",
            encode_us
        ));

        if encode_count > 0 {
            out.push_str("# HELP bitpolar_encode_avg_us Average encode latency\n");
            out.push_str("# TYPE bitpolar_encode_avg_us gauge\n");
            out.push_str(&format!(
                "bitpolar_encode_avg_us {}\n",
                encode_us / encode_count
            ));
        }

        out.push_str("# HELP bitpolar_decode_total Total decode operations\n");
        out.push_str("# TYPE bitpolar_decode_total counter\n");
        out.push_str(&format!("bitpolar_decode_total {}\n", decode_count));

        out.push_str("# HELP bitpolar_ip_estimate_total Total inner product estimations\n");
        out.push_str("# TYPE bitpolar_ip_estimate_total counter\n");
        out.push_str(&format!("bitpolar_ip_estimate_total {}\n", ip_count));

        if ip_count > 0 {
            out.push_str("# HELP bitpolar_ip_estimate_avg_us Average IP estimation latency\n");
            out.push_str("# TYPE bitpolar_ip_estimate_avg_us gauge\n");
            out.push_str(&format!(
                "bitpolar_ip_estimate_avg_us {}\n",
                ip_us / ip_count
            ));
        }

        out.push_str("# HELP bitpolar_bytes_original_total Total original bytes processed\n");
        out.push_str("# TYPE bitpolar_bytes_original_total counter\n");
        out.push_str(&format!("bitpolar_bytes_original_total {}\n", bytes_orig));

        out.push_str("# HELP bitpolar_bytes_compressed_total Total compressed bytes produced\n");
        out.push_str("# TYPE bitpolar_bytes_compressed_total counter\n");
        out.push_str(&format!("bitpolar_bytes_compressed_total {}\n", bytes_comp));

        if bytes_comp > 0 {
            let ratio = bytes_orig as f64 / bytes_comp as f64;
            out.push_str("# HELP bitpolar_compression_ratio Current compression ratio\n");
            out.push_str("# TYPE bitpolar_compression_ratio gauge\n");
            out.push_str(&format!("bitpolar_compression_ratio {:.2}\n", ratio));
        }

        out
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.encode_count.store(0, Ordering::Relaxed);
        self.encode_us_total.store(0, Ordering::Relaxed);
        self.decode_count.store(0, Ordering::Relaxed);
        self.decode_us_total.store(0, Ordering::Relaxed);
        self.ip_count.store(0, Ordering::Relaxed);
        self.ip_us_total.store(0, Ordering::Relaxed);
        self.bytes_original.store(0, Ordering::Relaxed);
        self.bytes_compressed.store(0, Ordering::Relaxed);
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_export() {
        let m = MetricsCollector::new();
        m.record_encode(3072, 2332, 180);
        m.record_encode(3072, 2332, 200);
        m.record_ip_estimate(10);

        let prom = m.to_prometheus();
        assert!(prom.contains("bitpolar_encode_total 2"));
        assert!(prom.contains("bitpolar_ip_estimate_total 1"));
        assert!(prom.contains("bitpolar_compression_ratio"));
    }

    #[test]
    fn test_reset() {
        let m = MetricsCollector::new();
        m.record_encode(100, 50, 10);
        assert_eq!(m.encode_count.load(Ordering::Relaxed), 1);
        m.reset();
        assert_eq!(m.encode_count.load(Ordering::Relaxed), 0);
    }
}
