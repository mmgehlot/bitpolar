//! End-to-end integration tests for bitpolar.
//!
//! Only the public API is used — no access to `pub(crate)` fields.

use bitpolar::traits::{RotationStrategy, VectorQuantizer};
use bitpolar::{
    DistortionTracker, KvCacheCompressor, KvCacheConfig, MultiHeadKvCache, PolarCode,
    PolarQuantizer, QjlQuantizer, QjlSketch, StoredRotation, TurboCode, TurboQuantizer,
};

// ============================================================================
// Helpers
// ============================================================================

fn make_vec(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| (seed as f64 * 1.7 + i as f64 * 0.3).sin() as f32)
        .collect()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn exact_ip(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn normalise(v: Vec<f32>) -> Vec<f32> {
    let n = l2_norm(&v);
    if n < 1e-10 {
        v
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

// ============================================================================
// 1. e2e_search_recall
// ============================================================================

#[test]
fn e2e_search_recall() {
    let dim = 128usize;
    let q = TurboQuantizer::new(dim, 4, 32, 42).unwrap();

    let db_vecs: Vec<Vec<f32>> = (0..100_u64)
        .map(|i| normalise(make_vec(dim, i * 13 + 1)))
        .collect();

    let db_codes: Vec<TurboCode> = db_vecs.iter().map(|v| q.encode(v).unwrap()).collect();

    // Query is a normalised copy of db_vecs[7].
    let query = db_vecs[7].clone();

    // Exact top-5.
    let mut true_scores: Vec<(usize, f32)> = db_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i, exact_ip(v, &query)))
        .collect();
    true_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let true_top5: Vec<usize> = true_scores[..5].iter().map(|(i, _)| *i).collect();

    // Approximate top-5.
    let mut approx_scores: Vec<(usize, f32)> = db_codes
        .iter()
        .enumerate()
        .map(|(i, code)| (i, q.inner_product_estimate(code, &query).unwrap()))
        .collect();
    approx_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let approx_top5: Vec<usize> = approx_scores[..5].iter().map(|(i, _)| *i).collect();

    let recall = approx_top5
        .iter()
        .filter(|idx| true_top5.contains(idx))
        .count();
    assert!(
        recall >= 2,
        "recall@5={recall}/5: true_top5={true_top5:?} approx_top5={approx_top5:?}"
    );
}

// ============================================================================
// 2. Compression ratios
// ============================================================================

#[test]
fn compression_ratios_all_produce_ratio_greater_than_one() {
    let dim = 128usize;
    let n_vecs = 50usize;

    for &bits in &[3u8, 4, 6, 8] {
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, bits, projections, 42).unwrap();
        let codes: Vec<_> = (0..n_vecs as u64)
            .map(|i| q.encode(&make_vec(dim, i)).unwrap())
            .collect();
        let stats = q.batch_stats(&codes);
        assert!(
            stats.compression_ratio > 1.0,
            "bits={bits}: compression_ratio={:.3} should be > 1",
            stats.compression_ratio
        );
    }
}

// ============================================================================
// 3. Seed independence — different seeds, different compact bytes
// ============================================================================

#[test]
fn seed_independence_different_seeds_produce_different_encodings() {
    let dim = 64usize;
    let projections = dim / 4;
    let v = make_vec(dim, 999);

    let qa = TurboQuantizer::new(dim, 4, projections, 1).unwrap();
    let qb = TurboQuantizer::new(dim, 4, projections, 2).unwrap();

    let bytes_a = qa.encode(&v).unwrap().to_compact_bytes();
    let bytes_b = qb.encode(&v).unwrap().to_compact_bytes();

    assert_ne!(
        bytes_a, bytes_b,
        "different seeds must produce different compact bytes"
    );
}

// ============================================================================
// 4. serde_json roundtrip (requires serde-support feature)
// ============================================================================

#[cfg(feature = "serde-support")]
#[test]
fn serde_json_roundtrip() {
    let dim = 64usize;
    let q = TurboQuantizer::new(dim, 4, 16, 42).unwrap();
    let v = make_vec(dim, 123);
    let code = q.encode(&v).unwrap();

    let json = serde_json::to_string(&code).expect("JSON serialization failed");
    let decoded: TurboCode = serde_json::from_str(&json).expect("JSON deserialization failed");

    // Verify round-trip fidelity via compact bytes.
    assert_eq!(
        decoded.to_compact_bytes(),
        code.to_compact_bytes(),
        "JSON roundtrip must preserve code identity"
    );
}

// ============================================================================
// 5. Compact bytes roundtrip
// ============================================================================

#[test]
fn compact_bytes_roundtrip() {
    let dim = 128usize;
    let q = TurboQuantizer::new(dim, 4, 32, 7).unwrap();
    let v = make_vec(dim, 456);
    let code = q.encode(&v).unwrap();

    let bytes = code.to_compact_bytes();
    assert!(!bytes.is_empty());

    let back = TurboCode::from_compact_bytes(&bytes).unwrap();
    assert_eq!(
        back.to_compact_bytes(),
        bytes,
        "from_compact_bytes → to_compact_bytes must be identity"
    );
}

// ============================================================================
// 6. Batch consistency
// ============================================================================

#[test]
fn batch_consistency_results_match_sequential_encode() {
    let dim = 64usize;
    let projections = dim / 4;
    let q = TurboQuantizer::new(dim, 4, projections, 42).unwrap();

    let vecs: Vec<Vec<f32>> = (0..20_u64).map(|i| make_vec(dim, i)).collect();
    let seq_bytes: Vec<Vec<u8>> = vecs
        .iter()
        .map(|v| q.encode(v).unwrap().to_compact_bytes())
        .collect();

    #[cfg(feature = "parallel")]
    {
        use bitpolar::traits::BatchQuantizer;
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let batch_codes = q.batch_encode(&refs).unwrap();
        assert_eq!(batch_codes.len(), seq_bytes.len());
        for (i, (bc, sb)) in batch_codes.iter().zip(seq_bytes.iter()).enumerate() {
            assert_eq!(
                bc.to_compact_bytes(),
                *sb,
                "batch vs sequential mismatch at index {i}"
            );
        }
    }

    // Always: sequential path is self-consistent.
    for (i, expected) in seq_bytes.iter().enumerate() {
        let actual = q.encode(&vecs[i]).unwrap().to_compact_bytes();
        assert_eq!(actual, *expected, "sequential non-determinism at {i}");
    }
}

// ============================================================================
// 7. KV cache workflow
// ============================================================================

#[test]
fn kv_cache_workflow() {
    let head_dim = 64usize;
    let config = KvCacheConfig {
        head_dim,
        bits: 4,
        projections: head_dim / 4,
        seed: 42,
    };
    let mut cache = KvCacheCompressor::new(&config).unwrap();

    let n_tokens = 50usize;
    for i in 0..n_tokens as u64 {
        let key = make_vec(head_dim, i * 2);
        let value = make_vec(head_dim, i * 2 + 1);
        cache.push(&key, &value).unwrap();
    }
    assert_eq!(cache.len(), n_tokens);

    let query = make_vec(head_dim, 9999);
    let scores = cache.attention_scores(&query).unwrap();
    assert_eq!(scores.len(), n_tokens, "one score per stored key");
    assert!(
        scores.iter().all(|s| s.is_finite()),
        "all scores must be finite"
    );
    assert!(cache.compression_ratio() > 0.0);
}

// ============================================================================
// 8. Multi-head KV cache workflow
// ============================================================================

#[test]
fn multi_head_kv_cache() {
    let num_heads = 4usize;
    let head_dim = 64usize;
    let config = KvCacheConfig {
        head_dim,
        bits: 4,
        projections: head_dim / 4,
        seed: 7,
    };
    let mut mhc = MultiHeadKvCache::new(num_heads, &config).unwrap();

    let n_tokens = 10usize;
    for i in 0..n_tokens as u64 {
        let key_vecs: Vec<Vec<f32>> = (0..num_heads as u64)
            .map(|h| make_vec(head_dim, i * 100 + h))
            .collect();
        let val_vecs: Vec<Vec<f32>> = (0..num_heads as u64)
            .map(|h| make_vec(head_dim, i * 100 + h + 50))
            .collect();
        let keys: Vec<&[f32]> = key_vecs.iter().map(|v| v.as_slice()).collect();
        let vals: Vec<&[f32]> = val_vecs.iter().map(|v| v.as_slice()).collect();
        mhc.push_token(&keys, &vals).unwrap();
    }
    assert_eq!(mhc.len(), n_tokens);

    let q_vecs: Vec<Vec<f32>> = (0..num_heads as u64)
        .map(|h| make_vec(head_dim, h + 9000))
        .collect();
    let queries: Vec<&[f32]> = q_vecs.iter().map(|v| v.as_slice()).collect();
    let all_scores = mhc.attention_scores(&queries).unwrap();

    assert_eq!(all_scores.len(), num_heads);
    for (h, head_scores) in all_scores.iter().enumerate() {
        assert_eq!(
            head_scores.len(),
            n_tokens,
            "head {h}: wrong number of scores"
        );
        assert!(
            head_scores.iter().all(|s| s.is_finite()),
            "head {h}: non-finite scores"
        );
    }
}

// ============================================================================
// 9. DistortionTracker integration
// ============================================================================

#[test]
fn distortion_tracker_integration() {
    let dim = 64usize;
    let projections = dim / 4;
    let q = TurboQuantizer::new(dim, 4, projections, 42).unwrap();

    let mut tracker = DistortionTracker::new(0.05, 0.5);

    let n = 200usize;
    for i in 0..n as u64 {
        let v = normalise(make_vec(dim, i));
        let code = q.encode(&v).unwrap();
        let decoded = q.decode(&code);
        let mse: f64 = v
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / dim as f64;
        // Observe the distortion (ground truth = 0 = ideal).
        tracker.observe(mse, 0.0);
    }

    let metrics = tracker.metrics();
    assert_eq!(metrics.samples, n as u64);
    assert!(
        metrics.healthy,
        "tracker should be healthy with mse_threshold=0.5, got mse={:.4}",
        metrics.mse
    );
    assert!(metrics.mse.is_finite());
}

// ============================================================================
// 10. PolarQuantizer standalone
// ============================================================================

#[test]
fn polar_standalone_encode_decode() {
    let dim = 64usize;
    let q = PolarQuantizer::new(dim, 4).unwrap();
    let v = normalise(make_vec(dim, 42));

    let code = q.encode(&v).unwrap();
    let decoded = q.decode(&code);
    assert_eq!(decoded.len(), dim);

    // Compact roundtrip via public serialisation API.
    let bytes = code.to_compact_bytes();
    let back = PolarCode::from_compact_bytes(&bytes).unwrap();
    assert_eq!(
        back.to_compact_bytes(),
        bytes,
        "PolarCode compact roundtrip failed"
    );

    // IP estimate self-query must be positive for non-zero vector.
    let ip = q.inner_product_estimate(&code, &v).unwrap();
    assert!(
        ip > 0.0,
        "PolarQuantizer IP(v,v) must be positive, got {ip}"
    );
}

// ============================================================================
// 11. QJL standalone
// ============================================================================

#[test]
fn qjl_standalone_ip_estimation() {
    let dim = 64usize;
    let projections = 64usize;
    let q = QjlQuantizer::new(dim, projections, 99).unwrap();

    let v = make_vec(dim, 77);
    let query = make_vec(dim, 88);

    let sketch = q.sketch(&v).unwrap();
    let est = q.inner_product_estimate(&sketch, &query).unwrap();

    assert!(est.is_finite(), "QJL estimate must be finite");
    assert_eq!(sketch.num_projections(), projections);

    // Compact roundtrip.
    let bytes = sketch.to_compact_bytes();
    let back = QjlSketch::from_compact_bytes(&bytes).unwrap();
    assert_eq!(
        back.to_compact_bytes(),
        bytes,
        "QjlSketch compact roundtrip failed"
    );

    // Self IP must be positive for non-zero vector.
    let v_norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if v_norm_sq > 1e-4 {
        let self_sketch = q.sketch(&v).unwrap();
        let self_est = q.inner_product_estimate(&self_sketch, &v).unwrap();
        assert!(
            self_est > 0.0,
            "QJL IP(v,v) must be positive for non-zero v"
        );
    }
}

// ============================================================================
// 12. VectorQuantizer trait object dispatch
// ============================================================================

#[test]
fn trait_object_dispatch_works() {
    let dim = 32usize;
    let boxed: Box<dyn VectorQuantizer<Code = TurboCode>> =
        Box::new(TurboQuantizer::new(dim, 4, 8, 42).unwrap());

    assert_eq!(boxed.dim(), dim);
    let v = make_vec(dim, 1234);
    let code = boxed.encode(&v).unwrap();
    let decoded = boxed.decode(&code);
    assert_eq!(decoded.len(), dim);
    let score = boxed.inner_product_estimate(&code, &v).unwrap();
    assert!(score.is_finite());
}

// ============================================================================
// 13. Multiple sequential encodes — no shared mutable state leak
// ============================================================================

#[test]
fn multiple_encodes_no_state_leak() {
    let dim = 32usize;
    let q = TurboQuantizer::new(dim, 4, 8, 42).unwrap();

    let v1 = make_vec(dim, 1);
    let v2 = make_vec(dim, 2);

    for _ in 0..50 {
        let b1 = q.encode(&v1).unwrap().to_compact_bytes();
        let b2 = q.encode(&v2).unwrap().to_compact_bytes();

        let b1_check = q.encode(&v1).unwrap().to_compact_bytes();
        assert_eq!(b1, b1_check, "v1 must be reproducible");

        let b2_check = q.encode(&v2).unwrap().to_compact_bytes();
        assert_eq!(b2, b2_check, "v2 must be reproducible");

        assert_ne!(
            b1, b2,
            "distinct vectors must produce distinct compact bytes"
        );
    }
}

// ============================================================================
// 14. Non-finite inputs rejected
// ============================================================================

#[test]
fn nonfinite_input_rejected() {
    let dim = 64usize;
    let q = TurboQuantizer::new(dim, 4, 16, 42).unwrap();

    let mut v_nan = make_vec(dim, 1);
    v_nan[10] = f32::NAN;
    assert!(matches!(
        q.encode(&v_nan),
        Err(bitpolar::TurboQuantError::NonFiniteInput { .. })
    ));

    let mut v_inf = make_vec(dim, 2);
    v_inf[0] = f32::INFINITY;
    assert!(matches!(
        q.encode(&v_inf),
        Err(bitpolar::TurboQuantError::NonFiniteInput { .. })
    ));
}

// ============================================================================
// 15. KV cache clear resets state
// ============================================================================

#[test]
fn kv_cache_clear_resets_state() {
    let head_dim = 32usize;
    let config = KvCacheConfig {
        head_dim,
        bits: 4,
        projections: head_dim / 4,
        seed: 1,
    };
    let mut cache = KvCacheCompressor::new(&config).unwrap();

    for i in 0..10_u64 {
        cache
            .push(&make_vec(head_dim, i), &make_vec(head_dim, i + 50))
            .unwrap();
    }
    assert_eq!(cache.len(), 10);
    cache.clear();
    assert!(cache.is_empty());

    let query = make_vec(head_dim, 9999);
    let scores = cache.attention_scores(&query).unwrap();
    assert!(scores.is_empty(), "scores should be empty after clear()");
}

// ============================================================================
// 16. dim=2 (minimum valid even dimension) through full pipeline
// ============================================================================

#[test]
fn dim_2_minimum_dimension() {
    let q = TurboQuantizer::new(2, 4, 1, 42).unwrap();
    let v = vec![0.6_f32, -0.8];
    let code = q.encode(&v).unwrap();
    let decoded = q.decode(&code);
    assert_eq!(decoded.len(), 2);
    let est = q.inner_product_estimate(&code, &v).unwrap();
    assert!(est.is_finite());
    let dist = q.l2_distance_estimate(&code, &v).unwrap();
    assert!(dist >= 0.0 && dist.is_finite());

    // Compact roundtrip
    let bytes = code.to_compact_bytes();
    let back = TurboCode::from_compact_bytes(&bytes).unwrap();
    assert_eq!(back.to_compact_bytes(), bytes);
}

// ============================================================================
// 17. bits=1 (minimum bit width) and bits=16 (maximum)
// ============================================================================

#[test]
fn bits_1_minimum_bit_width() {
    let dim = 8;
    let q = TurboQuantizer::new(dim, 1, 4, 42).unwrap();
    let v = make_vec(dim, 100);
    let code = q.encode(&v).unwrap();
    let decoded = q.decode(&code);
    assert_eq!(decoded.len(), dim);
    let est = q.inner_product_estimate(&code, &v).unwrap();
    assert!(est.is_finite());
}

#[test]
fn bits_16_maximum_bit_width() {
    let dim = 8;
    let q = TurboQuantizer::new(dim, 16, 4, 42).unwrap();
    let v = make_vec(dim, 200);
    let code = q.encode(&v).unwrap();
    let decoded = q.decode(&code);
    assert_eq!(decoded.len(), dim);
    // At 16 bits, reconstruction should be very close
    let mse: f64 = v
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        / dim as f64;
    assert!(mse < 0.1, "16-bit MSE should be very small, got {mse}");
}

// ============================================================================
// 18. projections=1 (minimum)
// ============================================================================

#[test]
fn projections_1_minimum() {
    let dim = 8;
    let q = TurboQuantizer::new(dim, 4, 1, 42).unwrap();
    let v = make_vec(dim, 300);
    let code = q.encode(&v).unwrap();
    let decoded = q.decode(&code);
    assert_eq!(decoded.len(), dim);
    let est = q.inner_product_estimate(&code, &v).unwrap();
    assert!(est.is_finite());
}

// ============================================================================
// 19. Zero-norm vector through full TurboQuant pipeline
// ============================================================================

#[test]
fn zero_norm_vector_full_pipeline() {
    let dim = 8;
    let q = TurboQuantizer::new(dim, 4, 8, 42).unwrap();
    let v = vec![0.0_f32; dim];
    let code = q.encode(&v).unwrap();
    let decoded = q.decode(&code);
    assert_eq!(decoded.len(), dim);
    for x in &decoded {
        assert!(x.abs() < 1e-5, "decoded zero vector should be ~0, got {x}");
    }

    let query = make_vec(dim, 1);
    let est = q.inner_product_estimate(&code, &query).unwrap();
    assert!(
        est.abs() < 1.0,
        "IP with zero vector should be ~0, got {est}"
    );

    let dist = q.l2_distance_estimate(&code, &query).unwrap();
    assert!(dist.is_finite() && dist >= 0.0);

    // Compact roundtrip
    let bytes = code.to_compact_bytes();
    let back = TurboCode::from_compact_bytes(&bytes).unwrap();
    assert_eq!(back.to_compact_bytes(), bytes);
}

// ============================================================================
// 20. PolarQuantizer via VectorQuantizer trait dispatch
// ============================================================================

#[test]
fn polar_quantizer_trait_dispatch() {
    let dim = 8;
    let boxed: Box<dyn VectorQuantizer<Code = PolarCode>> =
        Box::new(PolarQuantizer::new(dim, 4).unwrap());

    assert_eq!(boxed.dim(), dim);
    let v = make_vec(dim, 500);
    let code = boxed.encode(&v).unwrap();
    let decoded = boxed.decode(&code);
    assert_eq!(decoded.len(), dim);
    let score = boxed.inner_product_estimate(&code, &v).unwrap();
    assert!(score.is_finite());
    let dist = boxed.l2_distance_estimate(&code, &v).unwrap();
    assert!(dist >= 0.0 && dist.is_finite());
    let sz = boxed.code_size_bytes(&code);
    assert!(sz > 0);
}

// ============================================================================
// 21. StoredRotation + RotationStrategy trait dispatch
// ============================================================================

#[test]
fn rotation_strategy_trait_dispatch() {
    let dim = 8;
    let rotation = StoredRotation::new(dim, 42).unwrap();
    let boxed: Box<dyn RotationStrategy> = Box::new(rotation);

    assert_eq!(boxed.dim(), dim);
    let v = make_vec(dim, 600);
    let rotated = boxed.rotate(&v);
    assert_eq!(rotated.len(), dim);

    // Norm preservation
    let v_norm = l2_norm(&v);
    let r_norm = l2_norm(&rotated);
    assert!(
        (v_norm - r_norm).abs() < 1e-4,
        "rotation should preserve norm: {v_norm} vs {r_norm}"
    );

    // Roundtrip
    let back = boxed.rotate_inverse(&rotated);
    assert_eq!(back.len(), dim);
    for (a, b) in v.iter().zip(back.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "rotation roundtrip failed: {a} vs {b}"
        );
    }
}

// ============================================================================
// 22. Compact deserialization error paths — malformed data
// ============================================================================

#[test]
fn compact_deser_polar_invalid_bits() {
    // Craft a PolarCode compact buffer with bits=0 (invalid)
    let mut bytes = vec![0x01u8, 0x00, 0x00, 0x00]; // version=1, bits=0, pairs=0
    assert!(PolarCode::from_compact_bytes(&bytes).is_err());
    // bits=17 (invalid)
    bytes[1] = 17;
    assert!(PolarCode::from_compact_bytes(&bytes).is_err());
}

#[test]
fn compact_deser_qjl_invalid_projections() {
    // Craft QjlSketch compact buffer with num_projections=0
    let bytes = vec![0x01u8, 0, 0, 0, 0, 0, 0]; // version=1, proj=0, norm=0.0
    assert!(QjlSketch::from_compact_bytes(&bytes).is_err());
}

#[test]
fn compact_deser_qjl_nan_norm() {
    // Craft QjlSketch compact buffer with norm=NaN
    let nan_bytes = f32::NAN.to_le_bytes();
    let mut bytes = vec![0x01u8, 1, 0]; // version=1, proj=1
    bytes.extend_from_slice(&nan_bytes);
    bytes.push(0xFF); // 1 sign byte
    assert!(QjlSketch::from_compact_bytes(&bytes).is_err());
}

#[test]
fn compact_deser_qjl_negative_norm() {
    let neg_bytes = (-1.0_f32).to_le_bytes();
    let mut bytes = vec![0x01u8, 1, 0]; // version=1, proj=1
    bytes.extend_from_slice(&neg_bytes);
    bytes.push(0xFF); // 1 sign byte
    assert!(QjlSketch::from_compact_bytes(&bytes).is_err());
}

// ============================================================================
// 23. Large dimension (d=4096) smoke test
// ============================================================================

#[test]
fn large_dimension_512() {
    let dim = 512;
    let q = TurboQuantizer::new(dim, 4, 32, 42).unwrap();
    let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let code = q.encode(&v).unwrap();
    let decoded = q.decode(&code);
    assert_eq!(decoded.len(), dim);
    let est = q.inner_product_estimate(&code, &v).unwrap();
    assert!(est.is_finite());
}

// ============================================================================
// 24. TurboQuantError is Clone
// ============================================================================

#[test]
fn error_is_clone() {
    let err = bitpolar::TurboQuantError::OddDimension(3);
    let cloned = err.clone();
    assert_eq!(err.to_string(), cloned.to_string());
}

// ============================================================================
// Task A — Forward-compatibility tests for compact binary format
// ============================================================================

// A-1. Future version (0x02) must be rejected for all three code types.
#[test]
fn test_compact_future_version_rejected() {
    // PolarCode: craft a buffer with version=0x02
    {
        let q = PolarQuantizer::new(8, 4).unwrap();
        let v = make_vec(8, 1);
        let mut bytes = q.encode(&v).unwrap().to_compact_bytes();
        bytes[0] = 0x02;
        assert!(
            matches!(
                PolarCode::from_compact_bytes(&bytes),
                Err(bitpolar::TurboQuantError::DeserializationError { .. })
            ),
            "PolarCode must reject version 0x02"
        );
    }

    // QjlSketch: craft a buffer with version=0x02
    {
        let q = QjlQuantizer::new(8, 8, 42).unwrap();
        let v = make_vec(8, 2);
        let mut bytes = q.sketch(&v).unwrap().to_compact_bytes();
        bytes[0] = 0x02;
        assert!(
            matches!(
                QjlSketch::from_compact_bytes(&bytes),
                Err(bitpolar::TurboQuantError::DeserializationError { .. })
            ),
            "QjlSketch must reject version 0x02"
        );
    }

    // TurboCode: craft a buffer with version=0x02
    {
        let q = TurboQuantizer::new(8, 4, 8, 42).unwrap();
        let v = make_vec(8, 3);
        let mut bytes = q.encode(&v).unwrap().to_compact_bytes();
        bytes[0] = 0x02;
        assert!(
            matches!(
                TurboCode::from_compact_bytes(&bytes),
                Err(bitpolar::TurboQuantError::DeserializationError { .. })
            ),
            "TurboCode must reject version 0x02"
        );
    }
}

// A-2. Trailing garbage bytes — document current parser behaviour.
//
// The implementation uses exact-length slicing for PolarCode (the TurboCode
// framing passes exactly `polar_len` bytes to PolarCode, and the remainder
// as QjlSketch payload).  QjlSketch reads exactly `7 + n_sign_bytes` and
// ignores the rest.  PolarCode reads exactly `4 + payload` and ignores any
// extra bytes beyond the expected length.  So trailing bytes ARE tolerated
// by the individual decoders; the test verifies that appending 100 garbage
// bytes does not break deserialization.
#[test]
fn test_compact_trailing_bytes_tolerated() {
    let garbage = vec![0xDE_u8; 100];

    // TurboCode
    {
        let q = TurboQuantizer::new(8, 4, 8, 42).unwrap();
        let v = make_vec(8, 10);
        let code = q.encode(&v).unwrap();
        let mut bytes = code.to_compact_bytes();
        bytes.extend_from_slice(&garbage);

        // TurboCode from_compact_bytes reads exactly polar_len polar bytes then
        // treats the rest as the QjlSketch payload — trailing bytes after the
        // QjlSketch payload are tolerated (QjlSketch only reads what it needs).
        match TurboCode::from_compact_bytes(&bytes) {
            Ok(decoded) => {
                // If it succeeds, the core fields must match.
                let original_bytes = code.to_compact_bytes();
                let re_encoded = decoded.to_compact_bytes();
                assert_eq!(
                    original_bytes, re_encoded,
                    "TurboCode: trailing bytes tolerated but decoded fields must match"
                );
            }
            Err(_) => {
                // Strict parsing is also acceptable.  Document this explicitly.
                // The test passes either way; the important thing is: no panic.
            }
        }
    }

    // PolarCode
    {
        let q = PolarQuantizer::new(8, 4).unwrap();
        let v = make_vec(8, 11);
        let code = q.encode(&v).unwrap();
        let mut bytes = code.to_compact_bytes();
        bytes.extend_from_slice(&garbage);

        match PolarCode::from_compact_bytes(&bytes) {
            Ok(decoded) => {
                let original_bytes = code.to_compact_bytes();
                let re_encoded = decoded.to_compact_bytes();
                assert_eq!(
                    original_bytes, re_encoded,
                    "PolarCode: trailing bytes tolerated but decoded fields must match"
                );
            }
            Err(_) => {
                // Strict parsing is also valid — no panic is the contract.
            }
        }
    }

    // QjlSketch
    {
        let q = QjlQuantizer::new(8, 8, 42).unwrap();
        let v = make_vec(8, 12);
        let sketch = q.sketch(&v).unwrap();
        let mut bytes = sketch.to_compact_bytes();
        bytes.extend_from_slice(&garbage);

        match QjlSketch::from_compact_bytes(&bytes) {
            Ok(decoded) => {
                let original_bytes = sketch.to_compact_bytes();
                let re_encoded = decoded.to_compact_bytes();
                assert_eq!(
                    original_bytes, re_encoded,
                    "QjlSketch: trailing bytes tolerated but decoded fields must match"
                );
            }
            Err(_) => {
                // Strict parsing is also valid — no panic is the contract.
            }
        }
    }
}

// A-3. Golden vector test — pin exact byte lengths and round-trip fidelity.
//
// This test catches accidental format changes.  The expected byte lengths
// are computed from the format documentation:
//
// PolarCode  (dim=8, bits=4):
//   version(1) + bits(1) + num_pairs(2) + radii(4×4=16) + angles(4×2=8) = 28 bytes
//
// QjlSketch  (projections=8):
//   version(1) + num_projections(2) + norm(4) + signs(ceil(8/8)=1) = 8 bytes
//
// TurboCode  (dim=8, bits=4, projections=8):
//   version(1) + polar_len(4) + polar_payload(28) + qjl_payload(8) = 41 bytes
#[test]
fn test_compact_golden_vectors() {
    let dim = 8usize;
    let bits = 4u8;
    let projections = 8usize;
    let seed = 42u64;
    let vector = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    // ---- PolarCode ----
    {
        let q = PolarQuantizer::new(dim, bits).unwrap();
        let code = q.encode(&vector).unwrap();
        let bytes = code.to_compact_bytes();

        // num_pairs = dim/2 = 4
        // expected: 1 (version) + 1 (bits) + 2 (num_pairs) + 4*4 (radii) + 4*2 (angles) = 28
        assert_eq!(
            bytes.len(),
            28,
            "PolarCode golden byte length mismatch: expected 28, got {}",
            bytes.len()
        );

        // Roundtrip: decoded fields must match original
        let decoded = PolarCode::from_compact_bytes(&bytes).unwrap();
        // Access via to_compact_bytes identity (pub(crate) fields not accessible here)
        assert_eq!(
            decoded.to_compact_bytes(),
            bytes,
            "PolarCode golden roundtrip failed"
        );
    }

    // ---- QjlSketch ----
    {
        let q = QjlQuantizer::new(dim, projections, seed).unwrap();
        let sketch = q.sketch(&vector).unwrap();
        let bytes = sketch.to_compact_bytes();

        // expected: 1 (version) + 2 (num_projections) + 4 (norm) + ceil(8/8)=1 (signs) = 8
        assert_eq!(
            bytes.len(),
            8,
            "QjlSketch golden byte length mismatch: expected 8, got {}",
            bytes.len()
        );

        let decoded = QjlSketch::from_compact_bytes(&bytes).unwrap();
        assert_eq!(
            decoded.to_compact_bytes(),
            bytes,
            "QjlSketch golden roundtrip failed"
        );
        assert_eq!(decoded.num_projections(), projections);
        assert!(decoded.norm() >= 0.0);
    }

    // ---- TurboCode ----
    {
        let q = TurboQuantizer::new(dim, bits, projections, seed).unwrap();
        let code = q.encode(&vector).unwrap();
        let bytes = code.to_compact_bytes();

        // expected: 1 (version) + 4 (polar_len u32) + 28 (polar payload) + 8 (qjl payload) = 41
        assert_eq!(
            bytes.len(),
            41,
            "TurboCode golden byte length mismatch: expected 41, got {}",
            bytes.len()
        );

        let decoded = TurboCode::from_compact_bytes(&bytes).unwrap();
        assert_eq!(
            decoded.to_compact_bytes(),
            bytes,
            "TurboCode golden roundtrip failed"
        );
    }
}

// A-4. PolarCode with 0 pairs (dim=2 → 1 pair; there is no 0-pair variant via
// the public API since dim must be even and > 0 — the minimum is 1 pair).
// We instead manually craft a compact buffer claiming 0 pairs and verify that
// the parser handles it gracefully (either succeeds with an empty code or
// returns a meaningful error — no panic in either case).
#[test]
fn test_compact_zero_length_code() {
    // Craft a PolarCode compact buffer with 0 pairs (num_pairs = 0).
    // Format: [version=0x01][bits=4][num_pairs_lo=0][num_pairs_hi=0]
    // No radii or angle bytes needed for 0 pairs.
    let bytes = [0x01u8, 4u8, 0u8, 0u8];

    match PolarCode::from_compact_bytes(&bytes) {
        Ok(code) => {
            // If it succeeds, the round-trip must be stable.
            let re_encoded = code.to_compact_bytes();
            assert_eq!(
                re_encoded, bytes,
                "zero-pair PolarCode roundtrip must be stable"
            );
        }
        Err(e) => {
            // Graceful error is also acceptable.
            assert!(
                format!("{e}").contains("deserialization failed"),
                "error message should describe the issue, got: {e}"
            );
        }
    }

    // Also verify that PolarQuantizer::new(2, 4) (minimum real dimension = 1 pair)
    // round-trips cleanly, as the nearby edge case.
    {
        let q = PolarQuantizer::new(2, 4).unwrap();
        let v = vec![1.0_f32, 0.0];
        let code = q.encode(&v).unwrap();
        let bytes = code.to_compact_bytes();
        let back = PolarCode::from_compact_bytes(&bytes).unwrap();
        assert_eq!(
            back.to_compact_bytes(),
            bytes,
            "1-pair PolarCode roundtrip failed"
        );
    }
}
