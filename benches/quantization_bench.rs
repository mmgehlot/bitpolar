//! Criterion benchmarks for vector quantization throughput.
//!
//! Groups: `encode`, `decode`, `ip_estimate`, `l2_estimate`
//! Parameterized by dimension: 64, 128, 256, 512, 1024.
//! Fixed: bits=4, projections=dim/4, seed=42.

use bitpolar::TurboQuantizer;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const DIMS: &[usize] = &[64, 128, 256, 512, 1024];
const BITS: u8 = 4;
const SEED: u64 = 42;

/// Build a deterministic input vector for the given dimension.
fn make_vector(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * 0.01).sin()).collect()
}

// ============================================================================
// encode group
// ============================================================================

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    for &dim in DIMS {
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, BITS, projections, SEED).unwrap();
        let vector = make_vector(dim);

        group.bench_with_input(BenchmarkId::new("turbo", dim), &vector, |b, v| {
            b.iter(|| q.encode(black_box(v)).unwrap())
        });
    }

    group.finish();
}

// ============================================================================
// decode group
// ============================================================================

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    for &dim in DIMS {
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, BITS, projections, SEED).unwrap();
        let vector = make_vector(dim);
        let code = q.encode(&vector).unwrap();

        group.bench_with_input(BenchmarkId::new("turbo", dim), &code, |b, c| {
            b.iter(|| q.decode(black_box(c)))
        });
    }

    group.finish();
}

// ============================================================================
// ip_estimate group
// ============================================================================

fn bench_ip_estimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("ip_estimate");

    for &dim in DIMS {
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, BITS, projections, SEED).unwrap();
        let vector = make_vector(dim);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
        let code = q.encode(&vector).unwrap();

        group.bench_with_input(
            BenchmarkId::new("turbo", dim),
            &(code, query),
            |b, (c, q_vec)| {
                b.iter(|| {
                    q.inner_product_estimate(black_box(c), black_box(q_vec))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// l2_estimate group
// ============================================================================

fn bench_l2_estimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_estimate");

    for &dim in DIMS {
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, BITS, projections, SEED).unwrap();
        let vector = make_vector(dim);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.015).cos()).collect();
        let code = q.encode(&vector).unwrap();

        group.bench_with_input(
            BenchmarkId::new("turbo", dim),
            &(code, query),
            |b, (c, q_vec)| {
                b.iter(|| {
                    q.l2_distance_estimate(black_box(c), black_box(q_vec))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// compact serialization group
// ============================================================================

fn bench_compact_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("compact_serialize");

    for &dim in DIMS {
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, BITS, projections, SEED).unwrap();
        let code = q.encode(&make_vector(dim)).unwrap();

        group.bench_with_input(BenchmarkId::new("to_compact_bytes", dim), &code, |b, c| {
            b.iter(|| black_box(c).to_compact_bytes())
        });
    }

    group.finish();
}

fn bench_compact_deserialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("compact_deserialize");

    for &dim in DIMS {
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, BITS, projections, SEED).unwrap();
        let code = q.encode(&make_vector(dim)).unwrap();
        let bytes = code.to_compact_bytes();

        group.bench_with_input(
            BenchmarkId::new("from_compact_bytes", dim),
            &bytes,
            |b, bytes| {
                b.iter(|| bitpolar::TurboCode::from_compact_bytes(black_box(bytes)).unwrap())
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_ip_estimate,
    bench_l2_estimate,
    bench_compact_serialize,
    bench_compact_deserialize,
);
criterion_main!(benches);
