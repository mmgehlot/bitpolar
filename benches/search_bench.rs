//! Criterion benchmarks for approximate nearest-neighbour search.
//!
//! Covers:
//! - fp32_dot_product (f32 baseline) vs turbo_ip_estimate at matching dims
//! - batch_ip_estimate for database sizes 100, 1 000, 10 000
//! - compact_serialize / compact_deserialize round-trip

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use bitpolar::{TurboCode, TurboQuantizer};

const SEARCH_DIMS: &[usize] = &[64, 128, 256, 512, 1024];
const BITS: u8 = 4;
const SEED: u64 = 42;

fn make_vector(dim: usize, idx: usize) -> Vec<f32> {
    (0..dim).map(|j| ((idx * dim + j) as f32 * 1e-5).sin()).collect()
}

// ============================================================================
// Baseline: plain f32 dot product (no quantization)
// ============================================================================

fn bench_fp32_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp32_dot_product");

    for &dim in SEARCH_DIMS {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();

        group.bench_with_input(BenchmarkId::new("baseline", dim), &(a, b), |bench, (a, b)| {
            bench.iter(|| {
                black_box(a)
                    .iter()
                    .zip(black_box(b).iter())
                    .map(|(x, y)| x * y)
                    .sum::<f32>()
            })
        });
    }

    group.finish();
}

// ============================================================================
// TurboQuant IP estimate — same dimensions as baseline
// ============================================================================

fn bench_turbo_ip_estimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_ip_estimate");

    for &dim in SEARCH_DIMS {
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, BITS, projections, SEED).unwrap();
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
        let code = q.encode(&v).unwrap();

        group.bench_with_input(
            BenchmarkId::new("turbo", dim),
            &(code, query),
            |b, (code, query)| {
                b.iter(|| q.inner_product_estimate(black_box(code), black_box(query)).unwrap())
            },
        );
    }

    group.finish();
}

// ============================================================================
// Linear scan over database at dim=128
// ============================================================================

fn bench_linear_scan(c: &mut Criterion) {
    let dim = 128usize;
    let projections = (dim / 4).max(1);
    let q = TurboQuantizer::new(dim, BITS, projections, SEED).unwrap();

    let mut group = c.benchmark_group("linear_scan");

    for &db_size in &[100usize, 1_000, 10_000] {
        let db: Vec<TurboCode> = (0..db_size)
            .map(|i| q.encode(&make_vector(dim, i)).unwrap())
            .collect();

        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();

        group.bench_with_input(BenchmarkId::new("turbo_scan", db_size), &db, |b, db| {
            b.iter(|| {
                db.iter()
                    .map(|code| {
                        q.inner_product_estimate(black_box(code), black_box(&query)).unwrap()
                    })
                    .fold(f32::MIN, f32::max)
            })
        });

        // Baseline: plain f32 dot over uncompressed database at same size.
        let fp32_db: Vec<Vec<f32>> = (0..db_size).map(|i| make_vector(dim, i)).collect();
        group.bench_with_input(
            BenchmarkId::new("fp32_scan", db_size),
            &fp32_db,
            |b, db| {
                b.iter(|| {
                    db.iter()
                        .map(|v| {
                            black_box(v)
                                .iter()
                                .zip(black_box(&query).iter())
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                        })
                        .fold(f32::MIN, f32::max)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Batch IP estimate (parallel feature gate)
// ============================================================================

#[cfg(feature = "parallel")]
fn bench_batch_ip_estimate(c: &mut Criterion) {
    use bitpolar::traits::BatchQuantizer;

    let dim = 128usize;
    let projections = (dim / 4).max(1);
    let q = TurboQuantizer::new(dim, BITS, projections, SEED).unwrap();

    let mut group = c.benchmark_group("batch_ip_estimate");

    for &n_codes in &[100usize, 1_000, 10_000] {
        let codes: Vec<TurboCode> = (0..n_codes)
            .map(|i| q.encode(&make_vector(dim, i)).unwrap())
            .collect();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();

        group.bench_with_input(
            BenchmarkId::new("parallel", n_codes),
            &codes,
            |b, codes| {
                b.iter(|| q.batch_inner_product(black_box(codes), black_box(&query)).unwrap())
            },
        );
    }

    group.finish();
}

// ============================================================================
// Compact serialize / deserialize at dim=128
// ============================================================================

fn bench_compact_serialize(c: &mut Criterion) {
    let dim = 128usize;
    let projections = (dim / 4).max(1);
    let q = TurboQuantizer::new(dim, BITS, projections, SEED).unwrap();
    let code = q.encode(&make_vector(dim, 0)).unwrap();

    let mut group = c.benchmark_group("compact_codec");

    group.bench_function("serialize", |b| {
        b.iter(|| black_box(&code).to_compact_bytes())
    });

    let bytes = code.to_compact_bytes();
    group.bench_function("deserialize", |b| {
        b.iter(|| TurboCode::from_compact_bytes(black_box(&bytes)).unwrap())
    });

    group.finish();
}

// ============================================================================
// criterion_group / criterion_main
// ============================================================================

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_fp32_dot_product,
    bench_turbo_ip_estimate,
    bench_linear_scan,
    bench_batch_ip_estimate,
    bench_compact_serialize,
);

#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    bench_fp32_dot_product,
    bench_turbo_ip_estimate,
    bench_linear_scan,
    bench_compact_serialize,
);

criterion_main!(benches);
