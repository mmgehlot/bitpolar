//! Dataset benchmarks: recall and throughput on synthetic datasets
//! mimicking SIFT-128, GloVe-200, and OpenAI-1536 dimensionalities.
//!
//! Run: `cargo bench --bench dataset_bench`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use bitpolar::TurboQuantizer;
use bitpolar::traits::VectorQuantizer;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

/// Generate n random vectors of given dimension using a seeded RNG.
fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

/// Compute recall@k: fraction of true top-k that appear in approximate top-k.
fn recall_at_k(exact_scores: &[f32], approx_scores: &[f32], k: usize) -> f32 {
    let n = exact_scores.len();
    let k = k.min(n);

    // Get top-k indices by exact scores
    let mut exact_idx: Vec<usize> = (0..n).collect();
    exact_idx.sort_by(|&a, &b| exact_scores[b].partial_cmp(&exact_scores[a]).unwrap());
    let exact_topk: std::collections::HashSet<usize> = exact_idx[..k].iter().cloned().collect();

    // Get top-k indices by approximate scores
    let mut approx_idx: Vec<usize> = (0..n).collect();
    approx_idx.sort_by(|&a, &b| approx_scores[b].partial_cmp(&approx_scores[a]).unwrap());
    let approx_topk: std::collections::HashSet<usize> = approx_idx[..k].iter().cloned().collect();

    exact_topk.intersection(&approx_topk).count() as f32 / k as f32
}

fn bench_dataset(c: &mut Criterion) {
    // Dataset configurations mimicking real-world embeddings
    let configs = vec![
        ("SIFT-128", 128, 10000, 4),
        ("GloVe-200", 200, 10000, 4),
        ("MiniLM-384", 384, 5000, 4),
        ("OpenAI-1536", 1536, 1000, 4),
    ];

    let mut group = c.benchmark_group("dataset_recall");
    group.sample_size(10); // Fewer samples for large datasets

    for (name, dim, n, bits) in &configs {
        let vectors = generate_vectors(*n, *dim, 42);
        let projections = (*dim / 4).max(1);
        let q = TurboQuantizer::new(*dim, *bits, projections, 42).unwrap();

        // Encode all vectors
        let codes: Vec<_> = vectors.iter().map(|v| q.encode(v).unwrap()).collect();

        // Benchmark: search throughput (IP estimation for all codes against a query)
        let query = &vectors[0];
        group.bench_with_input(
            BenchmarkId::new("search_throughput", name),
            &(*n, *dim),
            |b, _| {
                b.iter(|| {
                    let _scores: Vec<f32> = codes
                        .iter()
                        .map(|code| q.inner_product_estimate(code, query).unwrap())
                        .collect();
                });
            },
        );

        // Measure recall@10
        let exact_scores: Vec<f32> = vectors
            .iter()
            .map(|v| v.iter().zip(query.iter()).map(|(a, b)| a * b).sum())
            .collect();
        let approx_scores: Vec<f32> = codes
            .iter()
            .map(|code| q.inner_product_estimate(code, query).unwrap())
            .collect();
        let recall = recall_at_k(&exact_scores, &approx_scores, 10);
        eprintln!("{}: recall@10 = {:.3} ({}-bit, n={})", name, recall, bits, n);
    }

    group.finish();
}

criterion_group!(benches, bench_dataset);
criterion_main!(benches);
