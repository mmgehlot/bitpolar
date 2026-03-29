//! BitPolar CLI — command-line vector quantization tool.
//!
//! Compress, decompress, search, and benchmark vector datasets.
//!
//! # Usage
//!
//! ```bash
//! bitpolar compress -i vectors.json -o compressed.bp --bits 4
//! bitpolar search -i compressed.bp -q query.json -k 10
//! bitpolar bench -n 10000 -d 384 --bits 3,4,6,8
//! bitpolar info compressed.bp
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;

use bitpolar::traits::VectorQuantizer;
use bitpolar::TurboQuantizer;

/// BitPolar: near-optimal vector quantization CLI
#[derive(Parser)]
#[command(name = "bitpolar", version, about)]
#[command(propagate_version = true)]
struct Cli {
    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress vectors to BitPolar format
    Compress(CompressArgs),
    /// Search a compressed index
    Search(SearchArgs),
    /// Run compression benchmarks
    Bench(BenchArgs),
    /// Show info about a compressed file
    Info(InfoArgs),
}

#[derive(clap::Args)]
struct CompressArgs {
    /// Input file (JSON array of float arrays)
    #[arg(short, long)]
    input: PathBuf,
    /// Output file (.bp format)
    #[arg(short, long)]
    output: PathBuf,
    /// Bits per dimension (3-8)
    #[arg(long, default_value = "4")]
    bits: u8,
    /// Number of QJL projections (default: dim/4)
    #[arg(long)]
    projections: Option<usize>,
    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,
}

#[derive(clap::Args)]
struct SearchArgs {
    /// Compressed index file (.bp)
    #[arg(short, long)]
    index: PathBuf,
    /// Query vectors file (JSON array of float arrays)
    #[arg(short, long)]
    query: PathBuf,
    /// Top-k results per query
    #[arg(short, long, default_value = "10")]
    k: usize,
}

#[derive(clap::Args)]
struct BenchArgs {
    /// Number of vectors to generate
    #[arg(short, long, default_value = "10000")]
    n: usize,
    /// Vector dimension
    #[arg(short, long, default_value = "384")]
    dim: usize,
    /// Comma-separated bit widths to benchmark
    #[arg(long, default_value = "3,4,6,8", value_delimiter = ',')]
    bits: Vec<u8>,
    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,
}

#[derive(clap::Args)]
struct InfoArgs {
    /// File to inspect
    file: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress(args) => cmd_compress(args),
        Commands::Search(args) => cmd_search(args),
        Commands::Bench(args) => cmd_bench(args),
        Commands::Info(args) => cmd_info(args),
    }
}

/// Compress vectors from JSON file to .bp format.
fn cmd_compress(args: CompressArgs) -> Result<()> {
    // Read input vectors from JSON
    let input_data = std::fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read {}", args.input.display()))?;
    let vectors: Vec<Vec<f32>> = serde_json::from_str(&input_data)
        .context("Failed to parse input as JSON array of float arrays")?;

    if vectors.is_empty() {
        anyhow::bail!("Input contains no vectors");
    }

    let dim = vectors[0].len();
    if dim == 0 {
        anyhow::bail!("Vectors have zero dimension");
    }

    // Validate all vectors have the same dimension
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != dim {
            anyhow::bail!(
                "Dimension mismatch: vector 0 has dim={}, but vector {} has dim={}",
                dim,
                i,
                v.len()
            );
        }
    }

    let n = vectors.len();
    let projections = args.projections.unwrap_or(dim / 4).max(1);

    eprintln!(
        "Compressing {} vectors (dim={}, bits={}, projections={}, seed={})",
        n, dim, args.bits, projections, args.seed
    );

    let q = TurboQuantizer::new(dim, args.bits, projections, args.seed)
        .context("Failed to create quantizer")?;

    // Encode all vectors and collect compact bytes
    let bar = indicatif::ProgressBar::new(n as u64);
    bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40} {pos}/{len} ({eta})")
            .unwrap(),
    );

    use bitpolar::traits::SerializableCode;
    let start = Instant::now();
    let mut codes: Vec<Vec<u8>> = Vec::with_capacity(n);
    for v in &vectors {
        let code = q.encode(v).context("Encode failed")?;
        codes.push(code.to_compact_bytes());
        bar.inc(1);
    }
    bar.finish();
    let elapsed = start.elapsed();

    // Write .bp file: header + codes
    let header = serde_json::json!({
        "magic": "BITPOLAR",
        "version": 1,
        "n_vectors": n,
        "dim": dim,
        "bits": args.bits,
        "projections": projections,
        "seed": args.seed,
        "code_lengths": codes.iter().map(|c| c.len()).collect::<Vec<_>>(),
    });
    let header_bytes = serde_json::to_vec(&header)?;

    let mut output = Vec::new();
    output.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    output.extend_from_slice(&header_bytes);
    for code in &codes {
        output.extend_from_slice(code);
    }
    std::fs::write(&args.output, &output)
        .with_context(|| format!("Failed to write {}", args.output.display()))?;

    let original_bytes = n * dim * 4;
    let compressed_bytes = output.len();
    eprintln!(
        "Done in {:.2}s: {} -> {} ({:.1}x compression)",
        elapsed.as_secs_f64(),
        format_bytes(original_bytes),
        format_bytes(compressed_bytes),
        original_bytes as f64 / compressed_bytes as f64,
    );

    Ok(())
}

/// Search a compressed index.
fn cmd_search(args: SearchArgs) -> Result<()> {
    // Read compressed index
    let data = std::fs::read(&args.index)
        .with_context(|| format!("Failed to read {}", args.index.display()))?;

    if data.len() < 4 {
        anyhow::bail!(
            "File too small to be a valid .bp file ({} bytes)",
            data.len()
        );
    }
    let header_len = u32::from_le_bytes(data[0..4].try_into()?) as usize;
    if 4 + header_len > data.len() {
        anyhow::bail!(
            "Corrupt .bp file: header length {} exceeds file size {}",
            header_len,
            data.len()
        );
    }
    let header: serde_json::Value = serde_json::from_slice(&data[4..4 + header_len])
        .context("Failed to parse .bp file header")?;

    let dim = header["dim"].as_u64().context("Missing 'dim' in header")? as usize;
    let bits = header["bits"]
        .as_u64()
        .context("Missing 'bits' in header")? as u8;
    let projections = header["projections"]
        .as_u64()
        .context("Missing 'projections'")? as usize;
    let seed = header["seed"].as_u64().context("Missing 'seed'")?;
    let code_lengths: Vec<usize> = header["code_lengths"]
        .as_array()
        .context("Missing 'code_lengths'")?
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let q = TurboQuantizer::new(dim, bits, projections, seed)?;

    // Parse codes from binary data
    use bitpolar::traits::SerializableCode;
    let mut offset = 4 + header_len;
    let mut codes = Vec::new();
    for &len in &code_lengths {
        let code = bitpolar::TurboCode::from_compact_bytes(&data[offset..offset + len])
            .context("Failed to deserialize code")?;
        codes.push(code);
        offset += len;
    }

    // Read query vectors
    let query_data = std::fs::read_to_string(&args.query)?;
    let queries: Vec<Vec<f32>> = serde_json::from_str(&query_data)?;

    // Search each query
    for (qi, query) in queries.iter().enumerate() {
        let mut scored: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .map(|(i, code)| {
                let score = q.inner_product_estimate(code, query).unwrap_or(f32::MIN);
                (i, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(args.k);

        println!("Query {}: {:?}", qi, scored);
    }

    Ok(())
}

/// Run compression benchmarks.
fn cmd_bench(args: BenchArgs) -> Result<()> {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(args.seed);

    // Generate random vectors
    let vectors: Vec<Vec<f32>> = (0..args.n)
        .map(|_| {
            (0..args.dim)
                .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect();

    println!("BitPolar Benchmark: {} vectors, dim={}", args.n, args.dim);
    println!("{:-<70}", "");
    println!(
        "{:<6} {:>12} {:>12} {:>12} {:>12}",
        "Bits", "Encode (ms)", "IP (ms)", "Compression", "Bytes/vec"
    );
    println!("{:-<70}", "");

    for &bits in &args.bits {
        let projections = (args.dim / 4).max(1);
        let q = TurboQuantizer::new(args.dim, bits, projections, args.seed)?;

        // Benchmark encode
        let start = Instant::now();
        let mut codes = Vec::with_capacity(args.n);
        for v in &vectors {
            codes.push(q.encode(v)?);
        }
        let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Benchmark inner product
        let query = &vectors[0];
        let start = Instant::now();
        for code in &codes {
            let _ = q.inner_product_estimate(code, query);
        }
        let ip_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Compute stats
        use bitpolar::traits::SerializableCode;
        let code_bytes: usize = codes.iter().map(|c| c.to_compact_bytes().len()).sum();
        let original_bytes = args.n * args.dim * 4;
        let compression = original_bytes as f64 / code_bytes as f64;
        let bytes_per_vec = code_bytes / args.n;

        println!(
            "{:<6} {:>12.1} {:>12.1} {:>11.1}x {:>12}",
            bits, encode_ms, ip_ms, compression, bytes_per_vec
        );
    }

    Ok(())
}

/// Show info about a .bp file.
fn cmd_info(args: InfoArgs) -> Result<()> {
    let data = std::fs::read(&args.file)?;
    if data.len() < 4 {
        anyhow::bail!("File too small ({} bytes)", data.len());
    }
    let header_len = u32::from_le_bytes(data[0..4].try_into()?) as usize;
    if 4 + header_len > data.len() {
        anyhow::bail!(
            "Corrupt file: header length {} exceeds file size",
            header_len
        );
    }
    let header: serde_json::Value =
        serde_json::from_slice(&data[4..4 + header_len]).context("Failed to parse header")?;

    println!("{}", serde_json::to_string_pretty(&header)?);
    println!("File size: {}", format_bytes(data.len()));

    Ok(())
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
