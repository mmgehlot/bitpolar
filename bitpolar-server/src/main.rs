//! BitPolar gRPC Server — vector compression microservice.
//!
//! Provides unary and streaming RPCs for encoding, decoding, and searching
//! compressed vectors. Deploy as a standalone service for language-agnostic
//! access to BitPolar quantization.
//!
//! # Usage
//!
//! ```bash
//! bitpolar-server --port 50051 --dim 384 --bits 4
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use bitpolar::traits::{SerializableCode, VectorQuantizer};
use bitpolar::TurboQuantizer;
use tonic::{transport::Server, Request, Response, Status};

// Include the generated protobuf code
pub mod proto {
    tonic::include_proto!("bitpolar.v1");
}

use proto::vector_compression_server::{VectorCompression, VectorCompressionServer};

/// Maximum k for search results to prevent DoS.
const MAX_SEARCH_K: usize = 10_000;

/// Server state: quantizer + in-memory index.
pub struct BitPolarService {
    /// Default quantizer (created from server config)
    quantizer: Arc<TurboQuantizer>,
    /// In-memory compressed index: id -> compact bytes (RwLock for concurrent reads)
    index: Arc<RwLock<HashMap<u64, Vec<u8>>>>,
}

impl BitPolarService {
    /// Create a new service with the given configuration.
    pub fn new(dim: usize, bits: u8, projections: usize, seed: u64) -> Result<Self, String> {
        let quantizer =
            TurboQuantizer::new(dim, bits, projections, seed).map_err(|e| e.to_string())?;
        Ok(Self {
            quantizer: Arc::new(quantizer),
            index: Arc::new(RwLock::new(HashMap::new())),
        })
    }
}

#[tonic::async_trait]
impl VectorCompression for BitPolarService {
    /// Encode a batch of vectors.
    async fn encode(
        &self,
        request: Request<proto::EncodeRequest>,
    ) -> Result<Response<proto::EncodeResponse>, Status> {
        let req = request.into_inner();
        let dim = req.dim as usize;
        let count = req.count as usize;

        // Guard against integer overflow in dim * count
        let expected_len = dim.checked_mul(count).ok_or_else(|| {
            Status::invalid_argument(format!("Overflow: dim={} x count={}", dim, count))
        })?;

        if req.vectors.len() != expected_len {
            return Err(Status::invalid_argument(format!(
                "Expected {} floats ({}x{}), got {}",
                expected_len,
                count,
                dim,
                req.vectors.len()
            )));
        }

        let bits = req.bits as u8;
        if !(3..=8).contains(&bits) {
            return Err(Status::invalid_argument(format!(
                "bits must be 3-8, got {}",
                bits
            )));
        }
        let projections = (dim / 4).max(1);
        let q = TurboQuantizer::new(dim, bits, projections, req.seed)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let mut codes = Vec::with_capacity(count);
        let mut compressed_size: u64 = 0;
        for i in 0..count {
            let vec = &req.vectors[i * dim..(i + 1) * dim];
            let code = q.encode(vec).map_err(|e| Status::internal(e.to_string()))?;
            let bytes = code.to_compact_bytes();
            compressed_size += bytes.len() as u64;
            codes.push(bytes);
        }

        Ok(Response::new(proto::EncodeResponse {
            codes,
            original_size: (count * dim * 4) as u64,
            compressed_size,
        }))
    }

    /// Decode compressed codes back to vectors.
    async fn decode(
        &self,
        request: Request<proto::DecodeRequest>,
    ) -> Result<Response<proto::DecodeResponse>, Status> {
        let req = request.into_inner();
        let dim = req.dim as usize;
        let bits = req.bits as u8;
        let projections = (dim / 4).max(1);

        let q = TurboQuantizer::new(dim, bits, projections, req.seed)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let mut vectors = Vec::new();
        for code_bytes in &req.codes {
            let code = bitpolar::TurboCode::from_compact_bytes(code_bytes)
                .map_err(|e| Status::internal(e.to_string()))?;
            let decoded = q.decode(&code);
            vectors.extend_from_slice(&decoded);
        }

        Ok(Response::new(proto::DecodeResponse {
            vectors,
            dim: dim as u32,
            count: req.codes.len() as u32,
        }))
    }

    /// Search the in-memory index.
    async fn search(
        &self,
        request: Request<proto::SearchRequest>,
    ) -> Result<Response<proto::SearchResponse>, Status> {
        let req = request.into_inner();
        let k = (req.k as usize).min(MAX_SEARCH_K);

        // Validate query dimension
        if req.query.len() != self.quantizer.dim() {
            return Err(Status::invalid_argument(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.quantizer.dim(),
                req.query.len()
            )));
        }

        // RwLock read — allows concurrent searches
        let index = self
            .index
            .read()
            .map_err(|_| Status::internal("Lock poisoned"))?;

        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(index.len());
        for (&id, code_bytes) in index.iter() {
            let code = bitpolar::TurboCode::from_compact_bytes(code_bytes)
                .map_err(|e| Status::internal(e.to_string()))?;
            let score = self
                .quantizer
                .inner_product_estimate(&code, &req.query)
                .map_err(|e| Status::internal(e.to_string()))?;
            scored.push((id, score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        let results = scored
            .into_iter()
            .map(|(id, score)| proto::SearchResult { id, score })
            .collect();

        Ok(Response::new(proto::SearchResponse { results }))
    }

    /// Add vectors to the index.
    async fn add_vectors(
        &self,
        request: Request<proto::AddVectorsRequest>,
    ) -> Result<Response<proto::AddVectorsResponse>, Status> {
        let req = request.into_inner();
        let dim = req.dim as usize;
        let count = req.ids.len();

        if dim != self.quantizer.dim() {
            return Err(Status::invalid_argument(format!(
                "Dimension {} != server dimension {}",
                dim,
                self.quantizer.dim()
            )));
        }

        let expected_len = dim
            .checked_mul(count)
            .ok_or_else(|| Status::invalid_argument("Vector count overflow"))?;

        if req.vectors.len() != expected_len {
            return Err(Status::invalid_argument("Vector count mismatch"));
        }

        // RwLock write — exclusive access for mutations
        let mut index = self
            .index
            .write()
            .map_err(|_| Status::internal("Lock poisoned"))?;

        for (i, &id) in req.ids.iter().enumerate() {
            let vec = &req.vectors[i * dim..(i + 1) * dim];
            let code = self
                .quantizer
                .encode(vec)
                .map_err(|e| Status::internal(e.to_string()))?;
            index.insert(id, code.to_compact_bytes());
        }

        Ok(Response::new(proto::AddVectorsResponse {
            count: count as u64,
            success: true,
        }))
    }

    /// Health check.
    async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthResponse>, Status> {
        let index = self
            .index
            .read()
            .map_err(|_| Status::internal("Lock poisoned"))?;
        Ok(Response::new(proto::HealthResponse {
            status: "healthy".to_string(),
            index_size: index.len() as u64,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let dim: usize = std::env::var("BITPOLAR_DIM")
        .unwrap_or_else(|_| "384".to_string())
        .parse()?;
    let bits: u8 = std::env::var("BITPOLAR_BITS")
        .unwrap_or_else(|_| "4".to_string())
        .parse()?;
    let port: u16 = std::env::var("BITPOLAR_PORT")
        .unwrap_or_else(|_| "50051".to_string())
        .parse()?;
    let seed: u64 = std::env::var("BITPOLAR_SEED")
        .unwrap_or_else(|_| "42".to_string())
        .parse()?;
    let projections = (dim / 4).max(1);

    let service = BitPolarService::new(dim, bits, projections, seed)
        .map_err(|e| format!("Failed to create service: {}", e))?;

    let host = std::env::var("BITPOLAR_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let addr = format!("{}:{}", host, port).parse()?;
    tracing::info!("BitPolar gRPC server starting on {}", addr);
    tracing::info!(
        "Config: dim={}, bits={}, projections={}, seed={}",
        dim,
        bits,
        projections,
        seed
    );

    Server::builder()
        .add_service(VectorCompressionServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
