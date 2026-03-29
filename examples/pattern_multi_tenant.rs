//! Multi-tenant vector quantization pattern.
//!
//! Each tenant gets a unique seed, ensuring complete isolation:
//! - Different rotation matrices per tenant
//! - No cross-tenant information leakage
//! - Independent quality metrics per tenant
//!
//! Run: `cargo run --example multi_tenant`

use bitpolar::traits::VectorQuantizer;
use bitpolar::TurboQuantizer;
use std::collections::HashMap;
use std::sync::Arc;

/// Tenant-isolated quantizer pool.
/// Each tenant gets a unique TurboQuantizer based on their tenant ID.
struct TenantPool {
    dim: usize,
    bits: u8,
    projections: usize,
    quantizers: HashMap<u64, Arc<TurboQuantizer>>,
}

impl TenantPool {
    fn new(dim: usize, bits: u8) -> Self {
        Self {
            dim,
            bits,
            projections: (dim / 4).max(1),
            quantizers: HashMap::new(),
        }
    }

    /// Get or create a quantizer for a tenant.
    /// Seed = tenant_id * 1000 + 42 (deterministic, isolated).
    fn for_tenant(&mut self, tenant_id: u64) -> Arc<TurboQuantizer> {
        self.quantizers
            .entry(tenant_id)
            .or_insert_with(|| {
                let seed = tenant_id * 1000 + 42;
                Arc::new(
                    TurboQuantizer::new(self.dim, self.bits, self.projections, seed)
                        .expect("Failed to create quantizer"),
                )
            })
            .clone()
    }
}

fn main() {
    let mut pool = TenantPool::new(128, 4);

    // Simulate 3 tenants
    for tenant_id in [1, 2, 3] {
        let q = pool.for_tenant(tenant_id);
        let vector = vec![0.1_f32; 128];
        let code = q.encode(&vector).unwrap();
        let score = q.inner_product_estimate(&code, &vector).unwrap();
        println!(
            "Tenant {}: encoded {} dims → {} bytes, self-similarity = {:.4}",
            tenant_id,
            q.dim(),
            {
                use bitpolar::traits::SerializableCode;
                code.to_compact_bytes().len()
            },
            score
        );
    }

    // Verify isolation: same vector, different tenants → different codes
    let v = vec![0.5_f32; 128];
    let q1 = pool.for_tenant(1);
    let q2 = pool.for_tenant(2);
    use bitpolar::traits::SerializableCode;
    let code1 = q1.encode(&v).unwrap().to_compact_bytes();
    let code2 = q2.encode(&v).unwrap().to_compact_bytes();
    assert_ne!(
        code1, code2,
        "Different tenants must produce different codes"
    );
    println!("\nTenant isolation verified: same vector → different codes per tenant");
}
