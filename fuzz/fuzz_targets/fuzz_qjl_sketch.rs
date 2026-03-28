#![no_main]

use bitpolar::qjl::QjlQuantizer;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Layout:
    //   byte 0     : raw dimension (mapped to 2..=128, even)
    //   byte 1     : raw projections (mapped to 1..=64)
    //   bytes 2..  : f32 values for the vector
    if data.len() < 10 {
        return;
    }

    let raw_dim = data[0] as usize;
    let dim = ((raw_dim % 64) + 1) * 2; // 2, 4, ..., 128

    let raw_proj = data[1] as usize;
    let projections = (raw_proj % 64) + 1; // 1..=64

    let float_bytes = &data[2..];
    let num_floats = float_bytes.len() / 4;
    if num_floats == 0 {
        return;
    }

    let mut vector: Vec<f32> = Vec::with_capacity(dim);
    for i in 0..num_floats.min(dim) {
        let off = i * 4;
        let v = f32::from_le_bytes([
            float_bytes[off],
            float_bytes[off + 1],
            float_bytes[off + 2],
            float_bytes[off + 3],
        ]);
        vector.push(if v.is_finite() { v } else { 0.0 });
    }
    vector.resize(dim, 0.0_f32);

    let seed = u64::from_le_bytes({
        let mut b = [0u8; 8];
        let n = data.len().min(8);
        b[..n].copy_from_slice(&data[..n]);
        b
    });

    let Ok(q) = QjlQuantizer::new(dim, projections, seed) else {
        return;
    };

    // sketch() must not panic.
    if let Ok(sketch) = q.sketch(&vector) {
        // inner_product_estimate must not panic either.
        let _ = q.inner_product_estimate(&sketch, &vector);
        // Compact round-trip must not panic.
        let bytes = sketch.to_compact_bytes();
        let _ = bitpolar::QjlSketch::from_compact_bytes(&bytes);
    }
});
