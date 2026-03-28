#![no_main]

use bitpolar::TurboQuantizer;
use bitpolar::traits::VectorQuantizer;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes for dim_hint and at least 4 bytes for one f32.
    if data.len() < 6 {
        return;
    }

    // First 2 bytes: derive an even dimension in 2..=128.
    let raw_dim = u16::from_le_bytes([data[0], data[1]]) as usize;
    // Clamp to 2..=128, ensure even.
    let dim = ((raw_dim % 64) + 1) * 2; // yields 2, 4, ..., 128

    // Remaining bytes: interpret as f32 values.
    let float_bytes = &data[2..];
    let num_floats = float_bytes.len() / 4;
    if num_floats == 0 {
        return;
    }

    let mut vector: Vec<f32> = Vec::with_capacity(num_floats);
    for i in 0..num_floats {
        let off = i * 4;
        let v = f32::from_le_bytes([
            float_bytes[off],
            float_bytes[off + 1],
            float_bytes[off + 2],
            float_bytes[off + 3],
        ]);
        // Replace non-finite values with 0 so we exercise the encode path,
        // not just the NaN guard. The guard test already tests that path.
        vector.push(if v.is_finite() { v } else { 0.0 });
    }

    // Pad or truncate to exactly `dim` elements.
    vector.resize(dim, 0.0_f32);

    // Use a modest projection count to keep fuzz runs fast.
    let projections = dim.max(4);
    let seed = u64::from_le_bytes({
        let mut b = [0u8; 8];
        let copy_len = data.len().min(8);
        b[..copy_len].copy_from_slice(&data[..copy_len]);
        b
    });

    let Ok(q) = TurboQuantizer::new(dim, 4, projections, seed) else {
        return;
    };

    // Must not panic — may return Err for non-finite input.
    if let Ok(code) = q.encode(&vector) {
        let _decoded = q.decode(&code);
        let _ip = q.inner_product_estimate(&code, &vector);
        let _l2 = q.l2_distance_estimate(&code, &vector);
        let _bytes = code.to_compact_bytes();
    }
});
