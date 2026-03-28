#![no_main]

use bitpolar::{PolarCode, QjlSketch, TurboCode};
use bitpolar::traits::SerializableCode;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // All three deserializers must not panic on arbitrary input.
    // Returning Err is perfectly fine.
    let _ = TurboCode::from_compact_bytes(data);
    let _ = PolarCode::from_compact_bytes(data);
    let _ = QjlSketch::from_compact_bytes(data);
});
