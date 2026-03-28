# llama.cpp Integration Guide

BitPolar can compress KV caches in llama.cpp via the C FFI layer.

## Prerequisites

1. Build the BitPolar C static library:
   ```bash
   cargo build --release --features ffi
   # Output: target/release/libbitpolar.a (Linux/macOS)
   # Output: target/release/bitpolar.lib (Windows)
   ```

2. Generate the C header:
   ```bash
   cbindgen --config cbindgen.toml --crate bitpolar --output bitpolar.h
   ```

## Integration Pattern

In llama.cpp's KV cache implementation (`llama-kv-cache.cpp`):

```cpp
#include "bitpolar.h"

// At model load time — create quantizer for KV cache heads
struct llama_kv_cache {
    // Existing fields...
    TurboQuantHandle* bp_quantizer;  // BitPolar quantizer for this head
};

// Initialize BitPolar quantizer
void llama_kv_cache_init_bitpolar(llama_kv_cache& cache, int head_dim) {
    uint32_t projections = head_dim / 4;
    cache.bp_quantizer = turboquant_new(head_dim, 4, projections, 42);
}

// Compress a KV entry
void llama_kv_cache_compress(llama_kv_cache& cache, const float* key, int dim) {
    TurboCodeHandle* code = turboquant_encode(cache.bp_quantizer, key, dim);
    // Store compressed code instead of float key
    uint8_t buf[4096];
    uint32_t len = turboquant_code_to_bytes(code, buf, sizeof(buf));
    // ... store buf[0..len] in cache
    turboquant_code_free(code);
}

// Compute attention score on compressed key
float llama_kv_cache_attention_score(
    llama_kv_cache& cache,
    TurboCodeHandle* key_code,
    const float* query,
    int dim
) {
    return turboquant_inner_product(cache.bp_quantizer, key_code, query, dim);
}
```

## Build Configuration

Add to llama.cpp's `CMakeLists.txt`:

```cmake
# BitPolar KV cache quantization (optional)
option(LLAMA_BITPOLAR "Enable BitPolar KV cache compression" OFF)
if (LLAMA_BITPOLAR)
    find_library(BITPOLAR_LIB bitpolar PATHS /path/to/bitpolar/target/release)
    target_include_directories(llama PRIVATE /path/to/bitpolar)
    target_link_libraries(llama ${BITPOLAR_LIB})
    target_compile_definitions(llama PRIVATE GGML_USE_BITPOLAR)
endif()
```

## Expected Results

| Metric | FP16 KV Cache | BitPolar 4-bit |
|---|---|---|
| Memory per token (32 heads, d=128) | 16 KB | ~6 KB |
| Attention quality | Exact | ≤0.5% perplexity increase |
| Compression ratio | 1x | ~2.7x |
| Context length (24GB GPU) | ~32K tokens | ~86K tokens |

## Status

This integration requires modifications to llama.cpp's core KV cache
management. We plan to submit an upstream PR once the C FFI stabilizes.
Community contributions welcome.
