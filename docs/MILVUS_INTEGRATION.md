# Milvus Integration Guide

BitPolar can be integrated into Milvus as a quantization backend via the C FFI.

## Architecture

Milvus uses a C++ core (Knowhere) for vector indexing. BitPolar integrates
at the quantization layer:

```
Milvus Collection → Knowhere Index → BitPolar Quantizer → Compressed Codes
                                                        ↓
                                   Query → BitPolar IP Estimation → Ranked Results
```

## Integration via Knowhere Plugin

Knowhere (Milvus's vector engine) supports custom quantization through its
`Quantizer` interface:

```cpp
#include "bitpolar.h"
#include "knowhere/quantizer.h"

class BitPolarQuantizer : public knowhere::Quantizer {
public:
    BitPolarQuantizer(int dim, int bits = 4)
        : dim_(dim), bits_(bits), code_size_(0) {
        uint32_t projections = dim / 4;
        handle_ = turboquant_new(dim, bits, projections, 42);
        // Determine code size by encoding a dummy vector
        std::vector<float> dummy(dim, 0.0f);
        auto code = turboquant_encode(handle_, dummy.data(), dim);
        code_size_ = turboquant_code_to_bytes(code, nullptr, 0);
        turboquant_code_free(code);
    }

    ~BitPolarQuantizer() {
        if (handle_) turboquant_free(handle_);
    }

    // Quantize a batch of vectors
    void Encode(const float* data, int n, uint8_t* codes) override {
        for (int i = 0; i < n; i++) {
            auto code = turboquant_encode(handle_, data + i * dim_, dim_);
            uint32_t len = turboquant_code_to_bytes(code, codes, 4096);
            codes += len;
            turboquant_code_free(code);
        }
    }

    // Compute distances between codes and a query
    void ComputeDistances(
        const uint8_t* codes, int n,
        const float* query,
        float* distances
    ) override {
        for (int i = 0; i < n; i++) {
            auto code = turboquant_code_from_bytes(codes, code_size_);
            distances[i] = -turboquant_inner_product(handle_, code, query, dim_);
            turboquant_code_free(code);
            codes += code_size_;
        }
    }

private:
    TurboQuantHandle* handle_;
    int dim_;
    int bits_;
    int code_size_;
};
```

## Milvus Configuration

```yaml
# milvus.yaml
indexCoord:
  quantization:
    type: bitpolar
    bits: 4
    projections: auto  # dim/4
```

## Status

This integration requires modifications to Milvus's Knowhere engine.
We plan to submit a proposal to the Milvus community once benchmarks
demonstrate clear advantages over their built-in scalar quantization.

Preliminary benchmarks on SIFT-1M show BitPolar achieves comparable
recall to Milvus's Product Quantization at ~600x faster indexing speed.
