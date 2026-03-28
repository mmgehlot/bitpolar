// Package bitpolar provides Go bindings for BitPolar vector quantization
// via CGO wrapping the C FFI layer.
//
// Usage:
//
//	q, err := bitpolar.NewQuantizer(128, 4, 32, 42)
//	if err != nil { log.Fatal(err) }
//	defer q.Close()
//
//	code, err := q.Encode(vector)
//	score, err := q.InnerProduct(code, query)
package bitpolar

/*
#cgo LDFLAGS: -L${SRCDIR}/../target/release -lbitpolar -ldl -lm -lpthread
#include <stdint.h>
#include <stdlib.h>

// Forward declarations matching bitpolar's C FFI (src/ffi.rs)
typedef struct TurboQuantizer TurboQuantizer;
typedef struct TurboCode TurboCode;

extern TurboQuantizer* bitpolar_new(uint32_t dim, uint8_t bits, uint32_t projections, uint64_t seed, int32_t* err_out);
extern void bitpolar_free(TurboQuantizer* handle);
extern TurboCode* bitpolar_encode(const TurboQuantizer* q, const float* vector, uint32_t dim, int32_t* err_out);
extern void bitpolar_code_free(TurboCode* code);
extern int32_t bitpolar_inner_product(const TurboQuantizer* q, const TurboCode* code, const float* query, uint32_t dim, float* result_out);
extern int32_t bitpolar_decode(const TurboQuantizer* q, const TurboCode* code, float* out, uint32_t dim);
extern int32_t bitpolar_code_to_bytes(const TurboCode* code, uint8_t* buf, uint32_t buf_len);
extern TurboCode* bitpolar_code_from_bytes(const uint8_t* buf, uint32_t len, int32_t* err_out);
extern int32_t bitpolar_batch_inner_product(const TurboQuantizer* q, const TurboCode** codes, uint32_t n_codes, const float* query, uint32_t dim, float* scores_out);
extern uint32_t bitpolar_dim(const TurboQuantizer* q);
extern uint32_t bitpolar_code_size(const TurboCode* code);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Quantizer wraps a BitPolar TurboQuantizer.
type Quantizer struct {
	handle *C.TurboQuantizer
	dim    int
}

// Code wraps a compressed BitPolar TurboCode.
type Code struct {
	handle *C.TurboCode
}

// NewQuantizer creates a new TurboQuantizer.
// Parameters: dim (vector dimension), bits (3-8), projections (typically dim/4), seed.
func NewQuantizer(dim, bits, projections int, seed uint64) (*Quantizer, error) {
	if bits < 3 || bits > 8 {
		return nil, fmt.Errorf("bits must be 3-8, got %d", bits)
	}
	var errCode C.int32_t
	h := C.bitpolar_new(C.uint32_t(dim), C.uint8_t(bits), C.uint32_t(projections), C.uint64_t(seed), &errCode)
	if h == nil {
		return nil, fmt.Errorf("failed to create quantizer (dim=%d, bits=%d, err=%d)", dim, bits, int(errCode))
	}
	return &Quantizer{handle: h, dim: dim}, nil
}

// Close frees the quantizer's memory. Must be called when done.
func (q *Quantizer) Close() {
	if q.handle != nil {
		C.bitpolar_free(q.handle)
		q.handle = nil
	}
}

// Encode compresses a float32 vector into a Code.
func (q *Quantizer) Encode(vector []float32) (*Code, error) {
	if len(vector) != q.dim {
		return nil, fmt.Errorf("dimension mismatch: expected %d, got %d", q.dim, len(vector))
	}
	var errCode C.int32_t
	h := C.bitpolar_encode(q.handle, (*C.float)(unsafe.Pointer(&vector[0])), C.uint32_t(q.dim), &errCode)
	if h == nil {
		return nil, fmt.Errorf("encode failed (err=%d)", int(errCode))
	}
	return &Code{handle: h}, nil
}

// Decode decompresses a Code back to an approximate float32 vector.
func (q *Quantizer) Decode(code *Code) ([]float32, error) {
	out := make([]float32, q.dim)
	rc := C.bitpolar_decode(q.handle, code.handle, (*C.float)(unsafe.Pointer(&out[0])), C.uint32_t(q.dim))
	if rc != 0 {
		return nil, fmt.Errorf("decode failed (err=%d)", int(rc))
	}
	return out, nil
}

// InnerProduct estimates the inner product between a compressed code and a query.
func (q *Quantizer) InnerProduct(code *Code, query []float32) (float32, error) {
	if len(query) != q.dim {
		return 0, fmt.Errorf("query dimension mismatch: expected %d, got %d", q.dim, len(query))
	}
	var result C.float
	rc := C.bitpolar_inner_product(q.handle, code.handle, (*C.float)(unsafe.Pointer(&query[0])), C.uint32_t(q.dim), &result)
	if rc != 0 {
		return 0, fmt.Errorf("inner_product failed (err=%d)", int(rc))
	}
	return float32(result), nil
}

// ToBytes serializes a Code to compact bytes.
func (code *Code) ToBytes() ([]byte, error) {
	// Query required size
	size := C.bitpolar_code_to_bytes(code.handle, nil, 0)
	if size < 0 {
		return nil, fmt.Errorf("code_to_bytes failed (err=%d)", int(size))
	}
	buf := make([]byte, int(size))
	written := C.bitpolar_code_to_bytes(code.handle, (*C.uint8_t)(unsafe.Pointer(&buf[0])), C.uint32_t(size))
	if written < 0 {
		return nil, fmt.Errorf("code_to_bytes write failed (err=%d)", int(written))
	}
	return buf[:int(written)], nil
}

// CodeFromBytes deserializes a Code from compact bytes.
func CodeFromBytes(data []byte) (*Code, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("cannot deserialize empty byte slice")
	}
	var errCode C.int32_t
	h := C.bitpolar_code_from_bytes((*C.uint8_t)(unsafe.Pointer(&data[0])), C.uint32_t(len(data)), &errCode)
	if h == nil {
		return nil, fmt.Errorf("deserialization failed (err=%d)", int(errCode))
	}
	return &Code{handle: h}, nil
}

// Close frees the code's memory.
func (code *Code) Close() {
	if code.handle != nil {
		C.bitpolar_code_free(code.handle)
		code.handle = nil
	}
}

// BatchInnerProduct estimates inner products between multiple codes and a single query.
func (q *Quantizer) BatchInnerProduct(codes []*Code, query []float32) ([]float32, error) {
	if len(query) != q.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", q.dim, len(query))
	}
	if len(codes) == 0 {
		return []float32{}, nil
	}
	cPtrs := make([]*C.TurboCode, len(codes))
	for i, c := range codes {
		if c == nil || c.handle == nil {
			return nil, fmt.Errorf("code at index %d has nil handle", i)
		}
		cPtrs[i] = c.handle
	}
	scores := make([]float32, len(codes))
	rc := C.bitpolar_batch_inner_product(
		q.handle,
		(**C.TurboCode)(unsafe.Pointer(&cPtrs[0])),
		C.uint32_t(len(codes)),
		(*C.float)(unsafe.Pointer(&query[0])),
		C.uint32_t(q.dim),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("batch_inner_product failed (err=%d)", int(rc))
	}
	return scores, nil
}

// Dim returns the vector dimension.
func (q *Quantizer) Dim() int {
	return q.dim
}
