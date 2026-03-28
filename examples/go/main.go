// BitPolar Go Quick Start — CGO bindings via C FFI.
//
// Prerequisites:
//   1. Build the C library: cargo build --release --features ffi
//   2. Set CGO_LDFLAGS and CGO_CFLAGS
//
// Usage:
//   cd examples/go
//   CGO_LDFLAGS="-L../../target/release -lbitpolar" go run main.go

package main

import (
	"fmt"
	"math"
	"math/rand"
)

func main() {
	fmt.Println("=== BitPolar Go (CGO) Quick Start ===")
	fmt.Println()

	// NOTE: Uncomment below when the C library is built.
	// Requires: cargo build --release --features ffi

	dim := 128
	// bits := 4
	// projections := 32
	// seed := uint64(42)

	// Create quantizer
	// q, err := bitpolar.NewQuantizer(dim, bits, projections, seed)
	// if err != nil {
	//     log.Fatal(err)
	// }
	// defer q.Close()
	// fmt.Printf("Quantizer: dim=%d\n", q.Dim())

	// Create a random vector
	vector := make([]float32, dim)
	for i := range vector {
		vector[i] = float32(rand.NormFloat64())
	}

	// Encode
	// code, err := q.Encode(vector)
	// if err != nil {
	//     log.Fatal(err)
	// }
	// defer code.Close()
	// fmt.Printf("Encoded: %dB → %dB\n", dim*4, len(code.ToBytes()))

	// Decode
	// decoded, err := q.Decode(code)
	// if err != nil {
	//     log.Fatal(err)
	// }

	// Compute reconstruction error
	// var errSum float64
	// var normSum float64
	// for i := range vector {
	//     diff := float64(vector[i] - decoded[i])
	//     errSum += diff * diff
	//     normSum += float64(vector[i] * vector[i])
	// }
	// relError := math.Sqrt(errSum) / math.Sqrt(normSum)
	// fmt.Printf("Reconstruction error: %.4f\n", relError)

	// Inner product
	// score, err := q.InnerProduct(code, vector)
	// if err != nil {
	//     log.Fatal(err)
	// }
	// fmt.Printf("Inner product: %.4f\n", score)

	// Serialize / deserialize
	// bytes := code.ToBytes()
	// restored, err := bitpolar.CodeFromBytes(bytes)
	// if err != nil {
	//     log.Fatal(err)
	// }
	// defer restored.Close()
	// fmt.Printf("Serialized: %d bytes\n", len(bytes))

	fmt.Println("Go CGO API pattern demonstrated")
	fmt.Printf("Vector dimension: %d, norm: %.4f\n", dim, vectorNorm(vector))
	fmt.Println("Build C lib: cargo build --release --features ffi")
	_ = math.Sqrt(0) // use math package
}

func vectorNorm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x * x)
	}
	return math.Sqrt(sum)
}
