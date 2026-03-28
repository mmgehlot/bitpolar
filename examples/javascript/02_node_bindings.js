/**
 * BitPolar Node.js Bindings — native NAPI-RS for server-side JavaScript.
 *
 * Prerequisites:
 *   npm install bitpolar
 *   # Requires building: cd bitpolar-node && npm run build
 *
 * Usage:
 *   node examples/javascript/02_node_bindings.js
 */

// const { TurboQuantizer, VectorIndex } = require('bitpolar');

async function main() {
    console.log("=== BitPolar Node.js (NAPI-RS) ===\n");

    // Create quantizer
    // const q = new TurboQuantizer(384, 4, 96, 42);
    // console.log(`Quantizer: dim=${q.dim}, code_size=${q.codeSizeBytes}B`);

    // Encode
    // const vector = new Float32Array(384).map(() => Math.random() - 0.5);
    // const code = q.encode(vector);
    // console.log(`Encoded: ${vector.byteLength}B → ${code.byteLength}B`);

    // Decode
    // const decoded = q.decode(code);
    // console.log(`Decoded: ${decoded.length} floats`);

    // Inner product
    // const score = q.innerProduct(code, vector);
    // console.log(`Self-similarity: ${score.toFixed(4)}`);

    // Build search index
    // const index = new VectorIndex(384, 4, 96, 42);
    // for (let i = 0; i < 1000; i++) {
    //     const vec = new Float32Array(384).map(() => Math.random() - 0.5);
    //     index.add(i, vec);
    // }
    // console.log(`Index: ${index.len} vectors`);

    // Search
    // const query = new Float32Array(384).map(() => Math.random() - 0.5);
    // const results = index.search(query, 10);
    // console.log(`Top 10: ${Array.from(results)}`);

    console.log("Node.js NAPI-RS API pattern demonstrated");
    console.log("Build: cd bitpolar-node && npm run build");
}

main().catch(console.error);
