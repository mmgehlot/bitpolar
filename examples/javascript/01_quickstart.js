/**
 * BitPolar JavaScript Quick Start — browser-side vector search via WASM.
 *
 * Prerequisites:
 *   npm install bitpolar-wasm
 *   # or: wasm-pack build bitpolar-wasm --target web
 *
 * Usage (Node.js):
 *   node examples/javascript/01_quickstart.js
 *
 * Usage (Browser):
 *   <script type="module" src="01_quickstart.js"></script>
 */

// For browser: import init, { WasmQuantizer, WasmVectorIndex } from 'bitpolar-wasm';
// For Node.js with the npm package:
// const { WasmQuantizer, WasmVectorIndex } = require('bitpolar-wasm');

async function main() {
    console.log("=== BitPolar WASM Quick Start ===\n");

    // NOTE: In actual usage, import from the WASM package.
    // This example shows the API pattern.

    // Create quantizer
    // const q = new WasmQuantizer(128, 4, 32, 42n);

    // Encode a vector
    // const vector = new Float32Array(128).fill(0.1);
    // const code = q.encode(vector);
    // console.log(`Encoded: ${vector.byteLength}B → ${code.byteLength}B`);

    // Decode
    // const decoded = q.decode(code);
    // console.log(`Decoded: ${decoded.length} floats`);

    // Build an index
    // const index = new WasmVectorIndex(128, 4, 32, 42n);
    // for (let i = 0; i < 100; i++) {
    //     const vec = new Float32Array(128).map(() => Math.random() - 0.5);
    //     index.add(i, vec);
    // }
    // console.log(`Index size: ${index.len()} vectors`);

    // Search
    // const query = new Float32Array(128).map(() => Math.random() - 0.5);
    // const results = index.search(query, 5);
    // console.log(`Top 5: ${Array.from(results)}`);

    // Search with scores (returns interleaved [id, score_bits, id, score_bits, ...])
    // const resultsWithScores = index.search_with_scores(query, 5);
    // for (let i = 0; i < resultsWithScores.length; i += 2) {
    //     const id = resultsWithScores[i];
    //     const scoreBits = resultsWithScores[i + 1];
    //     const score = new Float32Array(new Uint32Array([scoreBits]).buffer)[0];
    //     console.log(`  ID=${id}, score=${score.toFixed(4)}`);
    // }

    console.log("WASM API pattern demonstrated (uncomment imports to run)");
    console.log("Build WASM: cd bitpolar-wasm && wasm-pack build --target web");
}

main().catch(console.error);
