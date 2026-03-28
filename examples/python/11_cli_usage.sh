#!/bin/bash
# BitPolar CLI Usage Examples
#
# Prerequisites:
#   cargo install --path bitpolar-cli
#   # or: cargo build -p bitpolar-cli --release
#
# The CLI provides: compress, search, bench, info

echo "=== BitPolar CLI Usage ==="
echo ""

# 1. Create test data (JSON array of float arrays)
echo "Creating test data..."
python3 -c "
import json, random
random.seed(42)
vectors = [[random.gauss(0,1) for _ in range(128)] for _ in range(100)]
with open('/tmp/test_vectors.json', 'w') as f:
    json.dump(vectors, f)
print(f'Created 100 vectors of dim 128')
"

# 2. Compress vectors to .bp format
echo ""
echo "--- Compress ---"
echo "bitpolar compress -i /tmp/test_vectors.json -o /tmp/compressed.bp --bits 4"
# cargo run -p bitpolar-cli -- compress -i /tmp/test_vectors.json -o /tmp/compressed.bp --bits 4

# 3. Show file info
echo ""
echo "--- Info ---"
echo "bitpolar info /tmp/compressed.bp"
# cargo run -p bitpolar-cli -- info /tmp/compressed.bp

# 4. Search
echo ""
echo "--- Search ---"
echo "bitpolar search -i /tmp/compressed.bp -q /tmp/test_vectors.json -k 5"
# cargo run -p bitpolar-cli -- search -i /tmp/compressed.bp -q /tmp/test_vectors.json -k 5

# 5. Benchmark
echo ""
echo "--- Benchmark ---"
echo "bitpolar bench -n 10000 -d 384 --bits 3,4,6,8"
# cargo run -p bitpolar-cli -- bench -n 10000 -d 384 --bits 3,4,6,8

echo ""
echo "Build CLI: cargo build -p bitpolar-cli --release"
echo "Install:   cargo install --path bitpolar-cli"
