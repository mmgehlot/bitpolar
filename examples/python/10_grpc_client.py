"""BitPolar gRPC Client — connect to the BitPolar compression service.

The gRPC server provides language-agnostic access to BitPolar quantization.
Any language with gRPC support can compress, decompress, and search vectors.

Prerequisites:
    pip install grpcio grpcio-tools numpy
    # Start server: cargo run -p bitpolar-server

Usage:
    python examples/python/10_grpc_client.py
"""

import numpy as np

print("=== BitPolar gRPC Client ===\n")

print("The gRPC server provides these RPCs:")
print("  Encode     — compress a batch of vectors")
print("  Decode     — decompress codes back to vectors")
print("  Search     — search compressed index for nearest neighbors")
print("  AddVectors — add vectors to the in-memory index")
print("  Health     — check server status")
print()

print("Start the server:")
print("  BITPOLAR_DIM=384 BITPOLAR_BITS=4 cargo run -p bitpolar-server")
print()

print("Proto definition: bitpolar-server/proto/bitpolar/v1/service.proto")
print()

# Example client code (requires running server):
#
# import grpc
# from bitpolar.v1 import service_pb2, service_pb2_grpc
#
# channel = grpc.insecure_channel('localhost:50051')
# stub = service_pb2_grpc.VectorCompressionStub(channel)
#
# # Encode vectors
# vectors = np.random.randn(100, 384).astype(np.float32).flatten().tolist()
# response = stub.Encode(service_pb2.EncodeRequest(
#     vectors=vectors, dim=384, count=100, bits=4, seed=42
# ))
# print(f"Compressed {response.original_size}B → {response.compressed_size}B")
#
# # Add vectors to index
# ids = list(range(100))
# stub.AddVectors(service_pb2.AddVectorsRequest(
#     ids=ids, vectors=vectors, dim=384
# ))
#
# # Search
# query = np.random.randn(384).astype(np.float32).tolist()
# results = stub.Search(service_pb2.SearchRequest(query=query, k=10))
# for r in results.results:
#     print(f"  ID={r.id}, score={r.score:.4f}")
#
# # Health check
# health = stub.Health(service_pb2.HealthRequest())
# print(f"Status: {health.status}, index_size: {health.index_size}")

print("gRPC client pattern demonstrated")
print("Generate Python stubs: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service.proto")
