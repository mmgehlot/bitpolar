"""BitPolar Vector Database Integrations — Milvus, Weaviate, Pinecone, Redis, Elasticsearch, DuckDB, SQLite.

Prerequisites:
    pip install bitpolar numpy duckdb

Note: Only DuckDB and SQLite examples run without external servers.

Usage:
    python examples/python/23_vector_databases.py
"""

import numpy as np
import struct
import bitpolar

DIM = 128
q = bitpolar.TurboQuantizer(dim=DIM, bits=4, projections=32, seed=42)

# =============================================================================
# DuckDB Vector Store
# =============================================================================
print("=== DuckDB Vector Store ===\n")

try:
    import duckdb

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE vectors (id INTEGER PRIMARY KEY, code BLOB, label TEXT)")

    vectors = {i: np.random.randn(DIM).astype(np.float32) for i in range(20)}
    for vid, vec in vectors.items():
        code = q.encode(vec)
        con.execute("INSERT INTO vectors VALUES (?, ?, ?)", [vid, code, f"doc_{vid}"])
    print(f"Added {len(vectors)} vectors to DuckDB")

    count = con.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    print(f"Count: {count}")

    query = np.random.randn(DIM).astype(np.float32)
    rows = con.execute("SELECT id, code, label FROM vectors").fetchall()
    scores = [(r[0], q.inner_product(r[1], query), r[2]) for r in rows]
    scores.sort(key=lambda x: x[1], reverse=True)
    print("Top 3 search results:")
    for vid, score, label in scores[:3]:
        print(f"  id={vid}, score={score:.4f}, label={label}")

    con.execute("DELETE FROM vectors WHERE id = 0")
    count = con.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    print(f"After delete: {count} vectors\n")

except ImportError:
    print("duckdb not installed — pip install duckdb\n")

# =============================================================================
# SQLite Vector Store
# =============================================================================
print("=== SQLite Vector Store ===\n")

import sqlite3

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE vectors (id INTEGER PRIMARY KEY, code BLOB, label TEXT)")

for vid, vec in vectors.items():
    code = q.encode(vec)
    conn.execute("INSERT INTO vectors VALUES (?, ?, ?)", [vid, code, f"item_{vid}"])
conn.commit()
print(f"Added {len(vectors)} vectors to SQLite")

count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
print(f"Count: {count}")

query = np.random.randn(DIM).astype(np.float32)
rows = conn.execute("SELECT id, code, label FROM vectors").fetchall()
scores = [(r[0], q.inner_product(bytes(r[1]), query), r[2]) for r in rows]
scores.sort(key=lambda x: x[1], reverse=True)
print("Top 3 search results:")
for vid, score, label in scores[:3]:
    print(f"  id={vid}, score={score:.4f}, label={label}")

conn.execute("DELETE FROM vectors WHERE id = 0")
conn.commit()
count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
print(f"After delete: {count} vectors")

# =============================================================================
# Patterns for external vector DBs (require running servers)
# =============================================================================

# --- Milvus ---
# from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
# schema = CollectionSchema([
#     FieldSchema("id", DataType.INT64, is_primary=True),
#     FieldSchema("code", DataType.BINARY_VECTOR, dim=q.code_size_bytes * 8),
# ])
# collection = Collection("bitpolar_demo", schema)
# collection.insert([[0], [q.encode(vec)]])

# --- Weaviate ---
# import weaviate
# client = weaviate.Client("http://localhost:8080")
# client.data_object.create({"code": base64.b64encode(code).decode()}, "BitPolarVector")

# --- Pinecone ---
# import pinecone; pinecone.init(api_key="...", environment="...")
# index = pinecone.Index("bitpolar"); index.upsert([("id0", decoded.tolist())])

# --- Redis ---
# import redis; r = redis.Redis()
# r.hset("vec:0", mapping={"code": code, "label": "doc_0"})

# --- Elasticsearch ---
# from elasticsearch import Elasticsearch
# es = Elasticsearch(); es.index(index="vectors", body={"code": code.hex()})
