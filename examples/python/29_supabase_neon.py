"""BitPolar + Supabase/Neon — serverless Postgres vector compression.

Demonstrates storing BitPolar-compressed vectors in Postgres with
pgvector-compatible patterns. Falls back to SQLite for offline testing.

Prerequisites:
    pip install bitpolar numpy psycopg2-binary

Note: Requires running Postgres instance for full demo.

Usage:
    python examples/python/29_supabase_neon.py
"""

import numpy as np
import sqlite3
import bitpolar

DIM = 384
q = bitpolar.TurboQuantizer(dim=DIM, bits=4, projections=64, seed=42)

# =============================================================================
# Postgres SQL Setup (Supabase / Neon)
# =============================================================================
print("=== Supabase/Neon SQL Setup ===\n")

SETUP_SQL = """
-- Enable pgvector (Supabase has this pre-installed)
CREATE EXTENSION IF NOT EXISTS vector;

-- Table with compressed BitPolar codes alongside optional full vectors
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),          -- optional: full vector for exact search
    bitpolar_code BYTEA NOT NULL,   -- compressed BitPolar code
    metadata JSONB DEFAULT '{}'
);

-- Index on content for hybrid search
CREATE INDEX idx_documents_content ON documents USING gin(to_tsvector('english', content));
"""
print(SETUP_SQL)

# =============================================================================
# Connection Patterns
# =============================================================================
print("=== Connection Patterns ===\n")

# Supabase:
# import psycopg2
# conn = psycopg2.connect("postgresql://postgres:[password]@db.[ref].supabase.co:5432/postgres")

# Neon:
# conn = psycopg2.connect("postgresql://[user]:[password]@[endpoint].neon.tech/neondb?sslmode=require")

# =============================================================================
# SQLite Fallback Demo (runs offline)
# =============================================================================
print("=== SQLite Fallback Demo ===\n")

conn = sqlite3.connect(":memory:")
conn.execute("""CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    bitpolar_code BLOB NOT NULL,
    metadata TEXT DEFAULT '{}'
)""")

docs = [
    "BitPolar achieves 8x compression on OpenAI embeddings",
    "Supabase provides managed Postgres with pgvector support",
    "Neon offers serverless Postgres with autoscaling",
    "Vector search enables semantic retrieval in RAG pipelines",
    "Compressed vectors reduce storage costs by 87%",
    "BitPolar works without training or calibration data",
]

for doc in docs:
    embedding = np.random.randn(DIM).astype(np.float32)
    code = q.encode(embedding)
    conn.execute("INSERT INTO documents (content, bitpolar_code) VALUES (?, ?)", [doc, code])
conn.commit()

count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
print(f"Inserted {count} documents")

query_vec = np.random.randn(DIM).astype(np.float32)
rows = conn.execute("SELECT id, content, bitpolar_code FROM documents").fetchall()
scored = [(r[0], r[1], q.inner_product(bytes(r[2]), query_vec)) for r in rows]
scored.sort(key=lambda x: x[2], reverse=True)

print(f"\nSemantic search results:")
for doc_id, content, score in scored[:3]:
    print(f"  [{doc_id}] score={score:.4f} | {content}")

# Storage comparison
orig_bytes = count * DIM * 4
comp_bytes = sum(len(r[2]) for r in conn.execute("SELECT bitpolar_code FROM documents").fetchall())
print(f"\nStorage: {orig_bytes:,}B (float32) vs {comp_bytes:,}B (BitPolar) = {orig_bytes/comp_bytes:.1f}x")
