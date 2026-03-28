package com.bitpolar.spring;

import com.bitpolar.TurboQuantizer;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Spring AI-compatible vector store backed by BitPolar compression.
 *
 * <p>Stores document embeddings in compressed form using {@link TurboQuantizer}
 * and supports similarity search via approximate inner product scoring.
 * Thread-safe for concurrent reads and writes.
 *
 * <p>Usage:
 * <pre>{@code
 * BitPolarVectorStore store = new BitPolarVectorStore(384, 4, 42);
 *
 * // Add documents
 * List<Map<String, Object>> docs = List.of(
 *     Map.of("id", "doc1", "embedding", new float[]{...}, "content", "Hello world"),
 *     Map.of("id", "doc2", "embedding", new float[]{...}, "content", "Goodbye")
 * );
 * store.add(docs);
 *
 * // Search
 * float[] query = new float[]{...};
 * List<Map<String, Object>> results = store.similaritySearch(query, 10);
 * }</pre>
 */
public class BitPolarVectorStore implements AutoCloseable {

    private final TurboQuantizer quantizer;
    private final int dim;
    private final int bits;

    /**
     * Stored documents: id -> StoredEntry (compressed code + metadata).
     */
    private final ConcurrentHashMap<String, StoredEntry> store;

    /**
     * Ordered list of IDs for index-based access during search.
     * Synchronized on itself for mutation.
     */
    private final List<String> idIndex;

    /**
     * Internal container for a stored document's compressed data and metadata.
     */
    private static final class StoredEntry {
        final String id;
        final byte[] code;
        final Map<String, Object> metadata;

        StoredEntry(String id, byte[] code, Map<String, Object> metadata) {
            this.id = id;
            this.code = code;
            this.metadata = metadata;
        }
    }

    /**
     * Create a new BitPolarVectorStore.
     *
     * @param dim  vector dimension (must be even)
     * @param bits quantization precision (3-8)
     * @param seed random seed for deterministic behavior
     */
    public BitPolarVectorStore(int dim, int bits, long seed) {
        if (bits < 3 || bits > 8) {
            throw new IllegalArgumentException("bits must be 3-8, got " + bits);
        }
        this.dim = dim;
        this.bits = bits;
        int projections = Math.max(dim / 4, 1);
        this.quantizer = new TurboQuantizer(dim, bits, projections, seed);
        this.store = new ConcurrentHashMap<>();
        this.idIndex = Collections.synchronizedList(new ArrayList<>());
    }

    /**
     * Create a store with default seed (42).
     *
     * @param dim  vector dimension
     * @param bits quantization precision (3-8)
     */
    public BitPolarVectorStore(int dim, int bits) {
        this(dim, bits, 42L);
    }

    /**
     * Add documents to the store.
     *
     * <p>Each document map must contain:
     * <ul>
     *   <li>{@code "id"} — unique string identifier</li>
     *   <li>{@code "embedding"} — float[] vector of length {@code dim}</li>
     * </ul>
     *
     * <p>Optional fields:
     * <ul>
     *   <li>{@code "content"} — text content (String)</li>
     *   <li>Any other key-value metadata pairs</li>
     * </ul>
     *
     * <p>If a document with the same ID already exists, it is overwritten.
     *
     * @param documents list of document maps
     * @throws IllegalArgumentException if id or embedding is missing/invalid
     */
    public void add(List<Map<String, Object>> documents) {
        for (Map<String, Object> doc : documents) {
            Object idObj = doc.get("id");
            if (idObj == null) {
                throw new IllegalArgumentException("Document missing 'id' field");
            }
            String id = idObj.toString();

            Object embObj = doc.get("embedding");
            if (embObj == null) {
                throw new IllegalArgumentException(
                        "Document '" + id + "' missing 'embedding' field");
            }

            float[] embedding = toFloatArray(embObj);
            String docId = id;
            if (embedding.length == 0) {
                throw new IllegalArgumentException("Empty embedding not allowed for document " + docId);
            }
            if (embedding.length != dim) {
                throw new IllegalArgumentException(
                        "Document '" + id + "' embedding has " + embedding.length
                                + " dims, expected " + dim);
            }

            // Compress the embedding
            byte[] code = quantizer.encode(embedding);

            // Build metadata (everything except id and embedding)
            Map<String, Object> metadata = new LinkedHashMap<>();
            for (Map.Entry<String, Object> entry : doc.entrySet()) {
                String key = entry.getKey();
                if (!"id".equals(key) && !"embedding".equals(key)) {
                    metadata.put(key, entry.getValue());
                }
            }

            StoredEntry stored = new StoredEntry(id, code, metadata);
            boolean isNew = !store.containsKey(id);
            store.put(id, stored);
            if (isNew) {
                idIndex.add(id);
            }
        }
    }

    /**
     * Delete documents by ID.
     *
     * @param ids list of document IDs to remove
     * @return list of IDs that were actually present and removed
     */
    public List<String> delete(List<String> ids) {
        List<String> removed = new ArrayList<>();
        for (String id : ids) {
            StoredEntry entry = store.remove(id);
            if (entry != null) {
                removed.add(id);
                idIndex.remove(id);
            }
        }
        return removed;
    }

    /**
     * Perform similarity search against stored embeddings.
     *
     * <p>Scores each stored embedding against the query using BitPolar's
     * approximate inner product and returns the top-K results sorted
     * by score descending.
     *
     * @param query float[] query vector of length {@code dim}
     * @param topK  number of results to return
     * @return list of result maps, each containing:
     *         "id" (String), "score" (Float), plus all stored metadata
     * @throws IllegalArgumentException if query dimension is wrong
     */
    public List<Map<String, Object>> similaritySearch(float[] query, int topK) {
        if (query.length != dim) {
            throw new IllegalArgumentException(
                    "Query has " + query.length + " dims, expected " + dim);
        }

        // Snapshot the current entries for consistent iteration
        List<Map.Entry<String, StoredEntry>> entries;
        synchronized (idIndex) {
            entries = new ArrayList<>();
            for (String id : idIndex) {
                StoredEntry entry = store.get(id);
                if (entry != null) {
                    entries.add(Map.entry(id, entry));
                }
            }
        }

        if (entries.isEmpty()) {
            return Collections.emptyList();
        }

        // Score all entries
        int n = entries.size();
        float[] scores = new float[n];
        for (int i = 0; i < n; i++) {
            scores[i] = quantizer.innerProduct(entries.get(i).getValue().code, query);
        }

        // Top-K selection using partial sort
        int k = Math.min(topK, n);
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }

        // Sort by score descending and take top k
        Arrays.sort(indices, (a, b) -> Float.compare(scores[b], scores[a]));

        List<Map<String, Object>> results = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            int idx = indices[i];
            StoredEntry entry = entries.get(idx).getValue();

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("id", entry.id);
            result.put("score", scores[idx]);
            result.putAll(entry.metadata);
            results.add(result);
        }

        return results;
    }

    /**
     * Get a document by ID.
     *
     * @param id document ID
     * @return optional map with "id", decompressed "embedding", and metadata;
     *         empty if not found
     */
    public Optional<Map<String, Object>> get(String id) {
        StoredEntry entry = store.get(id);
        if (entry == null) {
            return Optional.empty();
        }

        Map<String, Object> doc = new LinkedHashMap<>();
        doc.put("id", entry.id);
        doc.put("embedding", quantizer.decode(entry.code));
        doc.putAll(entry.metadata);
        return Optional.of(doc);
    }

    /**
     * Return the number of stored documents.
     *
     * @return count of documents
     */
    public int size() {
        return store.size();
    }

    /**
     * Return all stored document IDs.
     *
     * @return unmodifiable list of IDs in insertion order
     */
    public List<String> listIds() {
        synchronized (idIndex) {
            return Collections.unmodifiableList(new ArrayList<>(idIndex));
        }
    }

    /**
     * Remove all documents from the store.
     */
    public void clear() {
        store.clear();
        synchronized (idIndex) {
            idIndex.clear();
        }
    }

    /**
     * Check if a document with the given ID exists.
     *
     * @param id document ID
     * @return true if present
     */
    public boolean contains(String id) {
        return store.containsKey(id);
    }

    /**
     * Get the configured vector dimension.
     *
     * @return vector dimension
     */
    public int getDim() {
        return dim;
    }

    /**
     * Get the configured quantization bits.
     *
     * @return bits
     */
    public int getBits() {
        return bits;
    }

    /**
     * Close the store, releasing the native quantizer resources.
     */
    @Override
    public void close() {
        quantizer.close();
        store.clear();
        synchronized (idIndex) {
            idIndex.clear();
        }
    }

    @Override
    public String toString() {
        return "BitPolarVectorStore{dim=" + dim + ", bits=" + bits
                + ", documents=" + store.size() + "}";
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /**
     * Convert an Object (float[], double[], List) to float[].
     */
    private static float[] toFloatArray(Object obj) {
        if (obj instanceof float[]) {
            return (float[]) obj;
        }
        if (obj instanceof double[]) {
            double[] d = (double[]) obj;
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) {
                f[i] = (float) d[i];
            }
            return f;
        }
        if (obj instanceof List) {
            @SuppressWarnings("unchecked")
            List<? extends Number> list = (List<? extends Number>) obj;
            float[] f = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                f[i] = list.get(i).floatValue();
            }
            return f;
        }
        throw new IllegalArgumentException(
                "Cannot convert " + obj.getClass().getName() + " to float[]");
    }
}
