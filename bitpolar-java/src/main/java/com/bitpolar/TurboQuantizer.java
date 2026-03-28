package com.bitpolar;

/**
 * Java bindings for BitPolar vector quantization via JNI.
 *
 * <p>Usage:
 * <pre>{@code
 * TurboQuantizer q = new TurboQuantizer(128, 4, 32, 42);
 * byte[] code = q.encode(new float[]{0.1f, 0.2f, ...});
 * float score = q.innerProduct(code, queryVector);
 * q.close();
 * }</pre>
 *
 * <p>Requires the native library to be loaded:
 * {@code System.loadLibrary("bitpolar")}
 */
public class TurboQuantizer implements AutoCloseable {

    // Native pointer to the Rust TurboQuantizer (opaque handle)
    private long nativeHandle;
    private final int dim;

    static {
        NativeLoader.load();
    }

    /**
     * Create a new TurboQuantizer.
     *
     * @param dim        Vector dimension (must be even)
     * @param bits       Quantization precision (3-8)
     * @param projections Number of QJL projections (typically dim/4)
     * @param seed       Random seed for deterministic behavior
     * @throws RuntimeException if construction fails
     */
    public TurboQuantizer(int dim, int bits, int projections, long seed) {
        this.dim = dim;
        this.nativeHandle = nativeNew(dim, bits, projections, seed);
        if (this.nativeHandle == 0) {
            throw new RuntimeException(
                    "Failed to create TurboQuantizer (dim=" + dim + ", bits=" + bits + ")");
        }
    }

    /**
     * Encode a float vector to compressed bytes.
     *
     * @param vector float array of length {@code dim}
     * @return compressed code as byte array
     */
    public byte[] encode(float[] vector) {
        if (vector.length != dim) {
            throw new IllegalArgumentException(
                    "Expected " + dim + " dims, got " + vector.length);
        }
        return nativeEncode(nativeHandle, vector);
    }

    /**
     * Decode compressed bytes back to an approximate float vector.
     *
     * @param code byte array from {@link #encode(float[])}
     * @return float array of length {@code dim}
     */
    public float[] decode(byte[] code) {
        return nativeDecode(nativeHandle, code, dim);
    }

    /**
     * Estimate inner product between a compressed code and a query vector.
     *
     * @param code  byte array from {@link #encode(float[])}
     * @param query float array of length {@code dim}
     * @return approximate inner product score
     */
    public float innerProduct(byte[] code, float[] query) {
        if (query.length != dim) {
            throw new IllegalArgumentException(
                    "Query expected " + dim + " dims, got " + query.length);
        }
        return nativeInnerProduct(nativeHandle, code, query);
    }

    /** Vector dimension. */
    public int getDim() {
        return dim;
    }

    /** Free native resources. */
    @Override
    public void close() {
        if (nativeHandle != 0) {
            nativeFree(nativeHandle);
            nativeHandle = 0;
        }
    }

    // JNI native methods (implemented in Rust via jni crate or C FFI bridge)
    private static native long nativeNew(int dim, int bits, int projections, long seed);
    private static native void nativeFree(long handle);
    private static native byte[] nativeEncode(long handle, float[] vector);
    private static native float[] nativeDecode(long handle, byte[] code, int dim);
    private static native float nativeInnerProduct(long handle, byte[] code, float[] query);
}
