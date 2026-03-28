/**
 * BitPolar Java Quick Start — JNI bindings via C FFI.
 *
 * Prerequisites:
 *   1. Build the C library: cargo build --release --features ffi
 *   2. Build JNI bridge: javac -d build bitpolar-java/src/main/java/com/bitpolar/TurboQuantizer.java
 *   3. Set java.library.path to point to target/release/
 *
 * Usage:
 *   javac -cp bitpolar-java/src/main/java examples/java/BitPolarDemo.java -d build
 *   java -Djava.library.path=target/release -cp build BitPolarDemo
 */

import java.util.Random;

public class BitPolarDemo {
    public static void main(String[] args) {
        System.out.println("=== BitPolar Java (JNI) Quick Start ===\n");

        int dim = 128;
        int bits = 4;
        int projections = 32;
        long seed = 42;

        // NOTE: Uncomment when JNI bridge is built.
        // Requires: cargo build --release --features ffi

        // try (com.bitpolar.TurboQuantizer q =
        //         new com.bitpolar.TurboQuantizer(dim, bits, projections, seed)) {
        //
        //     System.out.printf("Quantizer: dim=%d%n", q.getDim());
        //
        //     // Create random vector
        //     Random rng = new Random(42);
        //     float[] vector = new float[dim];
        //     for (int i = 0; i < dim; i++) {
        //         vector[i] = (float) rng.nextGaussian();
        //     }
        //
        //     // Encode
        //     byte[] code = q.encode(vector);
        //     System.out.printf("Encoded: %dB → %dB (%.1fx compression)%n",
        //             dim * 4, code.length, (float)(dim * 4) / code.length);
        //
        //     // Decode
        //     float[] decoded = q.decode(code);
        //     double error = reconstructionError(vector, decoded);
        //     System.out.printf("Reconstruction error: %.4f%n", error);
        //
        //     // Inner product
        //     float score = q.innerProduct(code, vector);
        //     System.out.printf("Inner product: %.4f%n", score);
        // }

        // Demo with simulated data
        Random rng = new Random(42);
        float[] vector = new float[dim];
        for (int i = 0; i < dim; i++) {
            vector[i] = (float) rng.nextGaussian();
        }

        System.out.println("Java JNI API pattern demonstrated");
        System.out.printf("Vector: dim=%d, norm=%.4f%n", dim, norm(vector));
        System.out.println("Build JNI: cargo build --release --features ffi");
    }

    static double norm(float[] v) {
        double sum = 0;
        for (float x : v) sum += x * x;
        return Math.sqrt(sum);
    }

    static double reconstructionError(float[] original, float[] decoded) {
        double errSum = 0, normSum = 0;
        for (int i = 0; i < original.length; i++) {
            double diff = original[i] - decoded[i];
            errSum += diff * diff;
            normSum += original[i] * original[i];
        }
        return Math.sqrt(errSum) / Math.sqrt(normSum);
    }
}
