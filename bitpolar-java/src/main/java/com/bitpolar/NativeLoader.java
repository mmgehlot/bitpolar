package com.bitpolar;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Loads the BitPolar JNI native library from JAR resources.
 *
 * <p>Extracts the platform-specific native library from the JAR to a
 * temporary directory, then loads it via {@code System.load()}.
 *
 * <p>Supports Linux (x86_64), macOS (x86_64, aarch64), and Windows (x86_64).
 */
final class NativeLoader {

    private static boolean loaded = false;

    private NativeLoader() {}

    /**
     * Load the native library. Safe to call multiple times (idempotent).
     *
     * @throws RuntimeException if the native library cannot be loaded
     */
    static synchronized void load() {
        if (loaded) return;

        String os = normalizeOs();
        String arch = normalizeArch();
        String libName = mapLibraryName("bitpolar_jni");

        String resourcePath = "/native/" + os + "/" + arch + "/" + libName;

        try (InputStream in = NativeLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                // Fallback: try System.loadLibrary (looks in java.library.path)
                try {
                    System.loadLibrary("bitpolar_jni");
                    loaded = true;
                    return;
                } catch (UnsatisfiedLinkError e) {
                    throw new RuntimeException(
                        "Native library not found in JAR at " + resourcePath +
                        " and not in java.library.path. OS=" + os + " arch=" + arch, e);
                }
            }

            Path tempDir = Files.createTempDirectory("bitpolar-native");
            tempDir.toFile().deleteOnExit();
            Path tempLib = tempDir.resolve(libName);
            tempLib.toFile().deleteOnExit();

            Files.copy(in, tempLib, StandardCopyOption.REPLACE_EXISTING);
            System.load(tempLib.toAbsolutePath().toString());
            loaded = true;
        } catch (IOException e) {
            throw new RuntimeException("Failed to extract native library", e);
        }
    }

    private static String normalizeOs() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("linux")) return "linux";
        if (os.contains("mac") || os.contains("darwin")) return "macos";
        if (os.contains("win")) return "windows";
        return os;
    }

    private static String normalizeArch() {
        String arch = System.getProperty("os.arch", "").toLowerCase();
        if (arch.equals("amd64") || arch.equals("x86_64")) return "x86_64";
        if (arch.equals("aarch64") || arch.equals("arm64")) return "aarch64";
        return arch;
    }

    private static String mapLibraryName(String name) {
        String os = normalizeOs();
        if (os.equals("linux")) return "lib" + name + ".so";
        if (os.equals("macos")) return "lib" + name + ".dylib";
        if (os.equals("windows")) return name + ".dll";
        return "lib" + name + ".so";
    }
}
