//! JNI bridge for BitPolar — implements Java native methods declared in
//! `com.bitpolar.TurboQuantizer`.
//!
//! Produces `libbitpolar_jni.so` / `libbitpolar_jni.dylib` / `bitpolar_jni.dll`
//! which Java loads via `System.loadLibrary("bitpolar_jni")`.

use jni::objects::{JByteArray, JClass, JFloatArray};
use jni::sys::{jfloat, jfloatArray, jint, jlong};
use jni::JNIEnv;

use bitpolar::traits::{SerializableCode, VectorQuantizer};
use bitpolar::TurboQuantizer;

/// Validate and dereference a jlong handle to a TurboQuantizer pointer.
///
/// Returns None and throws a Java exception if the handle is null/zero.
fn quantizer_ref<'a>(env: &mut JNIEnv, handle: jlong) -> Option<&'a TurboQuantizer> {
    if handle == 0 {
        let _ = env.throw_new(
            "java/lang/IllegalStateException",
            "TurboQuantizer already closed (handle is null)",
        );
        return None;
    }
    // Safety: the handle must be a valid pointer returned by nativeNew and not
    // yet freed by nativeFree. Java callers must not use after close(). The
    // reference is bound to lifetime 'a chosen by the caller — it must not
    // outlive the current JNI call.
    Some(unsafe { &*(handle as *const TurboQuantizer) })
}

// ---------------------------------------------------------------------------
// com.bitpolar.TurboQuantizer native methods
// ---------------------------------------------------------------------------

/// `private static native long nativeNew(int dim, int bits, int projections, long seed)`
#[no_mangle]
pub extern "system" fn Java_com_bitpolar_TurboQuantizer_nativeNew(
    mut env: JNIEnv,
    _class: JClass,
    dim: jint,
    bits: jint,
    projections: jint,
    seed: jlong,
) -> jlong {
    let result = TurboQuantizer::new(
        dim as usize,
        bits as u8,
        projections as usize,
        seed as u64,
    );

    match result {
        Ok(q) => Box::into_raw(Box::new(q)) as jlong,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            0
        }
    }
}

/// `private static native void nativeFree(long handle)`
#[no_mangle]
pub extern "system" fn Java_com_bitpolar_TurboQuantizer_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            drop(Box::from_raw(handle as *mut TurboQuantizer));
        }
    }
}

/// `private static native byte[] nativeEncode(long handle, float[] vector)`
#[no_mangle]
pub extern "system" fn Java_com_bitpolar_TurboQuantizer_nativeEncode<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    vector: JFloatArray<'local>,
) -> JByteArray<'local> {
    let q = match quantizer_ref(&mut env, handle) {
        Some(q) => q,
        None => return JByteArray::default(),
    };

    let len = match env.get_array_length(&vector) {
        Ok(l) => l as usize,
        Err(_) => {
            let _ = env.throw_new("java/lang/RuntimeException", "Failed to get array length");
            return JByteArray::default();
        }
    };

    let mut buf = vec![0.0_f32; len];
    if env.get_float_array_region(&vector, 0, &mut buf).is_err() {
        let _ = env.throw_new("java/lang/RuntimeException", "Failed to read float array");
        return JByteArray::default();
    }

    match q.encode(&buf) {
        Ok(code) => {
            let bytes = code.to_compact_bytes();
            match env.byte_array_from_slice(&bytes) {
                Ok(arr) => arr,
                Err(_) => {
                    let _ = env.throw_new("java/lang/RuntimeException", "Failed to create byte array");
                    JByteArray::default()
                }
            }
        }
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            JByteArray::default()
        }
    }
}

/// `private static native float[] nativeDecode(long handle, byte[] code, int dim)`
#[no_mangle]
pub extern "system" fn Java_com_bitpolar_TurboQuantizer_nativeDecode<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    code: JByteArray<'local>,
    dim: jint,
) -> jfloatArray {
    let q = match quantizer_ref(&mut env, handle) {
        Some(q) => q,
        None => return std::ptr::null_mut(),
    };

    let code_bytes = match env.convert_byte_array(code) {
        Ok(b) => b,
        Err(_) => {
            let _ = env.throw_new("java/lang/RuntimeException", "Failed to read byte array");
            return std::ptr::null_mut();
        }
    };

    let turbo_code = match bitpolar::TurboCode::from_compact_bytes(&code_bytes) {
        Ok(c) => c,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            return std::ptr::null_mut();
        }
    };

    let decoded = q.decode(&turbo_code);

    if decoded.len() != dim as usize {
        let _ = env.throw_new(
            "java/lang/RuntimeException",
            format!("Decode length mismatch: expected {}, got {}", dim, decoded.len()),
        );
        return std::ptr::null_mut();
    }

    let result = match env.new_float_array(dim) {
        Ok(arr) => arr,
        Err(_) => {
            let _ = env.throw_new("java/lang/RuntimeException", "Failed to create float array");
            return std::ptr::null_mut();
        }
    };

    if env.set_float_array_region(&result, 0, &decoded).is_err() {
        let _ = env.throw_new("java/lang/RuntimeException", "Failed to write float array");
        return std::ptr::null_mut();
    }

    result.into_raw()
}

/// `private static native float nativeInnerProduct(long handle, byte[] code, float[] query)`
#[no_mangle]
pub extern "system" fn Java_com_bitpolar_TurboQuantizer_nativeInnerProduct(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    code: JByteArray,
    query: JFloatArray,
) -> jfloat {
    let q = match quantizer_ref(&mut env, handle) {
        Some(q) => q,
        None => return 0.0,
    };

    let code_bytes = match env.convert_byte_array(code) {
        Ok(b) => b,
        Err(_) => {
            let _ = env.throw_new("java/lang/RuntimeException", "Failed to read code bytes");
            return 0.0;
        }
    };

    let turbo_code = match bitpolar::TurboCode::from_compact_bytes(&code_bytes) {
        Ok(c) => c,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            return 0.0;
        }
    };

    let len = match env.get_array_length(&query) {
        Ok(l) => l as usize,
        Err(_) => {
            let _ = env.throw_new("java/lang/RuntimeException", "Failed to get query length");
            return 0.0;
        }
    };

    let mut query_buf = vec![0.0_f32; len];
    if env.get_float_array_region(&query, 0, &mut query_buf).is_err() {
        let _ = env.throw_new("java/lang/RuntimeException", "Failed to read query array");
        return 0.0;
    }

    match q.inner_product_estimate(&turbo_code, &query_buf) {
        Ok(score) => score,
        Err(e) => {
            let _ = env.throw_new("java/lang/RuntimeException", e.to_string());
            0.0
        }
    }
}
