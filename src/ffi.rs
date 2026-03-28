//! C FFI layer for bitpolar.
//!
//! Provides a stable C ABI for cross-language bindings (Python via ctypes/cffi,
//! C, C++, Swift, etc.).  Every function uses the `extern "C"` calling convention.
//!
//! # Memory ownership
//!
//! Objects are allocated on the Rust heap and returned as opaque raw pointers.
//! The **caller owns** every non-null pointer returned and is responsible for
//! freeing it via the matching `*_free` function.  Double-free and use-after-free
//! are undefined behaviour — same as in C.
//!
//! # Error codes
//!
//! | Code | Meaning                          |
//! |------|----------------------------------|
//! |    0 | Success                          |
//! |   -1 | Dimension mismatch               |
//! |   -2 | Invalid parameters               |
//! |   -3 | Non-finite input (NaN / Inf)     |
//! |   -4 | Null pointer argument            |
//! |  -99 | Unknown / unexpected error       |
//!
//! # Thread safety
//!
//! Quantizers (`TurboQuantizer`) are immutable after construction and may be
//! shared across threads without synchronization.  Codes (`TurboCode`) are
//! likewise immutable after construction.

// The ffi module is allowed to use unsafe code — it IS the unsafe boundary.
#![allow(unsafe_code)]

use std::panic::catch_unwind;

use crate::{TurboCode, TurboQuantizer, TurboQuantError};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map a [`TurboQuantError`] to an integer error code.
fn err_code(e: &TurboQuantError) -> i32 {
    match e {
        TurboQuantError::DimensionMismatch { .. } => -1,
        TurboQuantError::ZeroDimension
        | TurboQuantError::OddDimension(_)
        | TurboQuantError::InvalidBitWidth(_)
        | TurboQuantError::ZeroProjections
        | TurboQuantError::DimensionTooLarge(_, _)
        | TurboQuantError::EmptyInput(_)
        | TurboQuantError::IndexOutOfBounds { .. } => -2,
        TurboQuantError::NonFiniteInput { .. } => -3,
        TurboQuantError::DeserializationError { .. } => -2,
    }
}

/// Write an error code to `*err_out` if the pointer is non-null.
///
/// # Safety
/// `err_out` must be null or point to a valid `i32`.
#[inline]
unsafe fn write_err(err_out: *mut i32, code: i32) {
    if !err_out.is_null() {
        unsafe { *err_out = code };
    }
}

// ---------------------------------------------------------------------------
// TurboQuantizer lifecycle
// ---------------------------------------------------------------------------

/// Create a new [`TurboQuantizer`].
///
/// # C contract
///
/// ```c
/// TurboQuantizer* q = bitpolar_new(128, 4, 32, 42, &err);
/// if (!q) { /* err contains error code */ }
/// // ... use q ...
/// bitpolar_free(q);
/// ```
///
/// Returns a non-null pointer on success.  On failure writes an error code to
/// `*err_out` (if non-null) and returns null.
///
/// The caller must eventually call [`bitpolar_free`] on a non-null return value.
///
/// # Safety
/// `err_out` must be null or point to a valid `i32`.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_new(
    dim: u32,
    bits: u8,
    projections: u32,
    seed: u64,
    err_out: *mut i32,
) -> *mut TurboQuantizer {
    let result = catch_unwind(|| {
        TurboQuantizer::new(dim as usize, bits, projections as usize, seed)
    });

    match result {
        Ok(Ok(q)) => {
            unsafe { write_err(err_out, 0) };
            Box::into_raw(Box::new(q))
        }
        Ok(Err(e)) => {
            unsafe { write_err(err_out, err_code(&e)) };
            std::ptr::null_mut()
        }
        Err(_) => {
            unsafe { write_err(err_out, -99) };
            std::ptr::null_mut()
        }
    }
}

/// Free a [`TurboQuantizer`] previously created by [`bitpolar_new`].
///
/// Calling this with a null pointer is a no-op (safe).
/// Calling this more than once with the same pointer is undefined behaviour.
///
/// # Safety
/// `ptr` must be null or a pointer previously returned by [`bitpolar_new`]
/// that has not yet been freed.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_free(ptr: *mut TurboQuantizer) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

// ---------------------------------------------------------------------------
// TurboCode lifecycle
// ---------------------------------------------------------------------------

/// Encode a vector using a [`TurboQuantizer`].
///
/// # Arguments
/// - `q` — non-null pointer to a quantizer
/// - `vector` — pointer to `dim` contiguous f32 values
/// - `dim` — number of elements in `vector` (must match the quantizer's dimension)
/// - `err_out` — if non-null, receives an error code on failure
///
/// Returns a non-null [`TurboCode`] pointer on success, null on failure.
/// The caller must free the returned pointer via [`bitpolar_code_free`].
///
/// # Safety
/// - `q` must be a valid non-null pointer to a [`TurboQuantizer`].
/// - `vector` must point to at least `dim` valid f32 values.
/// - `err_out` must be null or point to a valid `i32`.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_encode(
    q: *const TurboQuantizer,
    vector: *const f32,
    dim: u32,
    err_out: *mut i32,
) -> *mut TurboCode {
    if q.is_null() || vector.is_null() {
        unsafe { write_err(err_out, -4) };
        return std::ptr::null_mut();
    }
    if dim == 0 {
        unsafe { write_err(err_out, -2) };
        return std::ptr::null_mut();
    }
    // Validate that dim matches the quantizer's expected dimension before forming the slice.
    let expected_dim = unsafe { (&*q).dim() };
    if dim as usize != expected_dim {
        unsafe { write_err(err_out, -1) };
        return std::ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let quantizer = unsafe { &*q };
        let slice = unsafe { std::slice::from_raw_parts(vector, dim as usize) };
        quantizer.encode(slice)
    });

    match result {
        Ok(Ok(code)) => {
            unsafe { write_err(err_out, 0) };
            Box::into_raw(Box::new(code))
        }
        Ok(Err(e)) => {
            unsafe { write_err(err_out, err_code(&e)) };
            std::ptr::null_mut()
        }
        Err(_) => {
            unsafe { write_err(err_out, -99) };
            std::ptr::null_mut()
        }
    }
}

/// Free a [`TurboCode`] previously returned by [`bitpolar_encode`] or
/// [`bitpolar_code_from_bytes`].
///
/// Calling with null is a no-op.
///
/// # Safety
/// `ptr` must be null or a valid non-freed [`TurboCode`] pointer.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_code_free(ptr: *mut TurboCode) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

// ---------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------

/// Estimate the inner product between a stored code and a raw query vector.
///
/// Writes the result to `*result_out`.
///
/// # Returns
/// `0` on success, or a negative error code.
///
/// # Safety
/// - `q` must be a valid non-null [`TurboQuantizer`] pointer.
/// - `code` must be a valid non-null [`TurboCode`] pointer.
/// - `query` must point to at least `dim` valid f32 values.
/// - `result_out` must be a valid non-null `*mut f32`.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_inner_product(
    q: *const TurboQuantizer,
    code: *const TurboCode,
    query: *const f32,
    dim: u32,
    result_out: *mut f32,
) -> i32 {
    if q.is_null() || code.is_null() || query.is_null() || result_out.is_null() {
        return -4;
    }
    if dim == 0 {
        return -2;
    }
    // Validate dim matches quantizer dimension before forming the slice.
    let expected_dim = unsafe { (&*q).dim() };
    if dim as usize != expected_dim {
        return -1;
    }

    let result = catch_unwind(|| {
        let quantizer = unsafe { &*q };
        let tc = unsafe { &*code };
        let slice = unsafe { std::slice::from_raw_parts(query, dim as usize) };
        quantizer.inner_product_estimate(tc, slice)
    });

    match result {
        Ok(Ok(v)) => {
            unsafe { *result_out = v };
            0
        }
        Ok(Err(e)) => err_code(&e),
        Err(_) => -99,
    }
}

/// Decode a [`TurboCode`] into the caller-provided f32 buffer.
///
/// The buffer must have space for exactly `dim` f32 values (matching the
/// quantizer's dimension).
///
/// # Returns
/// `0` on success, or a negative error code.
///
/// # Safety
/// - `q` must be a valid non-null [`TurboQuantizer`] pointer.
/// - `code` must be a valid non-null [`TurboCode`] pointer.
/// - `out` must point to a writable buffer of at least `dim` f32 values.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_decode(
    q: *const TurboQuantizer,
    code: *const TurboCode,
    out: *mut f32,
    dim: u32,
) -> i32 {
    if q.is_null() || code.is_null() || out.is_null() {
        return -4;
    }

    let result = catch_unwind(|| {
        let quantizer = unsafe { &*q };
        let tc = unsafe { &*code };
        quantizer.decode(tc)
    });

    match result {
        Ok(decoded) => {
            let n = dim as usize;
            if decoded.len() != n {
                return -1;
            }
            unsafe {
                let dst = std::slice::from_raw_parts_mut(out, n);
                dst.copy_from_slice(&decoded);
            }
            0
        }
        Err(_) => -99,
    }
}

// ---------------------------------------------------------------------------
// Compact binary serialization
// ---------------------------------------------------------------------------

/// Serialize a [`TurboCode`] to compact binary bytes.
///
/// If `buf` is null, returns the required byte count without writing anything.
/// If `buf` is non-null, writes up to `buf_len` bytes and returns the actual
/// byte count written.  If the buffer is too small, returns -1.
///
/// # Returns
/// Number of bytes written (or required), or -1 on error.
///
/// # Safety
/// - `code` must be a valid non-null [`TurboCode`] pointer.
/// - If `buf` is non-null it must point to a writable buffer of at least `buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_code_to_bytes(
    code: *const TurboCode,
    buf: *mut u8,
    buf_len: u32,
) -> i32 {
    if code.is_null() {
        return -1;
    }

    let result = catch_unwind(|| {
        let tc = unsafe { &*code };
        tc.to_compact_bytes()
    });

    let bytes = match result {
        Ok(b) => b,
        Err(_) => return -1,
    };

    let needed = bytes.len();

    if buf.is_null() {
        // Caller is querying the required size.
        return needed as i32;
    }

    if (buf_len as usize) < needed {
        return -1;
    }

    unsafe {
        let dst = std::slice::from_raw_parts_mut(buf, needed);
        dst.copy_from_slice(&bytes);
    }
    needed as i32
}

/// Deserialize a [`TurboCode`] from compact binary bytes.
///
/// Returns a non-null pointer on success.  On failure writes an error code to
/// `*err_out` (if non-null) and returns null.
/// The caller must free the returned pointer via [`bitpolar_code_free`].
///
/// # Safety
/// - `buf` must point to at least `len` valid bytes.
/// - `err_out` must be null or point to a valid `i32`.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_code_from_bytes(
    buf: *const u8,
    len: u32,
    err_out: *mut i32,
) -> *mut TurboCode {
    if buf.is_null() {
        unsafe { write_err(err_out, -4) };
        return std::ptr::null_mut();
    }

    let result = catch_unwind(|| {
        let slice = unsafe { std::slice::from_raw_parts(buf, len as usize) };
        TurboCode::from_compact_bytes(slice)
    });

    match result {
        Ok(Ok(code)) => {
            unsafe { write_err(err_out, 0) };
            Box::into_raw(Box::new(code))
        }
        Ok(Err(e)) => {
            unsafe { write_err(err_out, err_code(&e)) };
            std::ptr::null_mut()
        }
        Err(_) => {
            unsafe { write_err(err_out, -99) };
            std::ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

/// Return the vector dimension of a [`TurboQuantizer`].
///
/// Returns 0 if `q` is null.
///
/// # Safety
/// `q` must be null or a valid non-freed [`TurboQuantizer`] pointer.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_dim(q: *const TurboQuantizer) -> u32 {
    if q.is_null() {
        return 0;
    }
    let quantizer = unsafe { &*q };
    quantizer.dim() as u32
}

/// Return the compact byte size of a [`TurboCode`].
///
/// Returns 0 if `code` is null.
///
/// # Safety
/// `code` must be null or a valid non-freed [`TurboCode`] pointer.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_code_size(code: *const TurboCode) -> u32 {
    if code.is_null() {
        return 0;
    }
    let tc = unsafe { &*code };
    tc.size_bytes() as u32
}

// ---------------------------------------------------------------------------
// Batch inner product
// ---------------------------------------------------------------------------

/// Estimate inner products between multiple codes and a single query.
///
/// Writes one f32 score per code to `scores_out`.
///
/// # Returns
/// `0` on success, or a negative error code.
///
/// # Safety
/// - `q` must be a valid non-null [`TurboQuantizer`] pointer.
/// - `codes` must point to an array of `n_codes` non-null [`TurboCode`] pointers.
/// - `query` must point to at least `dim` valid f32 values.
/// - `scores_out` must point to a writable buffer of at least `n_codes` f32 values.
#[no_mangle]
pub unsafe extern "C" fn bitpolar_batch_inner_product(
    q: *const TurboQuantizer,
    codes: *const *const TurboCode,
    n_codes: u32,
    query: *const f32,
    dim: u32,
    scores_out: *mut f32,
) -> i32 {
    if q.is_null() || query.is_null() || scores_out.is_null() {
        return -4;
    }
    if n_codes > 0 && codes.is_null() {
        return -4;
    }
    if dim == 0 && n_codes > 0 {
        return -2;
    }
    // Validate dim matches quantizer dimension before forming the slice.
    if n_codes > 0 {
        let expected_dim = unsafe { (&*q).dim() };
        if dim as usize != expected_dim {
            return -1;
        }
    }

    let result = catch_unwind(|| {
        let quantizer = unsafe { &*q };
        let q_slice = unsafe { std::slice::from_raw_parts(query, dim as usize) };
        let n = n_codes as usize;

        // Validate query once.
        crate::error::validate_finite(q_slice)?;

        // Process each code sequentially.
        let scores_slice = unsafe { std::slice::from_raw_parts_mut(scores_out, n) };
        for (i, score) in scores_slice.iter_mut().enumerate() {
            let code_ptr = unsafe { *codes.add(i) };
            if code_ptr.is_null() {
                return Err(TurboQuantError::EmptyInput("null code pointer in batch"));
            }
            let tc = unsafe { &*code_ptr };
            *score = quantizer.inner_product_estimate(tc, q_slice)?;
        }
        Ok(())
    });

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => err_code(&e),
        Err(_) => -99,
    }
}

// ---------------------------------------------------------------------------
// Tests — exercise the FFI from Rust, simulating a C caller
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn ffi_new_and_free() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        assert_eq!(err, 0);
        assert!(!q.is_null());
        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_free_null_is_noop() {
        // Must not panic or crash.
        unsafe { bitpolar_free(ptr::null_mut()) };
    }

    #[test]
    fn ffi_new_invalid_dim_zero() {
        let mut err: i32 = 0;
        let q = unsafe { bitpolar_new(0, 4, 8, 42, &mut err) };
        assert!(q.is_null());
        assert_eq!(err, -2);
    }

    #[test]
    fn ffi_new_odd_dimension() {
        let mut err: i32 = 0;
        let q = unsafe { bitpolar_new(3, 4, 8, 42, &mut err) };
        assert!(q.is_null());
        assert_eq!(err, -2);
    }

    #[test]
    fn ffi_new_zero_projections() {
        let mut err: i32 = 0;
        let q = unsafe { bitpolar_new(8, 4, 0, 42, &mut err) };
        assert!(q.is_null());
        assert_eq!(err, -2);
    }

    #[test]
    fn ffi_encode_decode_roundtrip() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        assert_eq!(err, 0);
        assert!(!q.is_null());

        let vector = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let code = unsafe { bitpolar_encode(q, vector.as_ptr(), 8, &mut err) };
        assert_eq!(err, 0);
        assert!(!code.is_null());

        let mut decoded = [0.0_f32; 8];
        let rc = unsafe { bitpolar_decode(q, code, decoded.as_mut_ptr(), 8) };
        assert_eq!(rc, 0);
        assert!(decoded.iter().all(|v| v.is_finite()));

        unsafe { bitpolar_code_free(code) };
        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_inner_product() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        assert_eq!(err, 0);

        let vector = [1.0_f32; 8];
        let code = unsafe { bitpolar_encode(q, vector.as_ptr(), 8, &mut err) };
        assert_eq!(err, 0);
        assert!(!code.is_null());

        let query = [0.5_f32; 8];
        let mut result = 0.0_f32;
        let rc = unsafe { bitpolar_inner_product(q, code, query.as_ptr(), 8, &mut result) };
        assert_eq!(rc, 0);
        assert!(result.is_finite());

        unsafe { bitpolar_code_free(code) };
        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_inner_product_null_returns_error() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        assert!(!q.is_null());

        let query = [0.5_f32; 8];
        let mut result = 0.0_f32;
        // null code pointer
        let rc = unsafe {
            bitpolar_inner_product(q, ptr::null(), query.as_ptr(), 8, &mut result)
        };
        assert_eq!(rc, -4);

        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_encode_null_returns_null() {
        let mut err: i32 = 0;
        // null quantizer
        let code = unsafe {
            bitpolar_encode(ptr::null(), [0.0_f32; 8].as_ptr(), 8, &mut err)
        };
        assert!(code.is_null());
        assert_eq!(err, -4);
    }

    #[test]
    fn ffi_encode_dim_mismatch() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        assert_eq!(err, 0);

        // vector has 4 elements but quantizer expects 8 → DimensionMismatch
        let short = [0.0_f32; 4];
        let code = unsafe { bitpolar_encode(q, short.as_ptr(), 4, &mut err) };
        assert!(code.is_null());
        assert_eq!(err, -1);

        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_encode_nan_rejected() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        assert_eq!(err, 0);

        let mut v = [0.1_f32; 8];
        v[3] = f32::NAN;
        let code = unsafe { bitpolar_encode(q, v.as_ptr(), 8, &mut err) };
        assert!(code.is_null());
        assert_eq!(err, -3);

        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_code_to_bytes_and_from_bytes() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        assert_eq!(err, 0);

        let vector = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let code = unsafe { bitpolar_encode(q, vector.as_ptr(), 8, &mut err) };
        assert_eq!(err, 0);
        assert!(!code.is_null());

        // Query required size (null buffer).
        let needed = unsafe { bitpolar_code_to_bytes(code, ptr::null_mut(), 0) };
        assert!(needed > 0, "required size must be positive");

        // Write bytes.
        let mut buf = vec![0u8; needed as usize];
        let written = unsafe {
            bitpolar_code_to_bytes(code, buf.as_mut_ptr(), buf.len() as u32)
        };
        assert_eq!(written, needed);

        // Deserialize.
        let code2 =
            unsafe { bitpolar_code_from_bytes(buf.as_ptr(), buf.len() as u32, &mut err) };
        assert_eq!(err, 0);
        assert!(!code2.is_null());

        // Verify the two codes serialize identically.
        let needed2 = unsafe { bitpolar_code_to_bytes(code2, ptr::null_mut(), 0) };
        assert_eq!(needed, needed2);
        let mut buf2 = vec![0u8; needed2 as usize];
        unsafe { bitpolar_code_to_bytes(code2, buf2.as_mut_ptr(), buf2.len() as u32) };
        assert_eq!(buf, buf2, "deserialized code must re-serialize identically");

        unsafe { bitpolar_code_free(code) };
        unsafe { bitpolar_code_free(code2) };
        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_code_to_bytes_buf_too_small() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        let vector = [0.1_f32; 8];
        let code = unsafe { bitpolar_encode(q, vector.as_ptr(), 8, &mut err) };
        assert!(!code.is_null());

        // Buffer of size 1 — always too small.
        let mut tiny = [0u8; 1];
        let rc = unsafe { bitpolar_code_to_bytes(code, tiny.as_mut_ptr(), 1) };
        assert_eq!(rc, -1, "should return -1 when buffer is too small");

        unsafe { bitpolar_code_free(code) };
        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_code_from_bytes_null_buf() {
        let mut err: i32 = 0;
        let code = unsafe { bitpolar_code_from_bytes(ptr::null(), 0, &mut err) };
        assert!(code.is_null());
        assert_eq!(err, -4);
    }

    #[test]
    fn ffi_code_from_bytes_corrupted() {
        let mut err: i32 = 0;
        let garbage = [0xFFu8; 16];
        let code =
            unsafe { bitpolar_code_from_bytes(garbage.as_ptr(), garbage.len() as u32, &mut err) };
        assert!(code.is_null());
        assert_ne!(err, 0);
    }

    #[test]
    fn ffi_dim() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        assert_eq!(err, 0);
        assert_eq!(unsafe { bitpolar_dim(q) }, 8);
        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_dim_null() {
        assert_eq!(unsafe { bitpolar_dim(ptr::null()) }, 0);
    }

    #[test]
    fn ffi_code_size() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        let vector = [0.1_f32; 8];
        let code = unsafe { bitpolar_encode(q, vector.as_ptr(), 8, &mut err) };
        assert!(!code.is_null());

        let sz = unsafe { bitpolar_code_size(code) };
        assert!(sz > 0);

        unsafe { bitpolar_code_free(code) };
        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_code_size_null() {
        assert_eq!(unsafe { bitpolar_code_size(ptr::null()) }, 0);
    }

    #[test]
    fn ffi_batch_inner_product() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        assert_eq!(err, 0);

        let vecs = [
            [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8_f32, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.5_f32; 8],
        ];

        let codes: Vec<*const TurboCode> = vecs
            .iter()
            .map(|v| {
                let c = unsafe { bitpolar_encode(q, v.as_ptr(), 8, &mut err) };
                assert_eq!(err, 0);
                c as *const TurboCode
            })
            .collect();

        let query = [0.3_f32; 8];
        let mut scores = [0.0_f32; 3];

        let rc = unsafe {
            bitpolar_batch_inner_product(
                q,
                codes.as_ptr(),
                codes.len() as u32,
                query.as_ptr(),
                8,
                scores.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0);
        assert!(scores.iter().all(|s| s.is_finite()));

        // Verify scores match sequential single-code calls.
        for (i, &code_ptr) in codes.iter().enumerate() {
            let mut single = 0.0_f32;
            let rc2 = unsafe {
                bitpolar_inner_product(q, code_ptr, query.as_ptr(), 8, &mut single)
            };
            assert_eq!(rc2, 0);
            assert!(
                (scores[i] - single).abs() < 1e-5,
                "batch score[{i}]={} != sequential={single}",
                scores[i]
            );
        }

        for &c in &codes {
            unsafe { bitpolar_code_free(c as *mut TurboCode) };
        }
        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_batch_inner_product_empty() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        assert_eq!(err, 0);

        let query = [0.3_f32; 8];
        let mut scores: [f32; 0] = [];
        // Pass null for codes array (n_codes=0 — no codes to process).
        let rc = unsafe {
            bitpolar_batch_inner_product(
                q,
                ptr::null(),
                0,
                query.as_ptr(),
                8,
                scores.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0);

        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_batch_inner_product_null_query() {
        let mut err: i32 = -99;
        let q = unsafe { bitpolar_new(8, 4, 8, 42, &mut err) };
        let vector = [0.1_f32; 8];
        let code = unsafe { bitpolar_encode(q, vector.as_ptr(), 8, &mut err) } as *const TurboCode;
        let codes = [code];
        let mut scores = [0.0_f32; 1];

        let rc = unsafe {
            bitpolar_batch_inner_product(q, codes.as_ptr(), 1, ptr::null(), 8, scores.as_mut_ptr())
        };
        assert_eq!(rc, -4);

        unsafe { bitpolar_code_free(code as *mut TurboCode) };
        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_err_out_null_is_safe() {
        // Passing null err_out must not crash.
        let q = unsafe { bitpolar_new(8, 4, 8, 42, ptr::null_mut()) };
        assert!(!q.is_null());
        unsafe { bitpolar_free(q) };
    }

    #[test]
    fn ffi_new_null_errout() {
        // Invalid params with null err_out — must not crash.
        let q = unsafe { bitpolar_new(0, 4, 8, 42, ptr::null_mut()) };
        assert!(q.is_null()); // still returns null on error
    }
}
