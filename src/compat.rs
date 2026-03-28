//! Compatibility layer for `std`/`alloc`/`no_std` builds.
//!
//! Centralizes all conditional imports so individual modules don't need
//! `#[cfg(...)]` on every `use` statement. All modules import from here.

// Re-export Vec from the appropriate source
#[cfg(feature = "std")]
pub use std::vec::Vec;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
pub use alloc::vec::Vec;

// Re-export vec! macro
#[cfg(feature = "std")]
pub use std::vec;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
pub use alloc::vec;

// Re-export String
#[cfg(feature = "std")]
pub use std::string::String;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
pub use alloc::string::String;

// Re-export format!
#[cfg(feature = "std")]
pub use std::format;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
pub use alloc::format;

/// Math functions that work in both `std` and `no_std` environments.
///
/// In `std` builds, these delegate to the standard library's inherent methods.
/// In `no_std` builds, they use the `libm` crate.
pub mod math {
    /// Square root of a 32-bit float.
    #[inline(always)]
    pub fn sqrtf(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::sqrtf(x)
        }
    }

    /// Arctangent of y/x, returning radians in [-pi, pi].
    #[inline(always)]
    pub fn atan2f(y: f32, x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            y.atan2(x)
        }
        #[cfg(not(feature = "std"))]
        {
            libm::atan2f(y, x)
        }
    }

    /// Sine of a 32-bit float (radians).
    #[inline(always)]
    pub fn sinf(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.sin()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::sinf(x)
        }
    }

    /// Cosine of a 32-bit float (radians).
    #[inline(always)]
    pub fn cosf(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.cos()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::cosf(x)
        }
    }

    /// Absolute value of a 32-bit float.
    #[inline(always)]
    pub fn fabsf(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.abs()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::fabsf(x)
        }
    }

    /// Exponential (e^x) of a 32-bit float.
    #[inline(always)]
    pub fn expf(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.exp()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::expf(x)
        }
    }

    /// Floor of a 32-bit float.
    #[inline(always)]
    pub fn floorf(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.floor()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::floorf(x)
        }
    }

    /// Ceiling of a 32-bit float.
    #[inline(always)]
    pub fn ceilf(x: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            x.ceil()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::ceilf(x)
        }
    }

    /// Square root of a 64-bit float.
    #[inline(always)]
    pub fn sqrt(x: f64) -> f64 {
        #[cfg(feature = "std")]
        {
            x.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::sqrt(x)
        }
    }

    /// Natural exponential of a 64-bit float.
    #[inline(always)]
    pub fn exp(x: f64) -> f64 {
        #[cfg(feature = "std")]
        {
            x.exp()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::exp(x)
        }
    }
}
