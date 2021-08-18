#[macro_use]
mod base;
#[cfg(target_feature = "sse2")]
pub mod sse;
#[cfg(target_feature = "avx2")]
pub mod avx;
pub mod vec;
