#[macro_use]
mod base;
mod murmur;
pub mod sse;
pub mod avx;
pub mod vec;
pub use vec::SimdVecImpl;
pub use sse::f32x4;
pub use avx::f32x8;
