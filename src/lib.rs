pub mod lsh;
pub mod simd;
pub mod net;
//mod expressionlib;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

