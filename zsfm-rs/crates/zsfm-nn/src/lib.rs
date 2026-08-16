//! Shared candle tensor primitives duplicated (byte-for-byte, or modulo a
//! hardcoded-vs-parameterized constant) across the per-model `infer/mod.rs`
//! files in `crates/models/*`. Extracted so the arithmetic lives in one
//! place; every call site keeps producing bit-identical results to its
//! former private copy.

mod ffn;
mod gguf;
mod linear;
mod mask;
mod norm;

pub use ffn::swiglu_ffn;
pub use gguf::{load_tensor, load_vec, load_weight, try_load_tensor};
pub use linear::{linear, linear_bias, linear_nobias};
pub use mask::make_causal_mask;
pub use norm::{layer_norm, rms_norm};
