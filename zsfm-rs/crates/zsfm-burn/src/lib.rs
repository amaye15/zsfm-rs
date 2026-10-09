//! Burn-backend inference spike.
//!
//! Parity + timing harness comparing the candle ops in `zsfm-nn` against
//! equivalent Burn implementations on the NdArray backend. No model weights
//! needed: all checks use fixed synthetic inputs so they run offline in CI.
//!
//! Every measurement lands in `benchmark/burn_migration.md`. The gate for
//! each op is max-abs-error below `1e-5` with timing recorded alongside.

pub mod attn;
pub mod bench;
pub mod bridge;
pub mod engine;
pub mod linear;
pub mod norm;

pub use bench::{bench_op, max_abs_err, OpMeasurement};
pub use bridge::{burn_weight_1d, burn_weight_2d};
pub use engine::{engine_from_env, Engine};
