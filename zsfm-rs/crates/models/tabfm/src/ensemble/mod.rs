pub mod aggregate;
pub mod calibration;
pub mod cat_encoder;
pub mod config_gen;
pub mod nnls;
pub mod oof;
pub mod orchestrate;
pub mod pyrandom;
pub mod scalers;

/// Runs `f` inside a dedicated `rayon` thread pool of `threads` workers, or on rayon's own
/// global default pool (all logical cores, or `RAYON_NUM_THREADS` if set) when `threads` is
/// `None`. Shared by the CLI (`--threads`) and the Python bindings (`n_threads`) so both tune
/// the same underlying `configs.par_iter()` loops in `orchestrate.rs`/`oof.rs`.
///
/// More threads than roughly the physical core count can *hurt* wall time here — Accelerate/BLAS
/// does its own internal matmul threading, and the two can oversubscribe the machine. Benchmark
/// before picking a non-default value for a given deployment.
pub fn with_thread_pool<T>(threads: Option<usize>, f: impl FnOnce() -> anyhow::Result<T> + Send) -> anyhow::Result<T>
where
    T: Send,
{
    match threads {
        Some(n) => rayon::ThreadPoolBuilder::new().num_threads(n).build()?.install(f),
        None => f(),
    }
}
