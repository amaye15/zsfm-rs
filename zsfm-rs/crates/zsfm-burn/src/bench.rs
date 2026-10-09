use std::time::Instant;

/// One timed op comparison: candle time, Burn time, max abs error.
pub struct OpMeasurement {
    pub op: &'static str,
    pub shape: &'static str,
    pub candle_ms: f64,
    pub burn_ms: f64,
    pub max_abs_err: f32,
}

impl OpMeasurement {
    pub fn markdown_row(&self) -> String {
        format!(
            "| {} | {} | {:.3} | {:.3} | {:.2e} |",
            self.op, self.shape, self.candle_ms, self.burn_ms, self.max_abs_err
        )
    }
}

/// Time `f` over `iters` warm-started iterations, return median ms.
pub fn bench_op(iters: usize, mut f: impl FnMut()) -> f64 {
    f(); // warmup
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    times[iters / 2]
}

/// Max absolute elementwise error between two f32 slices.
pub fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}
