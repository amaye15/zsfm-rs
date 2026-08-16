use anyhow::Result;
use candle_core::Tensor;

use crate::linear::linear_nobias;

/// SwiGLU feed-forward: `w2(silu(w1(x)) * w3(x))`, all projections bias-free.
pub fn swiglu_ffn(x: &Tensor, fc1_w: &Tensor, fc2_w: &Tensor, gate_w: &Tensor) -> Result<Tensor> {
    let content = linear_nobias(x, fc1_w)?.silu()?;
    let gate = linear_nobias(x, gate_w)?;
    let h = (content * gate)?;
    linear_nobias(&h, fc2_w)
}
