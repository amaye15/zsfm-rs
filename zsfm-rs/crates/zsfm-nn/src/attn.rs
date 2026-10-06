use anyhow::Result;
use candle_core::Tensor;

/// Scaled dot-product attention with additive mask:
/// `softmax(Q K^T * scale + mask) V`.
///
/// Shapes: `q/k/v` are `[batch, n_heads, seq_q/kv, head_dim]`,
/// `mask` broadcasts to `[batch, n_heads, seq_q, seq_kv]`.
/// Shared by Mitra and TabDPT attention paths (scale + mask args cover both).
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let attn = (q.matmul(&k.t()?)? * scale)?;
    let attn = match mask {
        Some(m) => attn.broadcast_add(m)?,
        None => attn,
    };
    let probs = candle_nn::ops::softmax_last_dim(&attn)?;
    Ok(probs.matmul(v)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn uniform_attention_averages_values() {
        let device = Device::Cpu;
        // One head, seq=2, dim=1. Q=K=0 → uniform weights → mean of V.
        let q = Tensor::zeros((1, 1, 2, 1), candle_core::DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 1, 2, 1), candle_core::DType::F32, &device).unwrap();
        let v = Tensor::from_vec(vec![2.0f32, 4.0], (1, 1, 2, 1), &device).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v, 1.0, None).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-5);
        assert!((vals[1] - 3.0).abs() < 1e-5);
    }
}
