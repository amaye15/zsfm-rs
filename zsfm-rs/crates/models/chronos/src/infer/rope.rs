//! Standard Llama-style Rotary Position Embedding (RoPE).
//!
//! rotate_half(x) = cat([-x[half:], x[:half]], dim=-1)
//! cos/sin are precomputed at construction time.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub struct RopeCache {
    /// Precomputed cosines: [max_seq, head_dim]
    cos: Tensor,
    /// Precomputed sines:   [max_seq, head_dim]
    sin: Tensor,
}

impl RopeCache {
    /// Build cos/sin tables for positions [0, max_seq) with the given theta and head_dim.
    pub fn new(head_dim: usize, max_seq: usize, theta: f64, device: &Device) -> Result<Self> {
        // inv_freq[i] = 1 / theta^(2i / head_dim)  for i in 0..head_dim/2
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?;

        // positions: [max_seq]
        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq,), device)?;

        // freqs: outer product [max_seq, half]
        // freqs[p, i] = p * inv_freq[i]
        let pos_col = positions.unsqueeze(1)?;           // [max_seq, 1]
        let inv_row = inv_freq.unsqueeze(0)?;            // [1, half]
        let freqs = pos_col.broadcast_mul(&inv_row)?;    // [max_seq, half]

        // emb = cat([freqs, freqs], dim=-1) → [max_seq, head_dim]
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;

        let cos = emb.cos()?.to_dtype(DType::F32)?;
        let sin = emb.sin()?.to_dtype(DType::F32)?;

        Ok(Self { cos, sin })
    }

    /// Apply RoPE to a query or key tensor.
    ///
    /// `x` has shape [batch, n_heads, seq_len, head_dim].
    /// `seq_len` must be ≤ the max_seq used at construction time.
    pub fn apply(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;

        // cos/sin: [seq_len, head_dim]
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;

        // Expand to [1, 1, seq_len, head_dim] for broadcasting.
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let half = x.dim(candle_core::D::Minus1)? / 2;

        // rotate_half: cat([-x[half:], x[:half]], dim=-1)
        let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
        let x2 = x.narrow(candle_core::D::Minus1, half, half)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;

        let result = (x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;
        Ok(result.to_dtype(dtype)?)
    }
}
