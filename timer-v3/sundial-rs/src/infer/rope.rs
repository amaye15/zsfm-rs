use anyhow::Result;
use candle_core::{DType, Tensor, D};

pub struct RopeCache {
    cos: Vec<f32>,
    sin: Vec<f32>,
    head_dim: usize,
}

impl RopeCache {
    pub fn new(head_dim: usize, max_seq: usize, theta: f64) -> Self {
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();
        let mut cos = vec![0.0f32; max_seq * head_dim];
        let mut sin = vec![0.0f32; max_seq * head_dim];
        for p in 0..max_seq {
            for i in 0..half {
                let angle = (p as f64 * inv_freq[i]) as f32;
                let (s, c) = angle.sin_cos();
                cos[p * head_dim + i] = c;
                cos[p * head_dim + half + i] = c;
                sin[p * head_dim + i] = s;
                sin[p * head_dim + half + i] = s;
            }
        }
        Self { cos, sin, head_dim }
    }

    /// Apply RoPE to x with shape [b, n_heads, seq, head_dim].
    ///
    /// Llama rotation: x * cos + rotate_half(x) * sin
    /// rotate_half(x) = cat([-x[..., half:], x[..., :half]], dim=-1)
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let dims = x.dims();
        let seq = dims[2];
        let hd = dims[3];
        let half = hd / 2;
        let dtype = x.dtype();
        let device = x.device();

        let cos_slice: Vec<f32> = self.cos[start_pos * hd..(start_pos + seq) * hd].to_vec();
        let sin_slice: Vec<f32> = self.sin[start_pos * hd..(start_pos + seq) * hd].to_vec();

        // [seq, hd] → [1, 1, seq, hd] for broadcasting over [b, nh, seq, hd]
        let cos_t = Tensor::from_vec(cos_slice, (seq, hd), device)?
            .unsqueeze(0)?.unsqueeze(0)?;
        let sin_t = Tensor::from_vec(sin_slice, (seq, hd), device)?
            .unsqueeze(0)?.unsqueeze(0)?;

        let x32 = x.to_dtype(DType::F32)?;
        let x1 = x32.narrow(D::Minus1, 0, half)?;
        let x2 = x32.narrow(D::Minus1, half, half)?;

        // rotate_half: cat([-x2, x1])
        let neg_x2 = x2.neg()?;
        let rotated = Tensor::cat(&[&neg_x2, &x1], D::Minus1)?;

        let cos_half = cos_t.narrow(D::Minus1, 0, half)?;
        let sin_half = sin_t.narrow(D::Minus1, 0, half)?;

        let cos_full = Tensor::cat(&[&cos_half, &cos_half], D::Minus1)?;
        let sin_full = Tensor::cat(&[&sin_half, &sin_half], D::Minus1)?;

        let out = (x32.broadcast_mul(&cos_full)? + rotated.broadcast_mul(&sin_full)?)?;
        Ok(out.to_dtype(dtype)?)
    }
}
