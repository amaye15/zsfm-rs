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
                // emb = cat([freqs, freqs]) → cos[i] == cos[half+i]
                cos[p * head_dim + i] = c;
                cos[p * head_dim + half + i] = c;
                sin[p * head_dim + i] = s;
                sin[p * head_dim + half + i] = s;
            }
        }
        Self { cos, sin, head_dim }
    }

    /// Apply RoPE to x with shape [b, seq, n_heads, head_dim].
    ///
    /// TimesFM rotation: [x1*cos - x2*sin, x2*cos + x1*sin] where x1=x[..,:half], x2=x[..,half:]
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let dims = x.dims();
        let seq = dims[1];
        let hd = dims[3];
        let half = hd / 2;
        let dtype = x.dtype();
        let device = x.device();

        let cos_slice: Vec<f32> = self.cos[start_pos * hd..(start_pos + seq) * hd].to_vec();
        let sin_slice: Vec<f32> = self.sin[start_pos * hd..(start_pos + seq) * hd].to_vec();

        // [seq, hd] → [1, seq, 1, hd] for broadcasting over [b, seq, nh, hd]
        let cos_t = Tensor::from_vec(cos_slice, (seq, hd), device)?
            .unsqueeze(0)?.unsqueeze(2)?;
        let sin_t = Tensor::from_vec(sin_slice, (seq, hd), device)?
            .unsqueeze(0)?.unsqueeze(2)?;

        let x32 = x.to_dtype(DType::F32)?;
        let x1 = x32.narrow(D::Minus1, 0, half)?;
        let x2 = x32.narrow(D::Minus1, half, half)?;
        let cos1 = cos_t.narrow(D::Minus1, 0, half)?;
        let sin1 = sin_t.narrow(D::Minus1, 0, half)?;

        let first = (x1.broadcast_mul(&cos1)? - x2.broadcast_mul(&sin1)?)?;
        let second = (x2.broadcast_mul(&cos1)? + x1.broadcast_mul(&sin1)?)?;

        Ok(Tensor::cat(&[&first, &second], D::Minus1)?.to_dtype(dtype)?)
    }
}
