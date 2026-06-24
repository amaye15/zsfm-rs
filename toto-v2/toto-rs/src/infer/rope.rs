// Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
//
// This product includes software developed at Datadog (https://www.datadoghq.com/)
// Copyright 2026 Datadog, Inc.

use anyhow::Result;
use candle_core::{Device, Tensor, D};

/// Precomputed RoPE (with xPos scaling) for a partial head dimension.
///
/// Applied to the first `proj_width` dims of each head.
/// For the 2.5B model: `partial_factor=(0.0, 0.5)` → `proj_width = qk_dim/2 = 32`.
pub struct RopeCache {
    pub proj_width: usize,
    cos: Vec<Vec<f32>>,         // [max_len][proj_width]
    sin: Vec<Vec<f32>>,         // [max_len][proj_width]
    xpos_base_scale: Vec<f32>,  // [proj_width/2]
}

impl RopeCache {
    pub fn new(qk_dim: usize, max_len: usize) -> Self {
        let proj_width = qk_dim / 2; // partial_factor (0.0, 0.5)
        let half = proj_width / 2;
        let base = 10000_f32;

        let theta: Vec<f32> = (0..half)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / proj_width as f32))
            .collect();

        let mut cos = vec![vec![0.0f32; proj_width]; max_len];
        let mut sin = vec![vec![0.0f32; proj_width]; max_len];
        for m in 0..max_len {
            for i in 0..half {
                let angle = m as f32 * theta[i];
                cos[m][2 * i] = angle.cos();
                cos[m][2 * i + 1] = angle.cos();
                sin[m][2 * i] = angle.sin();
                sin[m][2 * i + 1] = angle.sin();
            }
        }

        // xPos base scale: (2i + 0.4*proj_width) / (1.4*proj_width)
        let xpos_base_scale: Vec<f32> = (0..half)
            .map(|i| ((2 * i) as f32 + 0.4 * proj_width as f32) / (1.4 * proj_width as f32))
            .collect();

        Self { proj_width, cos, sin, xpos_base_scale }
    }

    /// Apply xPos-RoPE to query or key.
    ///
    /// `x`: `[batch, heads, seq, qk_dim]` — first `proj_width` dims are rotated.
    /// `seq_ids`: position index per sequence step.
    /// `xpos_exponent`: +1.0 for query, -1.0 for key.
    pub fn apply(
        &self,
        x: &Tensor,
        seq_ids: &[u32],
        xpos_exponent: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let qk_dim = x.dim(D::Minus1)?;
        let proj_width = self.proj_width;
        let half = proj_width / 2;
        let seq_len = seq_ids.len();

        let max_pos = seq_ids.iter().copied().max().unwrap_or(0) as f32;
        let center = ((max_pos as u32 + 1) / 2) as f32;

        let mut cos_data = vec![0.0f32; seq_len * proj_width];
        let mut sin_data = vec![0.0f32; seq_len * proj_width];
        for (si, &pos) in seq_ids.iter().enumerate() {
            let power = (pos as f32 - center) / 256.0; // xpos_scale_base = 256
            for i in 0..half {
                let xpos_s = self.xpos_base_scale[i].powf(power).powf(xpos_exponent);
                let c = self.cos[pos as usize][2 * i] * xpos_s;
                let s = self.sin[pos as usize][2 * i] * xpos_s;
                cos_data[si * proj_width + 2 * i] = c;
                cos_data[si * proj_width + 2 * i + 1] = c;
                sin_data[si * proj_width + 2 * i] = s;
                sin_data[si * proj_width + 2 * i + 1] = s;
            }
        }

        // [1, 1, seq, proj_width] for broadcasting; cast to match x dtype (e.g. F64)
        let pos_cos = Tensor::from_vec(cos_data, (1usize, 1, seq_len, proj_width), device)?
            .to_dtype(x.dtype())?;
        let pos_sin = Tensor::from_vec(sin_data, (1usize, 1, seq_len, proj_width), device)?
            .to_dtype(x.dtype())?;

        // Split rotated vs pass-through dims
        let x_rot = x.narrow(D::Minus1, 0, proj_width)?;
        let rot_x = rotate_half(&x_rot, proj_width)?;
        let rotated = (x_rot.broadcast_mul(&pos_cos)? + rot_x.broadcast_mul(&pos_sin)?)?;

        let result = if qk_dim > proj_width {
            let x_pass = x.narrow(D::Minus1, proj_width, qk_dim - proj_width)?.contiguous()?;
            Tensor::cat(&[&rotated, &x_pass], D::Minus1)?
        } else {
            rotated
        };
        Ok(result.contiguous()?)
    }
}

/// rotate_half: [a0, b0, a1, b1, ...] → [-b0, a0, -b1, a1, ...]
fn rotate_half(x: &Tensor, proj_width: usize) -> Result<Tensor> {
    let half = proj_width / 2;
    let mut shape = x.dims().to_vec();
    let last = shape.len() - 1;
    shape[last] = half;
    shape.push(2);
    let x_pairs = x.reshape(shape)?;
    let x1 = x_pairs.narrow(D::Minus1, 0, 1)?;
    let x2 = x_pairs.narrow(D::Minus1, 1, 1)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    let out_shape = x.dims().to_vec();
    Ok(rotated.reshape(out_shape)?)
}
