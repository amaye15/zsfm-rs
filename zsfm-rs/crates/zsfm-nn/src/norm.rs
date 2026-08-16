use anyhow::Result;
use candle_core::{Tensor, D};

/// RMSNorm over the last dim: `x / sqrt(mean(x^2) + eps) * weight` (weight optional — some
/// models fold the scale into a separate op and call this with `None`). Computes in `x`'s own
/// dtype — cast beforehand if a caller needs a fixed compute precision regardless of input dtype.
pub fn rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let rms = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (rms + eps)?.sqrt()?;
    let x = x.broadcast_div(&rms)?;
    match weight {
        Some(w) => Ok(x.broadcast_mul(w)?),
        None => Ok(x),
    }
}

/// Standard LayerNorm over the last dim: `(x - mean) / sqrt(var + eps) * weight + bias`.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let x = x.broadcast_sub(&mean)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let std = (var + eps)?.sqrt()?;
    let x = x.broadcast_div(&std)?;
    let x = x.broadcast_mul(weight)?;
    Ok(x.broadcast_add(bias)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn rms_norm_matches_manual_formula() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, -4.0], (1, 4), &device).unwrap();
        let w = Tensor::from_vec(vec![2.0f32, 2.0, 2.0, 2.0], (4,), &device).unwrap();
        let eps = 1e-6;

        let got: Vec<f32> = rms_norm(&x, Some(&w), eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let vals = [1.0f32, 2.0, 3.0, -4.0];
        let mean_sq: f32 = vals.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let rms = (mean_sq + eps as f32).sqrt();
        let want: Vec<f32> = vals.iter().map(|v| v / rms * 2.0).collect();

        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
    }

    #[test]
    fn rms_norm_without_weight_is_identity_scaled() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &device).unwrap();
        let got: Vec<f32> = rms_norm(&x, None, 0.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // rms = sqrt((9+16)/2) = sqrt(12.5)
        let rms = 12.5f32.sqrt();
        assert!((got[0] - 3.0 / rms).abs() < 1e-6);
        assert!((got[1] - 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_matches_manual_formula() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let w = Tensor::from_vec(vec![1.0f32; 4], (4,), &device).unwrap();
        let b = Tensor::from_vec(vec![0.0f32; 4], (4,), &device).unwrap();
        let eps = 1e-5;

        let got: Vec<f32> = layer_norm(&x, &w, &b, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let vals = [1.0f32, 2.0, 3.0, 4.0];
        let mean = vals.iter().sum::<f32>() / 4.0;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        let std = (var + eps as f32).sqrt();
        let want: Vec<f32> = vals.iter().map(|v| (v - mean) / std).collect();

        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
    }
}
