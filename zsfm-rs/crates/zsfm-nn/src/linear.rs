use anyhow::{Context, Result};
use candle_core::Tensor;

/// `y = x @ w^T + b`, flattening any leading dims of `x` into a batch dim so this works for
/// rank-2 or higher inputs against a rank-2 weight `[d_out, d_in]`.
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let d_in = *dims.last().context("linear: input has no dims")?;
    let lead: usize = dims[..dims.len() - 1].iter().product();
    let x2 = x.reshape((lead, d_in))?;
    let y2 = x2.matmul(&w.t()?)?;
    let d_out = w.dim(0)?;
    let mut out_dims = dims[..dims.len() - 1].to_vec();
    out_dims.push(d_out);
    let y = y2.reshape(out_dims)?;
    match b {
        Some(b) => Ok(y.broadcast_add(b)?),
        None => Ok(y),
    }
}

pub fn linear_nobias(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    linear(x, w, None)
}

pub fn linear_bias(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    linear(x, w, Some(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn matches_manual_matmul_for_3d_input() {
        let device = Device::Cpu;
        let x = Tensor::from_vec((0..24u32).map(|v| v as f32).collect(), (2, 3, 4), &device).unwrap();
        let w = Tensor::from_vec((0..8u32).map(|v| v as f32 * 0.5).collect(), (2, 4), &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, -1.0], (2,), &device).unwrap();

        let got = linear(&x, &w, Some(&b)).unwrap();
        assert_eq!(got.dims(), &[2, 3, 2]);

        // Manual reference: flatten to 2D, matmul, add bias, reshape.
        let x2 = x.reshape((6, 4)).unwrap();
        let want2 = x2.matmul(&w.t().unwrap()).unwrap().broadcast_add(&b).unwrap();
        let want = want2.reshape((2, 3, 2)).unwrap();

        let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let want_v: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got_v, want_v);
    }

    #[test]
    fn nobias_matches_direct_matmul_for_2d_input() {
        let device = Device::Cpu;
        let x = Tensor::from_vec((0..12u32).map(|v| v as f32).collect(), (3, 4), &device).unwrap();
        let w = Tensor::from_vec((0..8u32).map(|v| v as f32).collect(), (2, 4), &device).unwrap();

        let got = linear_nobias(&x, &w).unwrap();
        let want = x.matmul(&w.t().unwrap()).unwrap();

        let got_v: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let want_v: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(got_v, want_v);
    }
}
