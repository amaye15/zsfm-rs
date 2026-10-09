use burn::tensor::{activation, Tensor};
#[cfg(test)]
use burn::tensor::{Device, TensorData};

/// RMSNorm over the last dim, matching `zsfm_nn::rms_norm` with weight.
pub fn burn_rms_norm(x: Tensor<2>, w: Tensor<1>, eps: f32) -> Tensor<2> {
    let mean_sq = x.clone().square().mean_dim(1);
    let rms = (mean_sq.add_scalar(eps)).sqrt();
    let normed = x / rms;
    normed * w.unsqueeze_dim::<2>(0)
}

/// LayerNorm over the last dim, matching `zsfm_nn::layer_norm`.
pub fn burn_layer_norm(x: Tensor<2>, w: Tensor<1>, b: Tensor<1>, eps: f32) -> Tensor<2> {
    let mean = x.clone().mean_dim(1);
    let centered = x - mean;
    let var = centered.clone().square().mean_dim(1);
    let std = (var.add_scalar(eps)).sqrt();
    let normed = centered / std;
    normed * w.unsqueeze_dim::<2>(0) + b.unsqueeze_dim::<2>(0)
}

/// Softmax over the last dim for host-checked logits.
pub fn burn_softmax_last_dim(x: Tensor<2>) -> Tensor<2> {
    activation::softmax(x, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::flex()
    }

    #[test]
    fn rms_matches_manual() {
        let device = dev();
        let x = Tensor::<2>::from_data(TensorData::from([[1.0f32, 2.0, 3.0, -4.0]]), &device);
        let w = Tensor::<1>::from_data(TensorData::from([2.0f32, 2.0, 2.0, 2.0]), &device);
        let got: Vec<f32> = burn_rms_norm(x, w, 1e-6).to_data().try_to_vec().unwrap();
        let vals = [1.0f32, 2.0, 3.0, -4.0];
        let mean_sq: f32 = vals.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let rms = (mean_sq + 1e-6).sqrt();
        for (g, v) in got.iter().zip(vals.iter()) {
            assert!((g - v / rms * 2.0).abs() < 1e-5, "{g} vs {v}");
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let device = dev();
        let x = Tensor::<2>::from_data(TensorData::from([[1.0f32, 2.0, 3.0]]), &device);
        let p: Vec<f32> = burn_softmax_last_dim(x).to_data().try_to_vec().unwrap();
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
