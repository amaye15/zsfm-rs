use burn::tensor::{activation, Tensor};

/// Scaled dot-product attention with additive mask, matching
/// `zsfm_nn::scaled_dot_product_attention`.
/// `q/k/v`: [batch, heads, seq_q/kv, head_dim].
pub fn burn_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scale: f32,
    mask: Option<Tensor<4>>,
) -> Tensor<4> {
    let scores = q.matmul(k.permute([0, 1, 3, 2])).mul_scalar(scale);
    let scores = match mask {
        Some(m) => scores + m,
        None => scores,
    };
    let probs = activation::softmax(scores, 3);
    probs.matmul(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn uniform_attention_averages_values() {
        let device = Device::flex();
        let q = Tensor::<4>::zeros([1, 1, 2, 1], &device);
        let k = Tensor::<4>::zeros([1, 1, 2, 1], &device);
        let v = Tensor::<4>::from_data(TensorData::from([[[[2.0f32], [4.0]]]]), &device);
        let out = burn_attention(q, k, v, 1.0, None);
        let vals: Vec<f32> = out.to_data().try_to_vec().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-5);
        assert!((vals[1] - 3.0).abs() < 1e-5);
    }
}
