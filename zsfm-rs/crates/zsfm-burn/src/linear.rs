use burn::tensor::Tensor;
#[cfg(test)]
use burn::tensor::{Device, TensorData};

/// y = x @ w^T + b, matching `zsfm_nn::linear_bias` semantics.
/// `x`: [lead, d_in], `w`: [d_out, d_in], `b`: [d_out].
pub fn burn_linear_bias(x: Tensor<2>, w: Tensor<2>, b: Tensor<1>) -> Tensor<2> {
    let y: Tensor<2> = x.matmul(w.transpose());
    y + b.unsqueeze_dim::<2>(0)
}

/// y = x @ w^T, matching `zsfm_nn::linear_nobias`.
pub fn burn_linear_nobias(x: Tensor<2>, w: Tensor<2>) -> Tensor<2> {
    x.matmul(w.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::flex()
    }

    #[test]
    fn linear_matches_manual() {
        let device = dev();
        let x = Tensor::<2>::from_data(
            TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );
        let w = Tensor::<2>::from_data(
            TensorData::from([[1.0f32, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            &device,
        );
        let b = Tensor::<1>::from_data(TensorData::from([10.0f32, 20.0]), &device);
        let y = burn_linear_bias(x, w, b);
        let data: Vec<f32> = y.to_data().try_to_vec().unwrap();
        assert_eq!(data, vec![14.0, 22.0, 20.0, 25.0]);
    }
}
