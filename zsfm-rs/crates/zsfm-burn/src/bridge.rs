//! Candle-to-Burn weight bridge.
//!
//! During migration, Burn models load GGUF weights through the existing
//! candle GGUF reader (single parser) and convert each F32 tensor to Burn
//! `TensorData` with identical values and shape. Once all models port,
//! loading can move to `burn-store` directly.

use burn::tensor::{Device, Tensor, TensorData};

/// Burn `Tensor<2>` from a 2D candle F32 tensor (same values, same shape).
pub fn burn_weight_2d(t: &candle_core::Tensor, device: &Device) -> anyhow::Result<Tensor<2>> {
    let dims = t.dims().to_vec();
    anyhow::ensure!(dims.len() == 2, "expected 2D weight, got {dims:?}");
    let data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(Tensor::<2>::from_data(
        TensorData::new(data, [dims[0], dims[1]]),
        device,
    ))
}

/// Burn `Tensor<1>` from a 1D candle F32 tensor.
pub fn burn_weight_1d(t: &candle_core::Tensor, device: &Device) -> anyhow::Result<Tensor<1>> {
    let data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(Tensor::<1>::from_data(
        TensorData::new(data, [t.elem_count()]),
        device,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device as CDevice, Tensor as CTensor};

    #[test]
    fn roundtrip_values() {
        let t = CTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &CDevice::Cpu).unwrap();
        let dev = Device::flex();
        let b = burn_weight_2d(&t, &dev).unwrap();
        let back: Vec<f32> = b.to_data().try_to_vec().unwrap();
        assert_eq!(back, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
