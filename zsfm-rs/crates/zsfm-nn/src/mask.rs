use anyhow::Result;
use candle_core::{Device, Tensor};

/// Additive causal mask of shape `[1, 1, seq, seq]`: `0.0` where `num_masked <= k <= q`,
/// `-inf` elsewhere. `num_masked` lets a prefix of keys be masked out regardless of query
/// position (used for left-padded contexts); pass `0` for a plain causal mask.
pub fn make_causal_mask(seq: usize, num_masked: usize, device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = (0..seq)
        .flat_map(|q| {
            (0..seq).map(move |k| {
                if k <= q && k >= num_masked {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Ok(Tensor::from_vec(data, (seq, seq), device)?
        .unsqueeze(0)?
        .unsqueeze(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_causal_mask_shape_and_values() {
        let device = Device::Cpu;
        let m = make_causal_mask(3, 0, &device).unwrap();
        assert_eq!(m.dims(), &[1, 1, 3, 3]);
        let v: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        let inf = f32::NEG_INFINITY;
        assert_eq!(v, vec![0.0, inf, inf, 0.0, 0.0, inf, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn num_masked_blocks_prefix_keys() {
        let device = Device::Cpu;
        let m = make_causal_mask(3, 1, &device).unwrap();
        let v: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        let inf = f32::NEG_INFINITY;
        // key 0 is masked for every query now.
        assert_eq!(v, vec![inf, inf, inf, inf, 0.0, inf, inf, 0.0, 0.0]);
    }
}
