use candle_core::{Device, Tensor};
use zsfm_nn::{layer_norm_no_affine, rms_norm, softmax_host};

#[test]
fn shared_softmax_matches_cli_expectations() {
    let p = softmax_host(&[1.0, 2.0, 3.0]);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!(softmax_host(&[]).is_empty());
    // Uniform fallback on degenerate input is finite.
    let q = softmax_host(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
    assert!(q.iter().all(|v| v.is_finite()));
}

#[test]
fn shared_norms_agree_with_manual() {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
    let got = layer_norm_no_affine(&x, 1e-5)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mean = 2.5f32;
    let var =
        ((1.0 - mean).powi(2) + (2.0 - mean).powi(2) + (3.0 - mean).powi(2) + (4.0 - mean).powi(2))
            / 4.0;
    let std = (var + 1e-5).sqrt();
    assert!((got[0] - (1.0 - mean) / std).abs() < 1e-5);

    let w = Tensor::from_vec(vec![1.0f32; 4], (4,), &device).unwrap();
    let r = rms_norm(&x, Some(&w), 1e-6).unwrap();
    assert_eq!(r.dims(), &[1, 4]);
}
