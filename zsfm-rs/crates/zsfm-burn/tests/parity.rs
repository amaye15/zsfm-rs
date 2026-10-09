//! Candle-vs-Burn parity + timing, offline (fixed synthetic inputs).
//!
//! Run with `-- --nocapture` to print the timing table. Every case asserts
//! max-abs-error below 1e-5 and records median ms over 50 iterations.

use burn::tensor::{Device, Tensor, TensorData};
use candle_core::{Device as CDevice, Tensor as CTensor};
use zsfm_burn::bench::{bench_op, max_abs_err};

fn dev() -> Device {
    Device::flex()
}

fn pseudo_data(n: usize, seed: u64) -> Vec<f32> {
    // Deterministic pseudo-random in [-1, 1).
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn linear_parity_and_timing() {
    let (lead, d_in, d_out) = (32, 64, 64);
    let xv = pseudo_data(lead * d_in, 1);
    let wv = pseudo_data(d_out * d_in, 2);
    let bv = pseudo_data(d_out, 3);

    // Candle reference.
    let cdev = CDevice::Cpu;
    let x = CTensor::from_vec(xv.clone(), (lead, d_in), &cdev).unwrap();
    let w = CTensor::from_vec(wv.clone(), (d_out, d_in), &cdev).unwrap();
    let b = CTensor::from_vec(bv.clone(), d_out, &cdev).unwrap();
    let expected: Vec<f32> = zsfm_nn::linear(&x, &w, Some(&b))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    // Burn.
    let device = dev();
    let bx = Tensor::<2>::from_data(TensorData::new(xv.clone(), [lead, d_in]), &device);
    let bw = Tensor::<2>::from_data(TensorData::new(wv.clone(), [d_out, d_in]), &device);
    let bb = Tensor::<1>::from_data(TensorData::new(bv.clone(), [d_out]), &device);
    let got: Vec<f32> = zsfm_burn::linear::burn_linear_bias(bx.clone(), bw.clone(), bb.clone())
        .to_data()
        .try_to_vec()
        .unwrap();

    let err = max_abs_err(&expected, &got);
    let candle_ms = bench_op(50, || {
        let _ = zsfm_nn::linear(&x, &w, Some(&b)).unwrap();
    });
    let burn_ms = bench_op(50, || {
        let _ = zsfm_burn::linear::burn_linear_bias(bx.clone(), bw.clone(), bb.clone());
    });
    println!(
        "linear [{lead}x{d_in}->{d_out}]: candle {candle_ms:.3}ms burn {burn_ms:.3}ms err {err:.2e}"
    );
    assert!(err < 1e-5, "linear parity failed: {err}");
}

#[test]
fn rms_norm_parity_and_timing() {
    let (n, d) = (32, 64);
    let xv = pseudo_data(n * d, 11);
    let wv = pseudo_data(d, 12);

    let cdev = CDevice::Cpu;
    let x = CTensor::from_vec(xv.clone(), (n, d), &cdev).unwrap();
    let w = CTensor::from_vec(wv.clone(), d, &cdev).unwrap();
    let expected: Vec<f32> = zsfm_nn::rms_norm(&x, Some(&w), 1e-6)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let device = dev();
    let bx = Tensor::<2>::from_data(TensorData::new(xv.clone(), [n, d]), &device);
    let bw = Tensor::<1>::from_data(TensorData::new(wv.clone(), [d]), &device);
    let got: Vec<f32> = zsfm_burn::norm::burn_rms_norm(bx.clone(), bw.clone(), 1e-6)
        .to_data()
        .try_to_vec()
        .unwrap();

    let err = max_abs_err(&expected, &got);
    let candle_ms = bench_op(50, || {
        let _ = zsfm_nn::rms_norm(&x, Some(&w), 1e-6).unwrap();
    });
    let burn_ms = bench_op(50, || {
        let _ = zsfm_burn::norm::burn_rms_norm(bx.clone(), bw.clone(), 1e-6);
    });
    println!("rms_norm [{n}x{d}]: candle {candle_ms:.3}ms burn {burn_ms:.3}ms err {err:.2e}");
    assert!(err < 1e-4, "rms_norm parity failed: {err}");
}

#[test]
fn attention_parity_and_timing() {
    let (b, h, sq, sk, hd) = (2, 4, 16, 16, 32);
    let qv = pseudo_data(b * h * sq * hd, 21);
    let kv = pseudo_data(b * h * sk * hd, 22);
    let vv = pseudo_data(b * h * sk * hd, 23);

    let cdev = CDevice::Cpu;
    let q = CTensor::from_vec(qv.clone(), (b, h, sq, hd), &cdev).unwrap();
    let k = CTensor::from_vec(kv.clone(), (b, h, sk, hd), &cdev).unwrap();
    let v = CTensor::from_vec(vv.clone(), (b, h, sk, hd), &cdev).unwrap();
    let scale = 1.0 / (hd as f64).sqrt();
    let expected: Vec<f32> = zsfm_nn::scaled_dot_product_attention(&q, &k, &v, scale, None)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let device = dev();
    let bq = Tensor::<4>::from_data(TensorData::new(qv.clone(), [b, h, sq, hd]), &device);
    let bk = Tensor::<4>::from_data(TensorData::new(kv.clone(), [b, h, sk, hd]), &device);
    let bv = Tensor::<4>::from_data(TensorData::new(vv.clone(), [b, h, sk, hd]), &device);
    let got: Vec<f32> = zsfm_burn::attn::burn_attention(
        bq.clone(),
        bk.clone(),
        bv.clone(),
        1.0 / (hd as f32).sqrt(),
        None,
    )
    .to_data()
    .try_to_vec()
    .unwrap();

    let err = max_abs_err(&expected, &got);
    let candle_ms = bench_op(20, || {
        let _ = zsfm_nn::scaled_dot_product_attention(&q, &k, &v, scale, None).unwrap();
    });
    let burn_ms = bench_op(20, || {
        let _ = zsfm_burn::attn::burn_attention(
            bq.clone(),
            bk.clone(),
            bv.clone(),
            1.0 / (hd as f32).sqrt(),
            None,
        );
    });
    println!(
        "attn [b{b} h{h} sq{sq} hd{hd}]: candle {candle_ms:.3}ms burn {burn_ms:.3}ms err {err:.2e}"
    );
    assert!(err < 1e-4, "attention parity failed: {err}");
}
