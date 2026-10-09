//! TimesFM 2.5 inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle): same RevIN running stats, tokenizer,
//! prefill + KV-cached decode, RoPE, and denormalization. Weight loading goes
//! through the existing candle GGUF reader and converts each F32 tensor to
//! Burn `TensorData`, so there is exactly one GGUF parser. Architecture
//! constants mirror the fixed values in `super`.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};

const D_MODEL: usize = 1280;
const N_HEADS: usize = 16;
const HEAD_DIM: usize = 80;
const N_LAYERS: usize = 20;
const INPUT_PATCH: usize = 32;
const OUTPUT_PATCH: usize = 128;
const N_OUTPUTS: usize = 10;
const DECODE_IDX: usize = 5;
const M_PATCHES: usize = OUTPUT_PATCH / INPUT_PATCH;
const ROPE_THETA: f64 = 10000.0;
const MAX_SEQ: usize = 16384 / INPUT_PATCH + 256;
const NORM_EPS: f32 = 1e-6;
const REVIN_TOL: f32 = 1e-6;

fn device() -> Device {
    Device::flex()
}

fn burn2(t: &candle_core::Tensor) -> Result<Tensor<2>> {
    zsfm_burn::burn_weight_2d(t, &device())
}

fn burn1(t: &candle_core::Tensor) -> Result<Tensor<1>> {
    zsfm_burn::burn_weight_1d(t, &device())
}

fn load_t(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
) -> Result<Tensor<2>> {
    let t = zsfm_nn::load_tensor(content, reader, name, &CDevice::Cpu, DType::F32)?;
    burn2(&t)
}

fn load_v(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
) -> Result<Tensor<1>> {
    let t = zsfm_nn::load_tensor(content, reader, name, &CDevice::Cpu, DType::F32)?;
    burn1(&t)
}

fn load_w(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
    expected_d_out: usize,
) -> Result<Tensor<2>> {
    let t = zsfm_nn::load_weight(content, reader, name, expected_d_out, &CDevice::Cpu)?;
    burn2(&t)
}

struct BurnResidualBlockW {
    hidden_w: Tensor<2>,
    hidden_b: Option<Tensor<1>>,
    output_w: Tensor<2>,
    output_b: Option<Tensor<1>>,
    skip_w: Tensor<2>,
    skip_b: Option<Tensor<1>>,
}

struct BurnAttnW {
    qkv_w: Tensor<2>,
    out_w: Tensor<2>,
    q_norm_w: Tensor<1>,
    k_norm_w: Tensor<1>,
    q_scale: Tensor<1>,
    pre_norm_w: Tensor<1>,
    post_norm_w: Tensor<1>,
}

struct BurnFfnW {
    up_w: Tensor<2>,
    down_w: Tensor<2>,
    pre_norm_w: Tensor<1>,
    post_norm_w: Tensor<1>,
}

struct BurnBlockW {
    attn: BurnAttnW,
    ffn: BurnFfnW,
}

pub struct BurnTimesFMModel {
    tokenizer: BurnResidualBlockW,
    blocks: Vec<BurnBlockW>,
    out_point: BurnResidualBlockW,
    rope: BurnRope,
}

struct BurnRope {
    cos: Tensor<2>,
    sin: Tensor<2>,
    head_dim: usize,
}

impl BurnRope {
    fn build(head_dim: usize, max_seq: usize, theta: f64) -> Self {
        let dev = device();
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();
        let mut cos = vec![0.0f32; max_seq * head_dim];
        let mut sin = vec![0.0f32; max_seq * head_dim];
        for p in 0..max_seq {
            for i in 0..half {
                let angle = (p as f64 * inv_freq[i]) as f32;
                let (s, c) = angle.sin_cos();
                cos[p * head_dim + i] = c;
                cos[p * head_dim + half + i] = c;
                sin[p * head_dim + i] = s;
                sin[p * head_dim + half + i] = s;
            }
        }
        Self {
            cos: Tensor::<2>::from_data(TensorData::new(cos, [max_seq, head_dim]), &dev),
            sin: Tensor::<2>::from_data(TensorData::new(sin, [max_seq, head_dim]), &dev),
            head_dim,
        }
    }

    /// Apply RoPE to x with shape [b, seq, n_heads, head_dim].
    fn apply(&self, x: Tensor<4>, start_pos: usize) -> Tensor<4> {
        let [b, seq, _h, hd] = x.dims();
        let half = hd / 2;
        let cos = self
            .cos
            .clone()
            .narrow(0, start_pos, seq)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2);
        let sin = self
            .sin
            .clone()
            .narrow(0, start_pos, seq)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2);
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.clone().narrow(3, half, half);
        let cos1 = cos.clone().narrow(3, 0, half);
        let sin1 = sin.clone().narrow(3, 0, half);
        let _ = (b, self.head_dim);
        let first = x1.clone() * cos1.clone() - x2.clone() * sin1.clone();
        let second = x2 * cos1 + x1 * sin1;
        Tensor::cat(vec![first, second], 3)
    }
}

fn linear(x: Tensor<2>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<2> {
    match b {
        Some(bias) => zsfm_burn::linear::burn_linear_bias(x, w, bias),
        None => zsfm_burn::linear::burn_linear_nobias(x, w),
    }
}

fn linear_nd<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    let dims = x.dims();
    let last = dims[D - 1];
    let lead: usize = dims[..D - 1].iter().product();
    let out_d = w.dims()[0];
    let flat = x.reshape([lead, last]);
    let y = linear(flat, w, b);
    let mut out_dims = dims;
    out_dims[D - 1] = out_d;
    y.reshape(out_dims)
}

fn rms_norm_2d(x: Tensor<2>, w: Tensor<1>) -> Tensor<2> {
    zsfm_burn::norm::burn_rms_norm(x, w, NORM_EPS)
}

fn rms_norm_nd<const D: usize>(x: Tensor<D>, w: Tensor<1>) -> Tensor<D> {
    let dims = x.dims();
    let last = dims[D - 1];
    let lead: usize = dims[..D - 1].iter().product();
    let y = zsfm_burn::norm::burn_rms_norm(x.reshape([lead, last]), w, NORM_EPS);
    y.reshape(dims)
}

fn softmax_last<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::softmax(x, D - 1)
}

fn silu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::silu(x)
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

fn compute_q_scale(raw: Vec<f32>) -> Vec<f32> {
    let factor = 1.442695041f32 / (HEAD_DIM as f32).sqrt();
    raw.into_iter().map(|x| factor * softplus(x)).collect()
}

fn load_residual_block(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
    hidden_d_out: usize,
    out_d_out: usize,
    with_bias: bool,
) -> Result<BurnResidualBlockW> {
    let hidden_w = load_w(
        content,
        reader,
        &format!("{prefix}.hidden.weight"),
        hidden_d_out,
    )?;
    let hidden_b = with_bias
        .then(|| load_v(content, reader, &format!("{prefix}.hidden.bias")))
        .transpose()?;
    let output_w = load_w(
        content,
        reader,
        &format!("{prefix}.output.weight"),
        out_d_out,
    )?;
    let output_b = with_bias
        .then(|| load_v(content, reader, &format!("{prefix}.output.bias")))
        .transpose()?;
    let skip_w = load_w(content, reader, &format!("{prefix}.skip.weight"), out_d_out)?;
    let skip_b = with_bias
        .then(|| load_v(content, reader, &format!("{prefix}.skip.bias")))
        .transpose()?;
    Ok(BurnResidualBlockW {
        hidden_w,
        hidden_b,
        output_w,
        output_b,
        skip_w,
        skip_b,
    })
}

fn forward_residual_block(x: Tensor<2>, w: &BurnResidualBlockW) -> Tensor<2> {
    let h = silu(linear(x.clone(), w.hidden_w.clone(), w.hidden_b.clone()));
    let out = linear(h, w.output_w.clone(), w.output_b.clone());
    let skip = linear(x, w.skip_w.clone(), w.skip_b.clone());
    out + skip
}

fn burn_causal_mask(n: usize) -> Tensor<4> {
    let dev = device();
    let data: Vec<f32> = (0..n)
        .flat_map(|q| (0..n).map(move |k| if k <= q { 0.0f32 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::<2>::from_data(TensorData::new(data, [n, n]), &dev)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0)
}

fn burn_decode_mask(new_len: usize, cache_len: usize) -> Tensor<4> {
    let dev = device();
    let total = cache_len + new_len;
    let data: Vec<f32> = (0..new_len)
        .flat_map(|q_rel| {
            (0..total).map(move |k_abs| {
                if k_abs <= cache_len + q_rel {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::<2>::from_data(TensorData::new(data, [new_len, total]), &dev)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0)
}

impl BurnTimesFMModel {
    pub fn load(gguf_path: &Path) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let tokenizer =
            load_residual_block(&content, &mut reader, "tokenizer", D_MODEL, D_MODEL, true)?;

        let mut blocks = Vec::with_capacity(N_LAYERS);
        for n in 0..N_LAYERS {
            let b = format!("blk.{n}");
            let qkv_w = load_w(
                &content,
                &mut reader,
                &format!("{b}.attn_qkv.weight"),
                3 * D_MODEL,
            )?;
            let out_w = load_w(
                &content,
                &mut reader,
                &format!("{b}.attn_out.weight"),
                D_MODEL,
            )?;
            let q_norm_w = load_v(&content, &mut reader, &format!("{b}.attn_q_norm.weight"))?;
            let k_norm_w = load_v(&content, &mut reader, &format!("{b}.attn_k_norm.weight"))?;
            let q_scale_raw: Vec<f32> = zsfm_nn::load_vec(
                &content,
                &mut reader,
                &format!("{b}.attn_q_scale.weight"),
                &CDevice::Cpu,
            )?;
            let q_scale = Tensor::<1>::from_data(
                TensorData::new(compute_q_scale(q_scale_raw), [HEAD_DIM]),
                &device(),
            );
            let pre_norm_w = load_v(&content, &mut reader, &format!("{b}.pre_attn_norm.weight"))?;
            let post_norm_w = load_v(&content, &mut reader, &format!("{b}.post_attn_norm.weight"))?;
            let up_w = load_w(
                &content,
                &mut reader,
                &format!("{b}.ffn_up.weight"),
                D_MODEL,
            )?;
            let down_w = load_w(
                &content,
                &mut reader,
                &format!("{b}.ffn_down.weight"),
                D_MODEL,
            )?;
            let pre_ff = load_v(&content, &mut reader, &format!("{b}.pre_ff_norm.weight"))?;
            let post_ff = load_v(&content, &mut reader, &format!("{b}.post_ff_norm.weight"))?;
            blocks.push(BurnBlockW {
                attn: BurnAttnW {
                    qkv_w,
                    out_w,
                    q_norm_w,
                    k_norm_w,
                    q_scale,
                    pre_norm_w,
                    post_norm_w,
                },
                ffn: BurnFfnW {
                    up_w,
                    down_w,
                    pre_norm_w: pre_ff,
                    post_norm_w: post_ff,
                },
            });
        }

        let out_point =
            load_residual_block(&content, &mut reader, "out_point", D_MODEL, D_MODEL, false)?;
        let rope = BurnRope::build(HEAD_DIM, MAX_SEQ, ROPE_THETA);
        Ok(Self {
            tokenizer,
            blocks,
            out_point,
            rope,
        })
    }

    pub fn forecast(&self, context: &[f32], prediction_length: usize) -> Result<Vec<Vec<f32>>> {
        let p = INPUT_PATCH;
        let o = OUTPUT_PATCH;
        let q = N_OUTPUTS;
        let m = M_PATCHES;

        let len_front = if context.len() % p == 0 {
            0
        } else {
            p - context.len() % p
        };
        let mut vals = vec![0.0f32; len_front + context.len()];
        vals[len_front..].copy_from_slice(context);
        let mut mask = vec![true; len_front];
        mask.extend(vec![false; context.len()]);
        let n_ctx = (len_front + context.len()) / p;

        let mut patch_mus = Vec::with_capacity(n_ctx);
        let mut patch_sigmas = Vec::with_capacity(n_ctx);
        let (mut rs_n, mut rs_mu, mut rs_sigma) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n_ctx {
            let pv = &vals[i * p..(i + 1) * p];
            let pm = &mask[i * p..(i + 1) * p];
            (rs_n, rs_mu, rs_sigma) = update_running_stats(rs_n, rs_mu, rs_sigma, pv, pm);
            patch_mus.push(rs_mu);
            patch_sigmas.push(rs_sigma);
        }
        let (last_n, last_mu, last_sigma) = (rs_n, rs_mu, rs_sigma);
        let ctx_input = build_tokenizer_input(&vals, &mask, n_ctx, p, &patch_mus, &patch_sigmas);

        let num_decode_steps = if prediction_length <= o {
            0
        } else {
            (prediction_length - 1) / o
        };
        let total_steps = 1 + num_decode_steps;
        let mut all_outputs: Vec<Vec<[f32; N_OUTPUTS]>> = Vec::with_capacity(total_steps);

        let (ctx_out, mut kv_cache) = self.prefill(ctx_input, n_ctx);
        let ctx_flat: Vec<f32> = ctx_out
            .to_data()
            .try_to_vec()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let ctx_denorm = denorm_flat(&ctx_flat, &patch_mus, &patch_sigmas, o, q);
        all_outputs.push(ctx_denorm[n_ctx - 1].clone());

        let (mut ar_n, mut ar_mu, mut ar_sigma) = (last_n, last_mu, last_sigma);
        let mut last_ar: Vec<f32> = ctx_denorm[n_ctx - 1]
            .iter()
            .map(|row| row[DECODE_IDX])
            .collect();

        for step in 0..num_decode_steps {
            let new_vals_flat = last_ar.clone();
            let new_mask_flat = vec![false; o];
            let mut new_mus = Vec::with_capacity(m);
            let mut new_sigmas = Vec::with_capacity(m);
            for mi in 0..m {
                let pv = &new_vals_flat[mi * p..(mi + 1) * p];
                let pm = &new_mask_flat[mi * p..(mi + 1) * p];
                (ar_n, ar_mu, ar_sigma) = update_running_stats(ar_n, ar_mu, ar_sigma, pv, pm);
                new_mus.push(ar_mu);
                new_sigmas.push(ar_sigma);
            }
            let new_input =
                build_tokenizer_input(&new_vals_flat, &new_mask_flat, m, p, &new_mus, &new_sigmas);
            let rope_offset = n_ctx + m * step;
            let new_out = self.decode_chunk(new_input, &mut kv_cache, rope_offset);
            let new_flat: Vec<f32> = new_out
                .to_data()
                .try_to_vec()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let last_m_denorm = denorm_flat(&new_flat, &new_mus, &new_sigmas, o, q);
            let step_out = &last_m_denorm[m - 1];
            last_ar = step_out.iter().map(|row| row[DECODE_IDX]).collect();
            all_outputs.push(step_out.clone());
        }

        let total_available = all_outputs.len() * o;
        let n_take = prediction_length.min(total_available);
        let mut result: Vec<Vec<f32>> = vec![Vec::with_capacity(n_take); q];
        'outer: for step in &all_outputs {
            for timestep in step {
                for qi in 0..q {
                    if result[qi].len() >= prediction_length {
                        break 'outer;
                    }
                    result[qi].push(timestep[qi]);
                }
            }
        }
        Ok(result)
    }

    fn prefill(
        &self,
        tokenizer_input: Tensor<2>,
        n_patches: usize,
    ) -> (Tensor<2>, Vec<(Tensor<4>, Tensor<4>)>) {
        let x = forward_residual_block(tokenizer_input, &self.tokenizer);
        let mut hidden = x.unsqueeze_dim::<3>(0);
        let causal = burn_causal_mask(n_patches);
        let mut kv_cache = Vec::with_capacity(N_LAYERS);
        for block in &self.blocks {
            let (h_out, k, v) = self.prefill_block(hidden, block, n_patches, causal.clone());
            hidden = h_out;
            kv_cache.push((k, v));
        }
        let out_seq: Tensor<2> = hidden.squeeze_dim(0);
        let out = forward_residual_block(out_seq, &self.out_point);
        (out, kv_cache)
    }

    fn prefill_block(
        &self,
        x: Tensor<3>,
        w: &BurnBlockW,
        n_patches: usize,
        causal: Tensor<4>,
    ) -> (Tensor<3>, Tensor<4>, Tensor<4>) {
        let normed = rms_norm_nd(x.clone(), w.attn.pre_norm_w.clone());
        let (attn_out, k, v) = self.prefill_attn(normed, &w.attn, n_patches, causal);
        let attn_out = rms_norm_nd(attn_out, w.attn.post_norm_w.clone()) + x;
        let normed_ff = rms_norm_nd(attn_out.clone(), w.ffn.pre_norm_w.clone());
        let ff_out = self.forward_ffn(normed_ff, &w.ffn);
        let out = rms_norm_nd(ff_out, w.ffn.post_norm_w.clone()) + attn_out;
        (out, k, v)
    }

    fn prefill_attn(
        &self,
        x: Tensor<3>,
        w: &BurnAttnW,
        n_patches: usize,
        causal: Tensor<4>,
    ) -> (Tensor<3>, Tensor<4>, Tensor<4>) {
        burn_prefill_attn(x, w, &self.rope, n_patches, causal)
    }

    fn decode_chunk(
        &self,
        new_input: Tensor<2>,
        kv_cache: &mut Vec<(Tensor<4>, Tensor<4>)>,
        rope_offset: usize,
    ) -> Tensor<2> {
        let x = forward_residual_block(new_input, &self.tokenizer);
        let mut hidden = x.unsqueeze_dim::<3>(0);
        let cached_len = rope_offset;
        let decode_mask = burn_decode_mask(M_PATCHES, cached_len);
        for (li, block) in self.blocks.iter().enumerate() {
            let (h_out, k_new, v_new) = self.decode_block_kv(
                hidden,
                block,
                &kv_cache[li],
                rope_offset,
                decode_mask.clone(),
            );
            hidden = h_out;
            kv_cache[li] = (k_new, v_new);
        }
        let out_seq: Tensor<2> = hidden.squeeze_dim(0);
        forward_residual_block(out_seq, &self.out_point)
    }

    fn decode_block_kv(
        &self,
        x: Tensor<3>,
        w: &BurnBlockW,
        cache: &(Tensor<4>, Tensor<4>),
        rope_offset: usize,
        decode_mask: Tensor<4>,
    ) -> (Tensor<3>, Tensor<4>, Tensor<4>) {
        let normed = rms_norm_nd(x.clone(), w.attn.pre_norm_w.clone());
        let (attn_out, k_new, v_new) =
            self.decode_attn_kv(normed, &w.attn, cache, rope_offset, decode_mask);
        let attn_out = rms_norm_nd(attn_out, w.attn.post_norm_w.clone()) + x;
        let normed_ff = rms_norm_nd(attn_out.clone(), w.ffn.pre_norm_w.clone());
        let ff_out = self.forward_ffn(normed_ff, &w.ffn);
        let out = rms_norm_nd(ff_out, w.ffn.post_norm_w.clone()) + attn_out;
        (out, k_new, v_new)
    }

    fn decode_attn_kv(
        &self,
        x: Tensor<3>,
        w: &BurnAttnW,
        cache: &(Tensor<4>, Tensor<4>),
        rope_offset: usize,
        mask: Tensor<4>,
    ) -> (Tensor<3>, Tensor<4>, Tensor<4>) {
        burn_decode_attn(x, w, &self.rope, cache, rope_offset, mask)
    }

    fn forward_ffn(&self, x: Tensor<3>, w: &BurnFfnW) -> Tensor<3> {
        let h = linear_nd(x, w.up_w.clone(), None);
        let h = silu(h);
        linear_nd(h, w.down_w.clone(), None)
    }
}

fn burn_prefill_attn(
    x: Tensor<3>,
    w: &BurnAttnW,
    rope: &BurnRope,
    n_patches: usize,
    causal: Tensor<4>,
) -> (Tensor<3>, Tensor<4>, Tensor<4>) {
    let qkv = linear_nd(x, w.qkv_w.clone(), None);
    let q: Tensor<3> = qkv.clone().narrow(2, 0, D_MODEL);
    let k: Tensor<3> = qkv.clone().narrow(2, D_MODEL, D_MODEL);
    let v: Tensor<3> = qkv.narrow(2, 2 * D_MODEL, D_MODEL);
    let q: Tensor<4> = q.reshape([1, n_patches, N_HEADS, HEAD_DIM]);
    let k: Tensor<4> = k.reshape([1, n_patches, N_HEADS, HEAD_DIM]);
    let v: Tensor<4> = v.reshape([1, n_patches, N_HEADS, HEAD_DIM]);
    let q = rope.apply(q, 0);
    let k = rope.apply(k, 0);
    let q = rms_norm_nd(q, w.q_norm_w.clone());
    let k = rms_norm_nd(k, w.k_norm_w.clone());
    let scale = w
        .q_scale
        .clone()
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0);
    let q = q * scale;
    let q: Tensor<4> = q.permute([0, 2, 1, 3]);
    let k: Tensor<4> = k.permute([0, 2, 1, 3]);
    let v: Tensor<4> = v.permute([0, 2, 1, 3]);
    let scores = q.clone().matmul(k.clone().transpose()) + causal;
    let attn_w = softmax_last(scores);
    let ctx = attn_w.matmul(v.clone());
    let ctx: Tensor<4> = ctx.permute([0, 2, 1, 3]);
    let ctx: Tensor<3> = ctx.reshape([1, n_patches, D_MODEL]);
    (linear_nd(ctx, w.out_w.clone(), None), k, v)
}

fn burn_decode_attn(
    x: Tensor<3>,
    w: &BurnAttnW,
    rope: &BurnRope,
    cache: &(Tensor<4>, Tensor<4>),
    rope_offset: usize,
    mask: Tensor<4>,
) -> (Tensor<3>, Tensor<4>, Tensor<4>) {
    let qkv = linear_nd(x, w.qkv_w.clone(), None);
    let q: Tensor<3> = qkv.clone().narrow(2, 0, D_MODEL);
    let k: Tensor<3> = qkv.clone().narrow(2, D_MODEL, D_MODEL);
    let v: Tensor<3> = qkv.narrow(2, 2 * D_MODEL, D_MODEL);
    let q: Tensor<4> = q.reshape([1, M_PATCHES, N_HEADS, HEAD_DIM]);
    let k: Tensor<4> = k.reshape([1, M_PATCHES, N_HEADS, HEAD_DIM]);
    let v: Tensor<4> = v.reshape([1, M_PATCHES, N_HEADS, HEAD_DIM]);
    let q = rope.apply(q, rope_offset);
    let k = rope.apply(k, rope_offset);
    let q = rms_norm_nd(q, w.q_norm_w.clone());
    let k = rms_norm_nd(k, w.k_norm_w.clone());
    let scale = w
        .q_scale
        .clone()
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0);
    let q = q * scale;
    let q: Tensor<4> = q.permute([0, 2, 1, 3]);
    let k: Tensor<4> = k.permute([0, 2, 1, 3]);
    let v: Tensor<4> = v.permute([0, 2, 1, 3]);
    let k_full = Tensor::cat(vec![cache.0.clone(), k], 2);
    let v_full = Tensor::cat(vec![cache.1.clone(), v], 2);
    let scores = q.matmul(k_full.clone().transpose()) + mask;
    let attn_w = softmax_last(scores);
    let ctx = attn_w.matmul(v_full.clone());
    let ctx: Tensor<4> = ctx.permute([0, 2, 1, 3]);
    let ctx: Tensor<3> = ctx.reshape([1, M_PATCHES, D_MODEL]);
    (linear_nd(ctx, w.out_w.clone(), None), k_full, v_full)
}

fn update_running_stats(
    n: f32,
    mu: f32,
    sigma: f32,
    vals: &[f32],
    mask: &[bool],
) -> (f32, f32, f32) {
    let mut inc_n = 0.0f32;
    let mut sum_x = 0.0f32;
    for (i, &m) in mask.iter().enumerate() {
        if !m {
            inc_n += 1.0;
            sum_x += vals[i];
        }
    }
    let inc_mu = if inc_n == 0.0 { 0.0 } else { sum_x / inc_n };
    let inc_var = if inc_n == 0.0 {
        0.0
    } else {
        mask.iter()
            .enumerate()
            .filter(|(_, &m)| !m)
            .map(|(i, _)| (vals[i] - inc_mu).powi(2))
            .sum::<f32>()
            / inc_n
    };
    let inc_sigma = inc_var.sqrt();
    let new_n = n + inc_n;
    let safe_new_n = if new_n == 0.0 { 1.0 } else { new_n };
    let new_mu = if new_n == 0.0 {
        0.0
    } else {
        (n * mu + inc_mu * inc_n) / safe_new_n
    };
    let t1 = n * sigma.powi(2);
    let t2 = inc_n * inc_sigma.powi(2);
    let t3 = n * (mu - new_mu).powi(2);
    let t4 = inc_n * (inc_mu - new_mu).powi(2);
    let new_var = if new_n == 0.0 {
        0.0
    } else {
        (t1 + t2 + t3 + t4) / safe_new_n
    };
    (new_n, new_mu, new_var.max(0.0).sqrt())
}

fn build_tokenizer_input(
    vals: &[f32],
    mask: &[bool],
    n_patches: usize,
    p: usize,
    mus: &[f32],
    sigmas: &[f32],
) -> Tensor<2> {
    let mut data = vec![0.0f32; n_patches * 2 * p];
    for pi in 0..n_patches {
        let mu = mus[pi];
        let sigma = sigmas[pi];
        let sigma_safe = if sigma < REVIN_TOL { 1.0f32 } else { sigma };
        for i in 0..p {
            let idx = pi * p + i;
            let is_masked = mask[idx];
            let normed = if is_masked {
                0.0
            } else {
                (vals[idx] - mu) / sigma_safe
            };
            let base = pi * 2 * p;
            data[base + i] = normed;
            data[base + p + i] = if is_masked { 1.0 } else { 0.0 };
        }
    }
    Tensor::<2>::from_data(TensorData::new(data, [n_patches, 2 * p]), &device())
}

fn denorm_flat(
    flat: &[f32],
    mus: &[f32],
    sigmas: &[f32],
    o: usize,
    q: usize,
) -> Vec<Vec<[f32; N_OUTPUTS]>> {
    let n = mus.len();
    let mut result: Vec<Vec<[f32; N_OUTPUTS]>> = vec![vec![[0.0f32; N_OUTPUTS]; o]; n];
    for pi in 0..n {
        let mu = mus[pi];
        let sigma = sigmas[pi];
        for ti in 0..o {
            for qi in 0..q {
                let raw = flat[pi * (o * q) + ti * q + qi];
                result[pi][ti][qi] = raw * sigma + mu;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device as CDevice, Tensor as CTensor, D as CD};

    pub(crate) fn pseudo(n: usize, seed: u64) -> Vec<f32> {
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

    pub(crate) fn max_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn rope_matches_candle() {
        let dev = CDevice::Cpu;
        let candle_rope =
            super::super::rope::RopeCache::new(HEAD_DIM, 16, ROPE_THETA, &dev).unwrap();
        let burn_rope = BurnRope::build(HEAD_DIM, 16, ROPE_THETA);
        let xv = pseudo(1 * 4 * 2 * HEAD_DIM, 7);
        let cx = CTensor::from_vec(xv.clone(), (1, 4, 2, HEAD_DIM), &dev).unwrap();
        let a: Vec<f32> = candle_rope
            .apply(&cx, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let bx = Tensor::<4>::from_data(TensorData::new(xv, [1, 4, 2, HEAD_DIM]), &device());
        let b: Vec<f32> = burn_rope.apply(bx, 2).to_data().try_to_vec().unwrap();
        let err = max_err(&a, &b);
        assert!(err < 1e-5, "rope parity failed: {err:.2e}");
    }

    #[test]
    fn residual_block_matches_candle() {
        // Small dims: in=8, hidden=16, out=16, with bias.
        let (din, h, dout, n) = (8usize, 16usize, 16usize, 4usize);
        let w = |r: usize, c: usize, s: u64| pseudo(r * c, s);
        let hw = w(h, din, 21);
        let hb = pseudo(h, 22);
        let ow = w(dout, h, 23);
        let ob = pseudo(dout, 24);
        let sw = w(dout, din, 25);
        let sb = pseudo(dout, 26);
        let xv = pseudo(n * din, 27);

        let dev = CDevice::Cpu;
        let cx = CTensor::from_vec(xv.clone(), (n, din), &dev).unwrap();
        let lin = |x: &CTensor, wv: &[f32], r: usize, c: usize, bv: Option<&[f32]>| {
            use zsfm_nn::linear;
            let wt = CTensor::from_vec(wv.to_vec(), (r, c), &dev).unwrap();
            let bt = bv.map(|v| CTensor::from_vec(v.to_vec(), r, &dev).unwrap());
            linear(x, &wt, bt.as_ref()).unwrap()
        };
        let hh = candle_nn::ops::silu(&lin(&cx, &hw, h, din, Some(&hb))).unwrap();
        let out = lin(&hh, &ow, dout, h, Some(&ob));
        let skip = lin(&cx, &sw, dout, din, Some(&sb));
        let expected: Vec<f32> = (&out + &skip)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let bdev = device();
        let bxb = Tensor::<2>::from_data(TensorData::new(xv, [n, din]), &bdev);
        let bw = BurnResidualBlockW {
            hidden_w: Tensor::<2>::from_data(TensorData::new(hw, [h, din]), &bdev),
            hidden_b: Some(Tensor::<1>::from_data(TensorData::new(hb, [h]), &bdev)),
            output_w: Tensor::<2>::from_data(TensorData::new(ow, [dout, h]), &bdev),
            output_b: Some(Tensor::<1>::from_data(TensorData::new(ob, [dout]), &bdev)),
            skip_w: Tensor::<2>::from_data(TensorData::new(sw, [dout, din]), &bdev),
            skip_b: Some(Tensor::<1>::from_data(TensorData::new(sb, [dout]), &bdev)),
        };
        let got: Vec<f32> = forward_residual_block(bxb, &bw)
            .to_data()
            .try_to_vec()
            .unwrap();
        let err = max_err(&expected, &got);
        assert!(err < 1e-5, "residual parity failed: {err:.2e}");
    }

    #[test]
    fn prefill_attention_matches_candle_at_real_dims() {
        // Real widths (D_MODEL=1280, 16 heads x 80), 2 patches, random weights.
        // ~30MB of tensors; exercises split/rope/norm/scale/permute/scores/softmax.
        let n = 2usize;
        let dev = CDevice::Cpu;
        // Xavier-like scaling: real checkpoints have small weights, and an
        // unscaled U[-1,1] 1280-wide matrix would amplify 6e-5 stage noise
        // ~500x through the output projection. Scale matrices to O(0.02).
        let mk = |r: usize, c: usize, s: u64| {
            pseudo(r * c, s)
                .into_iter()
                .map(|v| v * 0.02)
                .collect::<Vec<_>>()
        };

        let qkv_v = mk(3 * D_MODEL, D_MODEL, 31);
        let out_v = mk(D_MODEL, D_MODEL, 32);
        let qn_v = pseudo(HEAD_DIM, 33);
        let kn_v = pseudo(HEAD_DIM, 34);
        let qs_raw = pseudo(HEAD_DIM, 35);
        let qs_v = compute_q_scale(qs_raw);
        let xv = pseudo(n * D_MODEL, 36);

        // Candle reference (mirrors super::prefill_attn, with causal mask).
        let cx = CTensor::from_vec(xv.clone(), (1, n, D_MODEL), &dev).unwrap();
        let qkv_w = CTensor::from_vec(qkv_v.clone(), (3 * D_MODEL, D_MODEL), &dev).unwrap();
        let qkv = zsfm_nn::linear(&cx, &qkv_w, None).unwrap();
        let q = qkv.narrow(CD::Minus1, 0, D_MODEL).unwrap();
        let k = qkv.narrow(CD::Minus1, D_MODEL, D_MODEL).unwrap();
        let v = qkv.narrow(CD::Minus1, 2 * D_MODEL, D_MODEL).unwrap();
        let reshape4 = |t: CTensor| t.reshape((1, n, N_HEADS, HEAD_DIM)).unwrap();
        let candle_rope =
            super::super::rope::RopeCache::new(HEAD_DIM, 32, ROPE_THETA, &dev).unwrap();
        let q = candle_rope.apply(&reshape4(q), 0).unwrap();
        let k = candle_rope.apply(&reshape4(k), 0).unwrap();
        let v = reshape4(v);
        let qn = CTensor::from_vec(qn_v.clone(), HEAD_DIM, &dev).unwrap();
        let kn = CTensor::from_vec(kn_v.clone(), HEAD_DIM, &dev).unwrap();
        let q = zsfm_nn::rms_norm(&q, Some(&qn), NORM_EPS as f64).unwrap();
        let k = zsfm_nn::rms_norm(&k, Some(&kn), NORM_EPS as f64).unwrap();
        let qs = CTensor::from_vec(qs_v.clone(), HEAD_DIM, &dev).unwrap();
        let q = q.broadcast_mul(&qs).unwrap();
        let p = |t: CTensor| t.permute([0, 2, 1, 3]).unwrap().contiguous().unwrap();
        let (q, k, v) = (p(q), p(k), p(v));
        let causal = zsfm_nn::make_causal_mask(n, 0, &dev).unwrap();
        let scores = q
            .matmul(&k.transpose(CD::Minus1, CD::Minus2).unwrap())
            .unwrap();
        let scores = scores.broadcast_add(&causal).unwrap();
        let ctx = candle_nn::ops::softmax(&scores, CD::Minus1)
            .unwrap()
            .matmul(&v)
            .unwrap();
        let ctx = ctx
            .permute([0, 2, 1, 3])
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((1, n, D_MODEL))
            .unwrap();
        let out_w = CTensor::from_vec(out_v.clone(), (D_MODEL, D_MODEL), &dev).unwrap();
        let expected: Vec<f32> = zsfm_nn::linear(&ctx, &out_w, None)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Burn.
        let bdev = device();
        let bx = Tensor::<3>::from_data(TensorData::new(xv, [1, n, D_MODEL]), &bdev);
        let bw = BurnAttnW {
            qkv_w: Tensor::<2>::from_data(TensorData::new(qkv_v, [3 * D_MODEL, D_MODEL]), &bdev),
            out_w: Tensor::<2>::from_data(
                TensorData::new(out_v.clone(), [D_MODEL, D_MODEL]),
                &bdev,
            ),
            q_norm_w: Tensor::<1>::from_data(TensorData::new(qn_v, [HEAD_DIM]), &bdev),
            k_norm_w: Tensor::<1>::from_data(TensorData::new(kn_v, [HEAD_DIM]), &bdev),
            q_scale: Tensor::<1>::from_data(TensorData::new(qs_v, [HEAD_DIM]), &bdev),
            pre_norm_w: Tensor::<1>::from_data(
                TensorData::new(vec![1.0f32; D_MODEL], [D_MODEL]),
                &bdev,
            ),
            post_norm_w: Tensor::<1>::from_data(
                TensorData::new(vec![1.0f32; D_MODEL], [D_MODEL]),
                &bdev,
            ),
        };
        let brope = BurnRope::build(HEAD_DIM, 32, ROPE_THETA);
        let t0 = std::time::Instant::now();
        let (got_t, _, _) = burn_prefill_attn(bx, &bw, &brope, n, burn_causal_mask(n));
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let got: Vec<f32> = got_t.to_data().try_to_vec().unwrap();
        let err = max_err(&expected, &got);
        println!("timesfm attn [n={n}]: burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-4, "attention parity failed: {err:.2e}");
    }
}
