//! Sundial inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle): patch embedding, transformer blocks
//! with Llama-style RoPE, and the flow-matching ODE sampler (Heun). Weight
//! loading goes through the existing candle GGUF reader and converts each
//! F32 tensor to Burn `TensorData`, so there is exactly one GGUF parser.
//! Architecture widths mirror the fixed constants in `super`; layer counts
//! come from GGUF metadata exactly like the candle path.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use zsfm_burn::linear::burn_linear_nd;
use zsfm_burn::norm::burn_layer_norm_nd;

const N_HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const EMBED_IN: usize = 32;
const PATCH_SIZE: usize = 16;
const TIME_DIM: usize = 256;
const MAX_SEQ: usize = 4096;
const ROPE_THETA: f64 = 10000.0;

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

fn linear<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    burn_linear_nd(x, w, b)
}

fn layer_norm<const D: usize>(x: Tensor<D>, w: Tensor<1>, b: Tensor<1>) -> Tensor<D> {
    burn_layer_norm_nd(x, w, b, 1e-5)
}

fn silu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::silu(x)
}

fn softmax_last<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::softmax(x, D - 1)
}

struct BurnEmbedW {
    hidden_w: Tensor<2>,
    hidden_b: Tensor<1>,
    output_w: Tensor<2>,
    output_b: Tensor<1>,
    skip_w: Tensor<2>,
    skip_b: Tensor<1>,
}

struct BurnAttnW {
    qkv_w: Tensor<2>,
    qkv_b: Tensor<1>,
    o_w: Tensor<2>,
}

struct BurnNormW {
    w: Tensor<1>,
    b: Tensor<1>,
}

struct BurnBlockW {
    attn: BurnAttnW,
    attn_norm: BurnNormW,
    ffn_norm: BurnNormW,
    gate_w: Tensor<2>,
    up_w: Tensor<2>,
    down_w: Tensor<2>,
}

struct BurnFlowResW {
    ln_w: Tensor<1>,
    ln_b: Tensor<1>,
    mlp1_w: Tensor<2>,
    mlp1_b: Tensor<1>,
    mlp2_w: Tensor<2>,
    mlp2_b: Tensor<1>,
    adaln_w: Tensor<2>,
    adaln_b: Tensor<1>,
}

struct BurnFlowW {
    t1_w: Tensor<2>,
    t1_b: Tensor<1>,
    t2_w: Tensor<2>,
    t2_b: Tensor<1>,
    cond_w: Tensor<2>,
    cond_b: Tensor<1>,
    in_w: Tensor<2>,
    in_b: Tensor<1>,
    res: Vec<BurnFlowResW>,
    out_w: Tensor<2>,
    out_b: Tensor<1>,
    out_adaln_w: Tensor<2>,
    out_adaln_b: Tensor<1>,
}

pub struct BurnSundialModel {
    embed: BurnEmbedW,
    blocks: Vec<BurnBlockW>,
    norm_w: Tensor<1>,
    norm_b: Tensor<1>,
    flow: BurnFlowW,
    rope_cos: Tensor<2>,
    rope_sin: Tensor<2>,
    n_steps: usize,
    output_len: usize,
    t_emb_table: Vec<Tensor<1>>,
}

fn get_u32(content: &gguf_file::Content, key: &str) -> Option<u32> {
    use candle_core::quantized::gguf_file::Value;
    match content.metadata.get(key) {
        Some(Value::U32(v)) => Some(*v),
        Some(Value::U64(v)) => Some(*v as u32),
        _ => None,
    }
}

fn sinusoidal_embed(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000.0f32.ln() * i as f32 / half as f32)).exp())
        .collect();
    let args: Vec<f32> = freqs.iter().map(|&f| t * f).collect();
    let mut emb = Vec::with_capacity(dim);
    for &a in &args {
        emb.push(a.cos());
    }
    for &a in &args {
        emb.push(a.sin());
    }
    emb
}

impl BurnSundialModel {
    pub fn load(path: &Path, steps_override: Option<usize>) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let n_layers = get_u32(&content, "sundial1.block_count").unwrap_or(12) as usize;
        let gguf_n_steps =
            get_u32(&content, "sundial1.flow.num_sampling_steps").unwrap_or(50) as usize;
        let n_steps = steps_override.unwrap_or(gguf_n_steps);
        let flow_depth = get_u32(&content, "sundial1.flow.depth").unwrap_or(3) as usize;
        let output_len = get_u32(&content, "sundial1.output_token_len").unwrap_or(720) as usize;

        let embed = BurnEmbedW {
            hidden_w: load_t(&content, &mut reader, "embed.hidden.weight")?,
            hidden_b: load_v(&content, &mut reader, "embed.hidden.bias")?,
            output_w: load_t(&content, &mut reader, "embed.output.weight")?,
            output_b: load_v(&content, &mut reader, "embed.output.bias")?,
            skip_w: load_t(&content, &mut reader, "embed.skip.weight")?,
            skip_b: load_v(&content, &mut reader, "embed.skip.bias")?,
        };

        let mut blocks = Vec::with_capacity(n_layers);
        for n in 0..n_layers {
            let qw = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &format!("blk.{n}.attn_q.weight"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let kw = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &format!("blk.{n}.attn_k.weight"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let vw = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &format!("blk.{n}.attn_v.weight"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let qb = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &format!("blk.{n}.attn_q.bias"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let kb = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &format!("blk.{n}.attn_k.bias"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let vb = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &format!("blk.{n}.attn_v.bias"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let dev = device();
            let flat1 =
                |t: &candle_core::Tensor| -> Result<Vec<f32>> { Ok(t.flatten_all()?.to_vec1()?) };
            let mut wcat = Vec::new();
            for t in [&qw, &kw, &vw] {
                wcat.extend_from_slice(&flat1(t)?);
            }
            let qkv_w = Tensor::<2>::from_data(
                TensorData::new(wcat, [qw.dim(0)? + kw.dim(0)? + vw.dim(0)?, qw.dim(1)?]),
                &dev,
            );
            let mut bcat = Vec::new();
            for t in [&qb, &kb, &vb] {
                bcat.extend_from_slice(&flat1(t)?);
            }
            let qkv_b = Tensor::<1>::from_data(
                TensorData::new(bcat, [qb.elem_count() + kb.elem_count() + vb.elem_count()]),
                &dev,
            );
            blocks.push(BurnBlockW {
                attn: BurnAttnW {
                    qkv_w,
                    qkv_b,
                    o_w: load_t(&content, &mut reader, &format!("blk.{n}.attn_out.weight"))?,
                },
                attn_norm: BurnNormW {
                    w: load_v(&content, &mut reader, &format!("blk.{n}.attn_norm.weight"))?,
                    b: load_v(&content, &mut reader, &format!("blk.{n}.attn_norm.bias"))?,
                },
                ffn_norm: BurnNormW {
                    w: load_v(&content, &mut reader, &format!("blk.{n}.ffn_norm.weight"))?,
                    b: load_v(&content, &mut reader, &format!("blk.{n}.ffn_norm.bias"))?,
                },
                gate_w: load_t(&content, &mut reader, &format!("blk.{n}.ffn_gate.weight"))?,
                up_w: load_t(&content, &mut reader, &format!("blk.{n}.ffn_up.weight"))?,
                down_w: load_t(&content, &mut reader, &format!("blk.{n}.ffn_down.weight"))?,
            });
        }

        let norm_w = load_v(&content, &mut reader, "norm.weight")?;
        let norm_b = load_v(&content, &mut reader, "norm.bias")?;

        let mut res_blocks = Vec::with_capacity(flow_depth);
        for k in 0..flow_depth {
            let fp = |s: &str| format!("flow.res.{k}.{s}");
            res_blocks.push(BurnFlowResW {
                ln_w: load_v(&content, &mut reader, &fp("ln.weight"))?,
                ln_b: load_v(&content, &mut reader, &fp("ln.bias"))?,
                mlp1_w: load_t(&content, &mut reader, &fp("mlp1.weight"))?,
                mlp1_b: load_v(&content, &mut reader, &fp("mlp1.bias"))?,
                mlp2_w: load_t(&content, &mut reader, &fp("mlp2.weight"))?,
                mlp2_b: load_v(&content, &mut reader, &fp("mlp2.bias"))?,
                adaln_w: load_t(&content, &mut reader, &fp("adaln.weight"))?,
                adaln_b: load_v(&content, &mut reader, &fp("adaln.bias"))?,
            });
        }
        let flow = BurnFlowW {
            t1_w: load_t(&content, &mut reader, "flow.t_proj1.weight")?,
            t1_b: load_v(&content, &mut reader, "flow.t_proj1.bias")?,
            t2_w: load_t(&content, &mut reader, "flow.t_proj2.weight")?,
            t2_b: load_v(&content, &mut reader, "flow.t_proj2.bias")?,
            cond_w: load_t(&content, &mut reader, "flow.cond.weight")?,
            cond_b: load_v(&content, &mut reader, "flow.cond.bias")?,
            in_w: load_t(&content, &mut reader, "flow.in_proj.weight")?,
            in_b: load_v(&content, &mut reader, "flow.in_proj.bias")?,
            res: res_blocks,
            out_w: load_t(&content, &mut reader, "flow.out_linear.weight")?,
            out_b: load_v(&content, &mut reader, "flow.out_linear.bias")?,
            out_adaln_w: load_t(&content, &mut reader, "flow.out_adaln.weight")?,
            out_adaln_b: load_v(&content, &mut reader, "flow.out_adaln.bias")?,
        };

        let (rope_cos, rope_sin) = build_rope_tables(HEAD_DIM, MAX_SEQ, ROPE_THETA);

        // Precompute time embeddings for all ODE steps.
        let t_emb_table: Vec<Tensor<1>> = (0..=n_steps)
            .map(|i| {
                let t_scaled = i as f32 / n_steps as f32 * 1000.0;
                let t_raw = Tensor::<2>::from_data(
                    TensorData::new(sinusoidal_embed(t_scaled, TIME_DIM), [1, TIME_DIM]),
                    &device(),
                );
                let t_h = silu(linear(t_raw, flow.t1_w.clone(), Some(flow.t1_b.clone())));
                linear(t_h, flow.t2_w.clone(), Some(flow.t2_b.clone()))
                    .reshape([flow.t2_b.dims()[0]])
            })
            .collect();

        Ok(Self {
            embed,
            blocks,
            norm_w,
            norm_b,
            flow,
            rope_cos,
            rope_sin,
            n_steps,
            output_len,
            t_emb_table,
        })
    }

    pub fn forecast(&self, context: &[f32]) -> Result<Vec<f32>> {
        let dev = device();
        let n = context.len();
        let mean = context.iter().sum::<f32>() / n as f32;
        let var = context
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / n as f32;
        let std = (var + 1e-5).sqrt();
        let normed: Vec<f32> = context.iter().map(|&x| (x - mean) / std).collect();

        let n_patches = (n + PATCH_SIZE - 1) / PATCH_SIZE;
        let mut embed_in = vec![0.0f32; n_patches * EMBED_IN];
        for p in 0..n_patches {
            let start = p * PATCH_SIZE;
            let end = (start + PATCH_SIZE).min(n);
            for i in start..end {
                embed_in[p * EMBED_IN + (i - start)] = normed[i];
                embed_in[p * EMBED_IN + PATCH_SIZE + (i - start)] = 1.0;
            }
        }
        let x = Tensor::<3>::from_data(TensorData::new(embed_in, [1, n_patches, EMBED_IN]), &dev);
        let mut h = self.embed_forward(x);
        let mask = burn_causal_mask(n_patches);
        for block in &self.blocks {
            h = self.block_forward(h, block, mask.clone());
        }
        h = layer_norm(h, self.norm_w.clone(), self.norm_b.clone());
        let cond: Tensor<2> = h.narrow(1, n_patches - 1, 1).squeeze_dim(0);
        let output = self.flow_sample(cond);
        let vals: Vec<f32> = output
            .to_data()
            .try_to_vec()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(vals.iter().map(|&v| v * std + mean).collect())
    }

    fn embed_forward(&self, x: Tensor<3>) -> Tensor<3> {
        let h = silu(linear(
            x.clone(),
            self.embed.hidden_w.clone(),
            Some(self.embed.hidden_b.clone()),
        ));
        let out = linear(
            h,
            self.embed.output_w.clone(),
            Some(self.embed.output_b.clone()),
        );
        let skip = linear(
            x,
            self.embed.skip_w.clone(),
            Some(self.embed.skip_b.clone()),
        );
        out + skip
    }

    fn block_forward(&self, x: Tensor<3>, blk: &BurnBlockW, mask: Tensor<4>) -> Tensor<3> {
        let h = layer_norm(x.clone(), blk.attn_norm.w.clone(), blk.attn_norm.b.clone());
        let h = self.attn_forward(h, &blk.attn, mask);
        let x = x + h;
        let h = layer_norm(x.clone(), blk.ffn_norm.w.clone(), blk.ffn_norm.b.clone());
        let h = self.ffn_forward(h, blk);
        x + h
    }

    fn attn_forward(&self, x: Tensor<3>, attn: &BurnAttnW, mask: Tensor<4>) -> Tensor<3> {
        let [b, seq, _] = x.dims();
        let d = N_HEADS * HEAD_DIM;
        let qkv = linear(x, attn.qkv_w.clone(), Some(attn.qkv_b.clone()));
        let q: Tensor<3> = qkv.clone().narrow(2, 0, d);
        let k: Tensor<3> = qkv.clone().narrow(2, d, d);
        let v: Tensor<3> = qkv.narrow(2, 2 * d, d);
        let pack = |t: Tensor<3>| t.reshape([b, seq, N_HEADS, HEAD_DIM]).permute([0, 2, 1, 3]);
        let (q, k, v) = (pack(q), pack(k), pack(v));
        let q = self.rope_apply(q, 0);
        let k = self.rope_apply(k, 0);
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let scores = q.matmul(k.transpose()).mul_scalar(scale);
        let scores = scores + mask;
        let aw = softmax_last(scores);
        let out = aw.matmul(v);
        let out: Tensor<3> = out.permute([0, 2, 1, 3]).reshape([b, seq, d]);
        linear(out, attn.o_w.clone(), None)
    }

    fn rope_apply(&self, x: Tensor<4>, start_pos: usize) -> Tensor<4> {
        let dims = x.dims();
        let seq = dims[2];
        let hd = dims[3];
        let half = hd / 2;
        let cos = self
            .rope_cos
            .clone()
            .narrow(0, start_pos, seq)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0);
        let sin = self
            .rope_sin
            .clone()
            .narrow(0, start_pos, seq)
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0);
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.clone().narrow(3, half, half);
        let rotated = Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3);
        x * cos + rotated * sin
    }

    fn ffn_forward(&self, x: Tensor<3>, blk: &BurnBlockW) -> Tensor<3> {
        let gate = silu(linear(x.clone(), blk.gate_w.clone(), None));
        let up = linear(x, blk.up_w.clone(), None);
        linear(gate * up, blk.down_w.clone(), None)
    }

    fn flow_net_eval(&self, x: Tensor<2>, c_act: Tensor<2>) -> Tensor<2> {
        let mut h = linear(x, self.flow.in_w.clone(), Some(self.flow.in_b.clone()));
        for res in &self.flow.res {
            let adaln = linear(
                c_act.clone(),
                res.adaln_w.clone(),
                Some(res.adaln_b.clone()),
            );
            let third = adaln.dims()[1] / 3;
            let shift = adaln.clone().narrow(1, 0, third);
            let scale = adaln.clone().narrow(1, third, third);
            let gate = adaln.narrow(1, 2 * third, third);
            let h_norm = layer_norm(h.clone(), res.ln_w.clone(), res.ln_b.clone());
            let h_mod = h_norm * scale.add_scalar(1.0) + shift;
            let h_mlp = silu(linear(h_mod, res.mlp1_w.clone(), Some(res.mlp1_b.clone())));
            let h_mlp = linear(h_mlp, res.mlp2_w.clone(), Some(res.mlp2_b.clone()));
            h = h + gate * h_mlp;
        }
        let adaln = linear(
            c_act,
            self.flow.out_adaln_w.clone(),
            Some(self.flow.out_adaln_b.clone()),
        );
        let half = adaln.dims()[1] / 2;
        let shift = adaln.clone().narrow(1, 0, half);
        let scale = adaln.narrow(1, half, half);
        let h_norm = burn_layer_norm_nd_no_affine(h);
        let h_mod = h_norm * scale.add_scalar(1.0) + shift;
        linear(
            h_mod,
            self.flow.out_w.clone(),
            Some(self.flow.out_b.clone()),
        )
    }

    fn flow_sample(&self, cond: Tensor<2>) -> Tensor<2> {
        let dev = device();
        let cond_emb = linear(
            cond,
            self.flow.cond_w.clone(),
            Some(self.flow.cond_b.clone()),
        );
        let dt = 1.0f32 / self.n_steps as f32;
        let mut x = Tensor::<2>::zeros([1, self.output_len], &dev);
        let mut c_cur = silu(self.t_emb_table[0].clone().unsqueeze_dim::<2>(0) + cond_emb.clone());
        for i in 0..self.n_steps {
            let k1 = self.flow_net_eval(x.clone(), c_cur.clone());
            let x_pred = x.clone() + k1.clone().mul_scalar(dt);
            let c_next =
                silu(self.t_emb_table[i + 1].clone().unsqueeze_dim::<2>(0) + cond_emb.clone());
            let k2 = self.flow_net_eval(x_pred, c_next.clone());
            x = x + (k1 + k2).mul_scalar(dt * 0.5);
            c_cur = c_next;
        }
        x
    }
}

fn burn_layer_norm_nd_no_affine(x: Tensor<2>) -> Tensor<2> {
    zsfm_burn::norm::burn_layer_norm_no_affine(x, 1e-5)
}

fn build_rope_tables(head_dim: usize, max_seq: usize, theta: f64) -> (Tensor<2>, Tensor<2>) {
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
    (
        Tensor::<2>::from_data(TensorData::new(cos, [max_seq, head_dim]), &dev),
        Tensor::<2>::from_data(TensorData::new(sin, [max_seq, head_dim]), &dev),
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use zsfm_gguf::{GGMLType, GGUFMetaValue, GGUFWriter};

    fn pseudo(n: usize, seed: u64) -> Vec<f32> {
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

    fn put2(w: &mut GGUFWriter, name: &str, rows: usize, cols: usize, seed: &mut u64, scale: f32) {
        *seed += 1;
        let data: Vec<f32> = pseudo(rows * cols, *seed)
            .iter()
            .map(|v| v * scale)
            .collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor(name, vec![cols as u64, rows as u64], GGMLType::F32, bytes);
    }

    fn put1(w: &mut GGUFWriter, name: &str, n: usize, seed: &mut u64) {
        *seed += 1;
        let bytes: Vec<u8> = pseudo(n, *seed)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.add_tensor(name, vec![n as u64], GGMLType::F32, bytes);
    }

    #[test]
    fn burn_matches_candle_on_synthetic_gguf() {
        // Real widths (768/12x64), 1 layer, depth 1, 2 ODE steps, output 4.
        let dir = std::env::temp_dir().join(format!("sundial-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("sundial".into()),
        );
        w.add_metadata("sundial1.block_count", GGUFMetaValue::Uint32(1));
        w.add_metadata("sundial1.flow.num_sampling_steps", GGUFMetaValue::Uint32(2));
        w.add_metadata("sundial1.flow.depth", GGUFMetaValue::Uint32(1));
        w.add_metadata("sundial1.output_token_len", GGUFMetaValue::Uint32(4));
        let mut seed = 700u64;
        let s = 0.05;
        put2(&mut w, "embed.hidden.weight", 32, 32, &mut seed, s);
        put1(&mut w, "embed.hidden.bias", 32, &mut seed);
        put2(&mut w, "embed.output.weight", 768, 32, &mut seed, s);
        put1(&mut w, "embed.output.bias", 768, &mut seed);
        put2(&mut w, "embed.skip.weight", 768, 32, &mut seed, s);
        put1(&mut w, "embed.skip.bias", 768, &mut seed);
        for t in ["attn_q", "attn_k", "attn_v"] {
            put2(&mut w, &format!("blk.0.{t}.weight"), 768, 768, &mut seed, s);
            put1(&mut w, &format!("blk.0.{t}.bias"), 768, &mut seed);
        }
        put2(&mut w, "blk.0.attn_out.weight", 768, 768, &mut seed, s);
        put1(&mut w, "blk.0.attn_norm.weight", 768, &mut seed);
        put1(&mut w, "blk.0.attn_norm.bias", 768, &mut seed);
        put1(&mut w, "blk.0.ffn_norm.weight", 768, &mut seed);
        put1(&mut w, "blk.0.ffn_norm.bias", 768, &mut seed);
        put2(&mut w, "blk.0.ffn_gate.weight", 768, 768, &mut seed, s);
        put2(&mut w, "blk.0.ffn_up.weight", 768, 768, &mut seed, s);
        put2(&mut w, "blk.0.ffn_down.weight", 768, 768, &mut seed, s);
        put1(&mut w, "norm.weight", 768, &mut seed);
        put1(&mut w, "norm.bias", 768, &mut seed);
        put2(&mut w, "flow.t_proj1.weight", 8, 256, &mut seed, s);
        put1(&mut w, "flow.t_proj1.bias", 8, &mut seed);
        put2(&mut w, "flow.t_proj2.weight", 8, 8, &mut seed, s);
        put1(&mut w, "flow.t_proj2.bias", 8, &mut seed);
        put2(&mut w, "flow.cond.weight", 8, 768, &mut seed, s);
        put1(&mut w, "flow.cond.bias", 8, &mut seed);
        put2(&mut w, "flow.in_proj.weight", 8, 4, &mut seed, s);
        put1(&mut w, "flow.in_proj.bias", 8, &mut seed);
        put1(&mut w, "flow.res.0.ln.weight", 8, &mut seed);
        put1(&mut w, "flow.res.0.ln.bias", 8, &mut seed);
        put2(&mut w, "flow.res.0.mlp1.weight", 8, 8, &mut seed, s);
        put1(&mut w, "flow.res.0.mlp1.bias", 8, &mut seed);
        put2(&mut w, "flow.res.0.mlp2.weight", 8, 8, &mut seed, s);
        put1(&mut w, "flow.res.0.mlp2.bias", 8, &mut seed);
        put2(&mut w, "flow.res.0.adaln.weight", 24, 8, &mut seed, s);
        put1(&mut w, "flow.res.0.adaln.bias", 24, &mut seed);
        put2(&mut w, "flow.out_linear.weight", 4, 8, &mut seed, s);
        put1(&mut w, "flow.out_linear.bias", 4, &mut seed);
        put2(&mut w, "flow.out_adaln.weight", 16, 8, &mut seed, s);
        put1(&mut w, "flow.out_adaln.bias", 16, &mut seed);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let candle = crate::SundialModel::load(&path, &candle_core::Device::Cpu, None).unwrap();
        let burn = BurnSundialModel::load(&path, None).unwrap();
        let ctx: Vec<f32> = (0..16).map(|i| 10.0 + 0.5 * i as f32).collect();
        let t0 = std::time::Instant::now();
        let a = candle.forecast(&ctx, &candle_core::Device::Cpu).unwrap();
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let b = burn.forecast(&ctx).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(a.len(), b.len());
        let err: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        println!("sundial synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-3, "sundial Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
