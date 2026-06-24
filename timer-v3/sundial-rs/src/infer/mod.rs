pub mod rope;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_core::quantized::gguf_file;
use candle_nn::ops;
use std::collections::HashMap;
use std::sync::Mutex;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use rope::RopeCache;

const N_HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const EMBED_IN: usize = 32;  // 2 * input_token_len
const PATCH_SIZE: usize = 16;
const TIME_DIM: usize = 256;
const MAX_SEQ: usize = 4096;
const ROPE_THETA: f64 = 10000.0;

struct EmbedW {
    hidden_w: Tensor, hidden_b: Tensor,
    output_w: Tensor, output_b: Tensor,
    skip_w: Tensor,   skip_b: Tensor,
}

struct AttnW {
    qkv_w: Tensor, qkv_b: Tensor,
    o_w: Tensor,
}

struct NormW { w: Tensor, b: Tensor }

struct BlockW {
    attn: AttnW,
    attn_norm: NormW,
    ffn_norm: NormW,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
}

struct FlowResW {
    ln: NormW,
    mlp1_w: Tensor, mlp1_b: Tensor,
    mlp2_w: Tensor, mlp2_b: Tensor,
    adaln_w: Tensor, adaln_b: Tensor,
}

struct FlowW {
    t1_w: Tensor, t1_b: Tensor,
    t2_w: Tensor, t2_b: Tensor,
    cond_w: Tensor, cond_b: Tensor,
    in_w: Tensor, in_b: Tensor,
    res: Vec<FlowResW>,
    out_w: Tensor, out_b: Tensor,
    out_adaln_w: Tensor, out_adaln_b: Tensor,
}

pub struct SundialModel {
    device: Device,
    embed: EmbedW,
    blocks: Vec<BlockW>,
    norm: NormW,
    flow: FlowW,
    rope: RopeCache,
    n_steps: usize,
    output_len: usize,
    // Precomputed: sinusoidal_embed → t1_proj → silu → t2_proj for steps 0..=n_steps
    t_emb_table: Vec<Tensor>,
    causal_mask_cache: Mutex<HashMap<usize, Tensor>>,
}

fn get_u32(content: &gguf_file::Content, key: &str) -> Option<u32> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::U32(v)) => Some(*v),
        Some(gguf_file::Value::U64(v)) => Some(*v as u32),
        _ => None,
    }
}

fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let in_dims = x.dims().to_vec();
    let d_in = *in_dims.last().unwrap();
    let d_out = w.dim(0)?;
    let n = x.elem_count() / d_in;
    let out2 = x.reshape((n, d_in))?.matmul(&w.t()?)?;
    let mut out_shape = in_dims;
    *out_shape.last_mut().unwrap() = d_out;
    let out = out2.reshape(out_shape)?;
    match b {
        Some(b) => Ok(out.broadcast_add(b)?),
        None => Ok(out),
    }
}

fn layer_norm(x: &Tensor, nw: &NormW) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(D::Minus1)?;
    let std = var.affine(1.0, 1e-5)?.sqrt()?;
    let normed = diff.broadcast_div(&std)?;
    Ok(normed.broadcast_mul(&nw.w)?.broadcast_add(&nw.b)?)
}

fn layer_norm_no_params(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let diff = x.broadcast_sub(&mean)?;
    let var = diff.sqr()?.mean_keepdim(D::Minus1)?;
    let std = var.affine(1.0, 1e-5)?.sqrt()?;
    Ok(diff.broadcast_div(&std)?)
}

fn silu(x: &Tensor) -> Result<Tensor> {
    ops::silu(x).map_err(anyhow::Error::from)
}

fn make_causal_mask(seq: usize, device: &Device) -> Result<Tensor> {
    let neg_inf = f32::NEG_INFINITY;
    let mask: Vec<f32> = (0..seq * seq)
        .map(|i| if (i % seq) <= (i / seq) { 0.0 } else { neg_inf })
        .collect();
    Tensor::from_vec(mask, (1, 1, seq, seq), device).map_err(anyhow::Error::from)
}

fn sinusoidal_embed(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000.0f32.ln() * i as f32 / half as f32)).exp())
        .collect();
    let args: Vec<f32> = freqs.iter().map(|&f| t * f).collect();
    let mut emb = Vec::with_capacity(dim);
    for &a in &args { emb.push(a.cos()); }
    for &a in &args { emb.push(a.sin()); }
    emb
}

impl SundialModel {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let f = File::open(path)
            .map_err(|e| anyhow::anyhow!("open {}: {}", path.display(), e))?;
        let mut reader = BufReader::new(f);
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| anyhow::anyhow!("read gguf: {}", e))?;

        let n_layers = get_u32(&content, "sundial1.block_count").unwrap_or(12) as usize;
        let n_steps = get_u32(&content, "sundial1.flow.num_sampling_steps").unwrap_or(50) as usize;
        let flow_depth = get_u32(&content, "sundial1.flow.depth").unwrap_or(3) as usize;
        let output_len = get_u32(&content, "sundial1.output_token_len").unwrap_or(720) as usize;

        macro_rules! lt {
            ($name:expr) => {{
                let name: &str = $name;
                let qt = content.tensor(&mut reader, name, device)
                    .map_err(|e| anyhow::anyhow!("load {}: {}", name, e))?;
                qt.dequantize(device)
                    .map_err(|e| anyhow::anyhow!("dequantize {}: {}", name, e))?
            }};
        }

        let embed = EmbedW {
            hidden_w: lt!("embed.hidden.weight"),
            hidden_b: lt!("embed.hidden.bias"),
            output_w: lt!("embed.output.weight"),
            output_b: lt!("embed.output.bias"),
            skip_w:   lt!("embed.skip.weight"),
            skip_b:   lt!("embed.skip.bias"),
        };

        let mut blocks = Vec::with_capacity(n_layers);
        for n in 0..n_layers {
            let p = |s: &str| format!("blk.{n}.{s}");
            let q_w = lt!(&p("attn_q.weight"));
            let k_w = lt!(&p("attn_k.weight"));
            let v_w = lt!(&p("attn_v.weight"));
            let q_b = lt!(&p("attn_q.bias"));
            let k_b = lt!(&p("attn_k.bias"));
            let v_b = lt!(&p("attn_v.bias"));
            let qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)
                .map_err(|e| anyhow::anyhow!("qkv_w cat blk.{n}: {e}"))?;
            let qkv_b = Tensor::cat(&[&q_b, &k_b, &v_b], 0)
                .map_err(|e| anyhow::anyhow!("qkv_b cat blk.{n}: {e}"))?;
            blocks.push(BlockW {
                attn: AttnW {
                    qkv_w,
                    qkv_b,
                    o_w: lt!(&p("attn_out.weight")),
                },
                attn_norm: NormW {
                    w: lt!(&p("attn_norm.weight")), b: lt!(&p("attn_norm.bias")),
                },
                ffn_norm: NormW {
                    w: lt!(&p("ffn_norm.weight")), b: lt!(&p("ffn_norm.bias")),
                },
                gate_w: lt!(&p("ffn_gate.weight")),
                up_w:   lt!(&p("ffn_up.weight")),
                down_w: lt!(&p("ffn_down.weight")),
            });
        }

        let norm = NormW {
            w: lt!("norm.weight"),
            b: lt!("norm.bias"),
        };

        let mut res_blocks = Vec::with_capacity(flow_depth);
        for k in 0..flow_depth {
            let fp = |s: &str| format!("flow.res.{k}.{s}");
            res_blocks.push(FlowResW {
                ln: NormW { w: lt!(&fp("ln.weight")), b: lt!(&fp("ln.bias")) },
                mlp1_w: lt!(&fp("mlp1.weight")), mlp1_b: lt!(&fp("mlp1.bias")),
                mlp2_w: lt!(&fp("mlp2.weight")), mlp2_b: lt!(&fp("mlp2.bias")),
                adaln_w: lt!(&fp("adaln.weight")), adaln_b: lt!(&fp("adaln.bias")),
            });
        }

        let flow = FlowW {
            t1_w: lt!("flow.t_proj1.weight"), t1_b: lt!("flow.t_proj1.bias"),
            t2_w: lt!("flow.t_proj2.weight"), t2_b: lt!("flow.t_proj2.bias"),
            cond_w: lt!("flow.cond.weight"),  cond_b: lt!("flow.cond.bias"),
            in_w:  lt!("flow.in_proj.weight"), in_b: lt!("flow.in_proj.bias"),
            res: res_blocks,
            out_w:      lt!("flow.out_linear.weight"),
            out_b:      lt!("flow.out_linear.bias"),
            out_adaln_w: lt!("flow.out_adaln.weight"),
            out_adaln_b: lt!("flow.out_adaln.bias"),
        };

        let rope = RopeCache::new(HEAD_DIM, MAX_SEQ, ROPE_THETA);

        // Precompute time embeddings for all ODE steps (sinusoidal → t1_proj → silu → t2_proj)
        let t_emb_table: Vec<Tensor> = (0..=n_steps)
            .map(|i| {
                let t_scaled = i as f32 / n_steps as f32 * 1000.0;
                let t_raw = Tensor::from_vec(sinusoidal_embed(t_scaled, TIME_DIM), (1, TIME_DIM), device)?;
                let t_h = silu(&linear(&t_raw, &flow.t1_w, Some(&flow.t1_b))?)?;
                linear(&t_h, &flow.t2_w, Some(&flow.t2_b))
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|e| anyhow::anyhow!("t_emb_table: {e}"))?;

        Ok(Self {
            device: device.clone(),
            embed, blocks, norm, flow, rope, n_steps, output_len, t_emb_table,
            causal_mask_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn forecast(&self, context: &[f32], device: &Device) -> Result<Vec<f32>> {
        let n = context.len();
        // ReVIN: whole-series population mean/std
        let mean = context.iter().sum::<f32>() / n as f32;
        let var = context.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let std = (var + 1e-5).sqrt();
        let normed: Vec<f32> = context.iter().map(|&x| (x - mean) / std).collect();

        // Pad to multiple of patch_size and build embedding input [values | mask]
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

        let x = Tensor::from_vec(embed_in, (1, n_patches, EMBED_IN), device)?;
        let mut h = self.embed_forward(&x)?;

        let mask = {
            let mut cache = self.causal_mask_cache.lock().unwrap();
            if !cache.contains_key(&n_patches) {
                cache.insert(n_patches, make_causal_mask(n_patches, &self.device)?);
            }
            cache[&n_patches].clone()
        };
        for block in &self.blocks {
            h = self.block_forward(&h, block, &mask)?;
        }
        h = layer_norm(&h, &self.norm)?;

        // Last patch hidden state as flow condition
        let cond = h.narrow(1, n_patches - 1, 1)?.squeeze(1)?;  // [1, hidden]

        // Flow matching: Euler from t=0 (noise) to t=1 (data)
        let output = self.flow_sample(&cond, device)?;

        let vals = output.to_vec2::<f32>()?;
        Ok(vals[0].iter().map(|&v| v * std + mean).collect())
    }

    fn embed_forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = silu(&linear(x, &self.embed.hidden_w, Some(&self.embed.hidden_b))?)?;
        let out = linear(&h, &self.embed.output_w, Some(&self.embed.output_b))?;
        let skip = linear(x, &self.embed.skip_w, Some(&self.embed.skip_b))?;
        Ok((out + skip)?)
    }

    fn block_forward(&self, x: &Tensor, blk: &BlockW, mask: &Tensor) -> Result<Tensor> {
        let h = layer_norm(x, &blk.attn_norm)?;
        let h = self.attn_forward(&h, &blk.attn, mask)?;
        let x = (x + &h)?;
        let h = layer_norm(&x, &blk.ffn_norm)?;
        let h = self.ffn_forward(&h, blk)?;
        Ok((x + h)?)
    }

    fn attn_forward(&self, x: &Tensor, attn: &AttnW, mask: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let d = N_HEADS * HEAD_DIM;
        let qkv = linear(x, &attn.qkv_w, Some(&attn.qkv_b))?;
        let q = qkv.narrow(D::Minus1, 0, d)?.contiguous()?;
        let k = qkv.narrow(D::Minus1, d, d)?.contiguous()?;
        let v = qkv.narrow(D::Minus1, 2 * d, d)?.contiguous()?;

        // [b, seq, H*D] → [b, H, seq, D]
        let q = q.reshape((b, seq, N_HEADS, HEAD_DIM))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.reshape((b, seq, N_HEADS, HEAD_DIM))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.reshape((b, seq, N_HEADS, HEAD_DIM))?.permute((0, 2, 1, 3))?.contiguous()?;

        let q = self.rope.apply(&q, 0)?;
        let k = self.rope.apply(&k, 0)?;

        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?.affine(scale, 0.0)?;
        let scores = scores.broadcast_add(mask)?;
        let aw = ops::softmax(&scores, D::Minus1)?;

        let out = aw.matmul(&v)?;
        let out = out.permute((0, 2, 1, 3))?.reshape((b, seq, N_HEADS * HEAD_DIM))?;
        linear(&out, &attn.o_w, None)
    }

    fn ffn_forward(&self, x: &Tensor, blk: &BlockW) -> Result<Tensor> {
        let gate = silu(&linear(x, &blk.gate_w, None)?)?;
        let up = linear(x, &blk.up_w, None)?;
        let h = (gate * &up)?;
        linear(&h, &blk.down_w, None)
    }

    // Evaluate the flow network at a given state x and timestep-conditioned activations c_act.
    fn flow_net_eval(&self, x: &Tensor, c_act: &Tensor) -> Result<Tensor> {
        let mut h = linear(x, &self.flow.in_w, Some(&self.flow.in_b))?;
        for res in &self.flow.res {
            let adaln = linear(c_act, &res.adaln_w, Some(&res.adaln_b))?;
            let third = adaln.dim(D::Minus1)? / 3;
            let shift = adaln.narrow(D::Minus1, 0, third)?;
            let scale = adaln.narrow(D::Minus1, third, third)?;
            let gate  = adaln.narrow(D::Minus1, 2 * third, third)?;
            let h_norm = layer_norm(&h, &res.ln)?;
            let h_mod = h_norm.broadcast_mul(&scale.affine(1.0, 1.0)?)?.broadcast_add(&shift)?;
            let h_mlp = silu(&linear(&h_mod, &res.mlp1_w, Some(&res.mlp1_b))?)?;
            let h_mlp = linear(&h_mlp, &res.mlp2_w, Some(&res.mlp2_b))?;
            h = (h + (gate * h_mlp)?)?;
        }
        let adaln = linear(c_act, &self.flow.out_adaln_w, Some(&self.flow.out_adaln_b))?;
        let half = adaln.dim(D::Minus1)? / 2;
        let shift = adaln.narrow(D::Minus1, 0, half)?;
        let scale = adaln.narrow(D::Minus1, half, half)?;
        let h_norm = layer_norm_no_params(&h)?;
        let h_mod = h_norm.broadcast_mul(&scale.affine(1.0, 1.0)?)?.broadcast_add(&shift)?;
        linear(&h_mod, &self.flow.out_w, Some(&self.flow.out_b))
    }

    // Heun's method (2nd-order Runge-Kutta) for the flow ODE.
    // Each step does 2 network evals but converges quadratically, so n_steps/2 fewer steps
    // are needed compared to Euler for equivalent accuracy. For OT-FM (linear paths) the
    // default 50-step Euler run can be replaced by ~10 Heun steps with near-identical output.
    fn flow_sample(&self, cond: &Tensor, device: &Device) -> Result<Tensor> {
        let cond_emb = linear(cond, &self.flow.cond_w, Some(&self.flow.cond_b))?;
        let dt = 1.0f64 / self.n_steps as f64;
        let mut x = Tensor::zeros((1, self.output_len), DType::F32, device)?;

        // c[i] = silu(t_emb_table[i] + cond_emb); c2 at step i == c1 at step i+1,
        // so compute once and carry forward to halve the add+silu cost.
        let mut c_cur = silu(&(&self.t_emb_table[0] + &cond_emb)?)?;
        for i in 0..self.n_steps {
            let k1 = self.flow_net_eval(&x, &c_cur)?;

            let x_pred = (x.clone() + k1.affine(dt, 0.0)?)?;

            let c_next = silu(&(&self.t_emb_table[i + 1] + &cond_emb)?)?;
            let k2 = self.flow_net_eval(&x_pred, &c_next)?;

            x = (x + ((&k1 + &k2)? * (dt * 0.5))?)?;
            c_cur = c_next;
        }
        Ok(x)
    }
}
