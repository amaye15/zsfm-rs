use anyhow::Result;
use candle_core::{Device, Tensor};

/// Partial interleaved RoPE for Moirai-2.0.
///
/// Applies rotary embeddings to the first `rope_dim` dimensions of each head,
/// leaving dims [rope_dim..head_dim] unchanged. Uses interleaved (paired) rotation:
/// position 2i and 2i+1 share the same frequency theta_i.
///
/// cos_table/sin_table: precomputed flat [max_pos * half_rope] tables.
/// x shape: [n_heads, seq_len, head_dim]
/// time_ids: position index for each token in the sequence.
pub fn apply_partial_rope(
    x: &Tensor,
    time_ids: &[usize],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    device: &Device,
    cos_table: &[f32],
    sin_table: &[f32],
) -> Result<Tensor> {
    let seq_len = time_ids.len();
    let half_rope = rope_dim / 2;

    let mut x_data: Vec<f32> = x.flatten_all()?.to_vec1()?;

    for h in 0..n_heads {
        for (t, &pos) in time_ids.iter().enumerate() {
            let base = h * seq_len * head_dim + t * head_dim;
            let trig_base = pos * half_rope;
            for i in 0..half_rope {
                let cos_v = cos_table[trig_base + i];
                let sin_v = sin_table[trig_base + i];
                let idx_even = base + 2 * i;
                let idx_odd  = base + 2 * i + 1;
                let v_even = x_data[idx_even];
                let v_odd  = x_data[idx_odd];
                x_data[idx_even] = cos_v * v_even - sin_v * v_odd;
                x_data[idx_odd]  = sin_v * v_even + cos_v * v_odd;
            }
        }
    }

    Ok(Tensor::from_vec(x_data, (n_heads, seq_len, head_dim), device)?)
}
