use anyhow::Result;
use candle_core::{Device, Tensor};

/// Build Llama-style RoPE cos/sin tables for positions `[0, max_seq)`.
///
/// Returns `(cos, sin)` each shaped `[max_seq, head_dim]` (interleaved
/// `cat([freqs, freqs])` layout). Shared by Chronos, Sundial, TimesFM, Toto.
pub fn rope_tables(
    head_dim: usize,
    max_seq: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64) as f32)
        .collect();
    build_rope_tables(&inv_freq, head_dim, max_seq, device)
}

/// Build cos/sin tables from explicit `inv_freq` (`[half]`).
pub fn build_rope_tables(
    inv_freq: &[f32],
    head_dim: usize,
    max_seq: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    anyhow::ensure!(inv_freq.len() == half, "inv_freq len must be head_dim/2");
    let inv = Tensor::from_vec(inv_freq.to_vec(), (half,), device)?;
    let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, (max_seq,), device)?;
    let pos_col = positions.unsqueeze(1)?;
    let inv_row = inv.unsqueeze(0)?;
    let freqs = pos_col.broadcast_mul(&inv_row)?;
    let emb = Tensor::cat(&[&freqs, &freqs], 1)?;
    Ok((
        emb.cos()?.to_dtype(candle_core::DType::F32)?,
        emb.sin()?.to_dtype(candle_core::DType::F32)?,
    ))
}

/// Apply interleaved (paired) rotary embeddings in place on host data.
///
/// Rotates pairs `(2i, 2i+1)` of the first `rope_dim` dims of each
/// `[n_heads, seq_len, head_dim]` row using flat `[max_pos * half_rope]`
/// tables. Shared by Moirai-2, TabICL, TabPFN, Lag-Llama (rank arg covers
/// their dim ordering differences).
#[allow(clippy::too_many_arguments)]
pub fn apply_interleaved_rope(
    x_data: &mut [f32],
    time_ids: &[usize],
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    rope_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) -> anyhow::Result<()> {
    let half_rope = rope_dim / 2;
    anyhow::ensure!(rope_dim % 2 == 0, "rope_dim must be even");
    anyhow::ensure!(
        x_data.len() == n_heads * seq_len * head_dim,
        "x_data len mismatch"
    );
    for h in 0..n_heads {
        for (t, &pos) in time_ids.iter().enumerate() {
            let base = h * seq_len * head_dim + t * head_dim;
            let trig_base = pos * half_rope;
            anyhow::ensure!(
                trig_base + half_rope <= cos_table.len(),
                "rope table too small for pos {pos}"
            );
            for i in 0..half_rope {
                let cos_v = cos_table[trig_base + i];
                let sin_v = sin_table[trig_base + i];
                let ie = base + 2 * i;
                let io = base + 2 * i + 1;
                let ve = x_data[ie];
                let vo = x_data[io];
                x_data[ie] = cos_v * ve - sin_v * vo;
                x_data[io] = sin_v * ve + cos_v * vo;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_shape() {
        let device = Device::Cpu;
        let (cos, sin) = rope_tables(8, 4, 10_000.0, &device).unwrap();
        assert_eq!(cos.dims(), &[4, 8]);
        assert_eq!(sin.dims(), &[4, 8]);
    }

    #[test]
    fn interleaved_rope_identity_at_pos0() {
        // At pos 0, cos=1 sin=0 → input unchanged.
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let cos = vec![1.0f32, 1.0];
        let sin = vec![0.0f32, 0.0];
        apply_interleaved_rope(&mut x, &[0], 1, 1, 4, 4, &cos, &sin).unwrap();
        assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
