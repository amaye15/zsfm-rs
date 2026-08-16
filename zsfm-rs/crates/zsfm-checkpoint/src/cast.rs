//! Canonical dtype casting for GGUF conversion: {F32, F16, BF16} sources to any
//! writable [`GGMLType`], including Q8_0 block quantization. Bit-for-bit the same
//! math as the per-model converters (which predate this crate and keep their own
//! verified copies).

use zsfm_gguf::GGMLType;

/// Source dtype of raw checkpoint tensor bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrcDtype {
    F32,
    F16,
    BF16,
}

impl SrcDtype {
    pub fn bytes_per_elem(self) -> usize {
        match self {
            SrcDtype::F32 => 4,
            SrcDtype::F16 | SrcDtype::BF16 => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SrcDtype::F32 => "F32",
            SrcDtype::F16 => "F16",
            SrcDtype::BF16 => "BF16",
        }
    }
}

/// Cast raw little-endian tensor bytes from `src` to `dst`.
///
/// Same-dtype casts are byte-for-byte passthrough. F32→BF16 truncates the
/// mantissa (matching the existing converters); everything routed through F32
/// uses exact widening.
pub fn cast_data(data: &[u8], src: SrcDtype, dst: GGMLType) -> anyhow::Result<Vec<u8>> {
    // Passthrough when the representation already matches.
    match (src, dst) {
        (SrcDtype::F32, GGMLType::F32)
        | (SrcDtype::F16, GGMLType::F16)
        | (SrcDtype::BF16, GGMLType::BF16) => return Ok(data.to_vec()),
        _ => {}
    }

    let f32_values = decode_to_f32(data, src)?;
    match dst {
        GGMLType::F32 => Ok(f32_to_bytes(&f32_values)),
        GGMLType::F16 => Ok(f32_values
            .iter()
            .flat_map(|&v| f32_to_f16_bits(v).to_le_bytes())
            .collect()),
        GGMLType::BF16 => Ok(f32_values
            .iter()
            .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()),
        GGMLType::Q8_0 => quantize_q8_0(&f32_values),
    }
}

/// Decode raw bytes of `src` dtype into f32 values (exact for all three sources).
pub fn decode_to_f32(data: &[u8], src: SrcDtype) -> anyhow::Result<Vec<f32>> {
    match src {
        SrcDtype::F32 => parse_f32_le(data),
        SrcDtype::F16 => {
            if data.len() % 2 != 0 {
                anyhow::bail!("f16 data length not divisible by 2");
            }
            Ok(data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        SrcDtype::BF16 => {
            if data.len() % 2 != 0 {
                anyhow::bail!("bf16 data length not divisible by 2");
            }
            Ok(data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect())
        }
    }
}

pub fn f32_to_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn quantize_q8_0(values: &[f32]) -> anyhow::Result<Vec<u8>> {
    const BLOCK: usize = 32;
    if values.len() % BLOCK != 0 {
        anyhow::bail!(
            "Q8_0 requires element count divisible by {BLOCK}, got {}",
            values.len()
        );
    }
    let n_blocks = values.len() / BLOCK;
    let mut out = vec![0u8; n_blocks * 34];
    for b in 0..n_blocks {
        let blk = &values[b * BLOCK..(b + 1) * BLOCK];
        let amax = blk.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        let d = if amax == 0.0 { 0.0f32 } else { amax / 127.0 };
        let d_inv = if d == 0.0 { 0.0f32 } else { 1.0 / d };
        let base = b * 34;
        out[base..base + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for i in 0..BLOCK {
            out[base + 2 + i] = (blk[i] * d_inv).round().clamp(-127.0, 127.0) as i8 as u8;
        }
    }
    Ok(out)
}

fn parse_f32_le(data: &[u8]) -> anyhow::Result<Vec<f32>> {
    if data.len() % 4 != 0 {
        anyhow::bail!("f32 data length not divisible by 4");
    }
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = (mantissa | 0x0080_0000) >> (1 - new_exp);
        return sign | (m >> 13) as u16;
    }
    sign | ((new_exp as u16) << 10) | (mantissa >> 13) as u16
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x03FF) as u32;
    let f32_bits = if exp == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut m = mantissa;
            let mut e = 0i32;
            while m & 0x0400 == 0 {
                m <<= 1;
                e += 1;
            }
            sign | ((127 - 15 - e + 1) as u32) << 23 | (m & 0x03FF) << 13
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mantissa << 13)
    } else {
        sign | ((exp + 127 - 15) as u32) << 23 | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_exact_values() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.099975586] {
            assert_eq!(f16_to_f32(f32_to_f16_bits(v)), v, "roundtrip {v}");
        }
    }

    #[test]
    fn passthrough_same_dtype() {
        let bytes: Vec<u8> = (0..16).collect();
        assert_eq!(cast_data(&bytes, SrcDtype::F32, GGMLType::F32).unwrap(), bytes);
        assert_eq!(cast_data(&bytes, SrcDtype::F16, GGMLType::F16).unwrap(), bytes);
        assert_eq!(cast_data(&bytes, SrcDtype::BF16, GGMLType::BF16).unwrap(), bytes);
    }

    #[test]
    fn f32_to_bf16_truncates() {
        let v = 1.2345678f32;
        let out = cast_data(&v.to_le_bytes(), SrcDtype::F32, GGMLType::BF16).unwrap();
        let bits = u16::from_le_bytes([out[0], out[1]]);
        assert_eq!(bits, (v.to_bits() >> 16) as u16);
    }

    #[test]
    fn bf16_to_f32_exact_widening() {
        let bits: u16 = 0x3FA0; // 1.25 in bf16
        let out = cast_data(&bits.to_le_bytes(), SrcDtype::BF16, GGMLType::F32).unwrap();
        let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
        assert_eq!(v, 1.25);
    }

    #[test]
    fn q8_0_block_layout() {
        let vals: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let out = quantize_q8_0(&vals).unwrap();
        assert_eq!(out.len(), 34);
        let d = f16_to_f32(u16::from_le_bytes([out[0], out[1]]));
        assert!((d - 31.0 / 127.0).abs() < 1e-3);
        assert_eq!(out[2] as i8, 0); // 0.0 quantizes to 0
        assert_eq!(out[33] as i8, 127); // amax quantizes to 127
    }

    #[test]
    fn q8_0_rejects_partial_block() {
        assert!(quantize_q8_0(&[1.0f32; 31]).is_err());
    }
}
