/// Map TabDPT safetensors tensor names to GGUF naming convention. The per-layer `kappa`/
/// `max_len_f`/`n0` registered buffers are skipped — they're constants derived from
/// `base_len`/`max_len` (identical across every layer), recomputed from config instead of
/// round-tripped through GGUF.
pub fn map_tensor_name(name: &str) -> Option<String> {
    match name {
        "encoder.weight" => return Some("encoder.weight".into()),
        "encoder.bias" => return Some("encoder.bias".into()),
        "head.0.weight" => return Some("head_fc1.weight".into()),
        "head.0.bias" => return Some("head_fc1.bias".into()),
        "head.2.weight" => return Some("head_fc2.weight".into()),
        "head.2.bias" => return Some("head_fc2.bias".into()),
        "thinking_embed" => return Some("thinking_embed".into()),
        _ => {}
    }

    if let Some(rest) = name.strip_prefix("transformer_encoder.") {
        let (n_str, rest) = rest.split_once('.')?;
        let n: u32 = n_str.parse().ok()?;
        let suffix = match rest {
            "attn_norm.weight" => "attn_norm.weight",
            "attn_norm.bias" => "attn_norm.bias",
            "ff_norm.weight" => "ff_norm.weight",
            "ff_norm.bias" => "ff_norm.bias",
            "q_proj.weight" => "q_proj.weight",
            "k_proj.weight" => "k_proj.weight",
            "v_proj.weight" => "v_proj.weight",
            "out_proj.weight" => "out_proj.weight",
            "q_gate.weight" => "q_gate.weight",
            "q_norm.weight" => "q_norm.weight",
            "k_norm.weight" => "k_norm.weight",
            "ff.up.weight" => "ff_up.weight",
            "ff.down.weight" => "ff_down.weight",
            "kappa" | "max_len_f" | "n0" => return None,
            _ => return None,
        };
        return Some(format!("blk.{n}.{suffix}"));
    }

    if let Some(rest) = name.strip_prefix("y_encoders.") {
        let (n_str, rest) = rest.split_once('.')?;
        let n: u32 = n_str.parse().ok()?;
        let suffix = match rest {
            "0.weight" => "fc1.weight",
            "0.bias" => "fc1.bias",
            "2.weight" => "fc2.weight",
            "2.bias" => "fc2.bias",
            _ => return None,
        };
        return Some(format!("y_enc.{n}.{suffix}"));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level() {
        assert_eq!(map_tensor_name("encoder.weight"), Some("encoder.weight".into()));
        assert_eq!(map_tensor_name("head.2.bias"), Some("head_fc2.bias".into()));
        assert_eq!(map_tensor_name("thinking_embed"), Some("thinking_embed".into()));
    }

    #[test]
    fn block_and_y_encoder() {
        assert_eq!(map_tensor_name("transformer_encoder.0.q_proj.weight"), Some("blk.0.q_proj.weight".into()));
        assert_eq!(map_tensor_name("transformer_encoder.31.ff.down.weight"), Some("blk.31.ff_down.weight".into()));
        assert_eq!(map_tensor_name("y_encoders.5.2.weight"), Some("y_enc.5.fc2.weight".into()));
    }

    #[test]
    fn skips_scalar_buffers() {
        assert_eq!(map_tensor_name("transformer_encoder.0.kappa"), None);
        assert_eq!(map_tensor_name("transformer_encoder.0.max_len_f"), None);
        assert_eq!(map_tensor_name("transformer_encoder.0.n0"), None);
    }
}
