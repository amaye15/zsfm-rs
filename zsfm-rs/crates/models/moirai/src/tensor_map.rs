/// Map Moirai safetensors tensor names to GGUF naming convention.
pub fn map_tensor_name(name: &str) -> Option<String> {
    match name {
        "in_proj.weight"          => return Some("in_proj.weight".into()),
        "in_proj.bias"            => return Some("in_proj.bias".into()),
        "mask_encoding.weight"    => return Some("mask_embed.weight".into()),
        "encoder.norm.weight"     => return Some("norm_f.weight".into()),
        // mixture distribution heads
        "param_proj.proj.components.0.loc.weight"        => return Some("head.st_loc.weight".into()),
        "param_proj.proj.components.0.loc.bias"          => return Some("head.st_loc.bias".into()),
        "param_proj.proj.components.0.scale.weight"      => return Some("head.st_scale.weight".into()),
        "param_proj.proj.components.0.scale.bias"        => return Some("head.st_scale.bias".into()),
        "param_proj.proj.components.0.df.weight"         => return Some("head.st_df.weight".into()),
        "param_proj.proj.components.0.df.bias"           => return Some("head.st_df.bias".into()),
        "param_proj.proj.weights_logits.weight"          => return Some("head.mix_w.weight".into()),
        "param_proj.proj.weights_logits.bias"            => return Some("head.mix_w.bias".into()),
        // skip unused distribution components (Normal, NB, LogNormal)
        _ if name.starts_with("param_proj.proj.components.1.")
          || name.starts_with("param_proj.proj.components.2.")
          || name.starts_with("param_proj.proj.components.3.") => return None,
        _ => {}
    }

    if let Some(rest) = name.strip_prefix("encoder.layers.") {
        let (n_str, rest) = rest.split_once('.')?;
        let n: u32 = n_str.parse().ok()?;

        let suffix = match rest {
            "norm1.weight"                         => "norm1.weight",
            "norm2.weight"                         => "norm2.weight",
            "self_attn.q_proj.weight"              => "attn_q.weight",
            "self_attn.k_proj.weight"              => "attn_k.weight",
            "self_attn.v_proj.weight"              => "attn_v.weight",
            "self_attn.out_proj.weight"            => "attn_o.weight",
            "self_attn.q_norm.weight"              => "attn_qn.weight",
            "self_attn.k_norm.weight"              => "attn_kn.weight",
            "self_attn.var_attn_bias.emb.weight"   => "attn_vbias.weight",
            "ffn.fc1.weight"                       => "ffn_fc1.weight",
            "ffn.fc2.weight"                       => "ffn_fc2.weight",
            "ffn.fc_gate.weight"                   => "ffn_gate.weight",
            _ => return None,
        };
        return Some(format!("blk.{n}.{suffix}"));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_proj() {
        assert_eq!(map_tensor_name("in_proj.weight"), Some("in_proj.weight".into()));
    }

    #[test]
    fn encoder_block() {
        assert_eq!(
            map_tensor_name("encoder.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".into())
        );
        assert_eq!(
            map_tensor_name("encoder.layers.23.ffn.fc_gate.weight"),
            Some("blk.23.ffn_gate.weight".into())
        );
    }

    #[test]
    fn head() {
        assert_eq!(
            map_tensor_name("param_proj.proj.components.0.loc.weight"),
            Some("head.st_loc.weight".into())
        );
    }
}
