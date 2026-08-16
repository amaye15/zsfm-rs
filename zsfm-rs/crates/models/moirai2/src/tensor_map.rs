/// Map Moirai-2.0-R-small safetensors tensor names to GGUF naming convention.
pub fn map_tensor_name(name: &str) -> Option<String> {
    match name {
        // ResidualBlock in_proj
        "in_proj.hidden_layer.weight"   => return Some("in_proj.hidden.weight".into()),
        "in_proj.hidden_layer.bias"     => return Some("in_proj.hidden.bias".into()),
        "in_proj.output_layer.weight"   => return Some("in_proj.output.weight".into()),
        "in_proj.output_layer.bias"     => return Some("in_proj.output.bias".into()),
        "in_proj.residual_layer.weight" => return Some("in_proj.residual.weight".into()),
        "in_proj.residual_layer.bias"   => return Some("in_proj.residual.bias".into()),
        // ResidualBlock out_proj
        "out_proj.hidden_layer.weight"   => return Some("out_proj.hidden.weight".into()),
        "out_proj.hidden_layer.bias"     => return Some("out_proj.hidden.bias".into()),
        "out_proj.output_layer.weight"   => return Some("out_proj.output.weight".into()),
        "out_proj.output_layer.bias"     => return Some("out_proj.output.bias".into()),
        "out_proj.residual_layer.weight" => return Some("out_proj.residual.weight".into()),
        "out_proj.residual_layer.bias"   => return Some("out_proj.residual.bias".into()),
        // Final norm
        "encoder.norm.weight" => return Some("norm_f.weight".into()),
        _ => {}
    }

    if let Some(rest) = name.strip_prefix("encoder.layers.") {
        let (n_str, rest) = rest.split_once('.')?;
        let n: u32 = n_str.parse().ok()?;

        let suffix = match rest {
            "norm1.weight"                       => "norm1.weight",
            "norm2.weight"                       => "norm2.weight",
            "self_attn.q_proj.weight"            => "attn_q.weight",
            "self_attn.k_proj.weight"            => "attn_k.weight",
            "self_attn.v_proj.weight"            => "attn_v.weight",
            "self_attn.out_proj.weight"          => "attn_o.weight",
            "self_attn.q_norm.weight"            => "attn_qn.weight",
            "self_attn.k_norm.weight"            => "attn_kn.weight",
            "self_attn.var_attn_bias.emb.weight" => "attn_vbias.weight",
            "ffn.fc1.weight"                     => "ffn_fc1.weight",
            "ffn.fc2.weight"                     => "ffn_fc2.weight",
            "ffn.fc_gate.weight"                 => "ffn_gate.weight",
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
        assert_eq!(
            map_tensor_name("in_proj.hidden_layer.weight"),
            Some("in_proj.hidden.weight".into())
        );
    }

    #[test]
    fn encoder_block() {
        assert_eq!(
            map_tensor_name("encoder.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".into())
        );
        assert_eq!(
            map_tensor_name("encoder.layers.5.ffn.fc_gate.weight"),
            Some("blk.5.ffn_gate.weight".into())
        );
    }

    #[test]
    fn out_proj() {
        assert_eq!(
            map_tensor_name("out_proj.output_layer.bias"),
            Some("out_proj.output.bias".into())
        );
    }
}
