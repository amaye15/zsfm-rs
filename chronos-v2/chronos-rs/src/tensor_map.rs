/// Convert a Chronos-2 HuggingFace tensor name to the GGUF naming convention.
/// Returns `None` for unrecognised names (caller will warn and skip them).
///
/// Tensor names are derived from the Python class hierarchy in model.py / layers.py.
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // Token embedding ([PAD] / [REG])
    if hf_name == "shared.weight" {
        return Some("token_embd.weight".into());
    }

    // Input patch embedding (ResidualBlock: in_dim=3*patch_size, h_dim=d_ff, out_dim=d_model)
    match hf_name {
        "input_patch_embedding.hidden_layer.weight"   => return Some("input_patch.hidden.weight".into()),
        "input_patch_embedding.hidden_layer.bias"     => return Some("input_patch.hidden.bias".into()),
        "input_patch_embedding.output_layer.weight"   => return Some("input_patch.output.weight".into()),
        "input_patch_embedding.output_layer.bias"     => return Some("input_patch.output.bias".into()),
        "input_patch_embedding.residual_layer.weight" => return Some("input_patch.skip.weight".into()),
        "input_patch_embedding.residual_layer.bias"   => return Some("input_patch.skip.bias".into()),
        _ => {}
    }

    // Output patch embedding (ResidualBlock: in_dim=d_model, h_dim=d_ff, out_dim=num_q*patch_size)
    match hf_name {
        "output_patch_embedding.hidden_layer.weight"   => return Some("output_patch.hidden.weight".into()),
        "output_patch_embedding.hidden_layer.bias"     => return Some("output_patch.hidden.bias".into()),
        "output_patch_embedding.output_layer.weight"   => return Some("output_patch.output.weight".into()),
        "output_patch_embedding.output_layer.bias"     => return Some("output_patch.output.bias".into()),
        "output_patch_embedding.residual_layer.weight" => return Some("output_patch.skip.weight".into()),
        "output_patch_embedding.residual_layer.bias"   => return Some("output_patch.skip.bias".into()),
        _ => {}
    }

    // Final encoder layer norm
    if hf_name == "encoder.final_layer_norm.weight" {
        return Some("enc_norm.weight".into());
    }

    // Per-block tensors: encoder.block.{N}.layer.{L}.<suffix>
    let rest = hf_name.strip_prefix("encoder.block.")?;
    let (block_str, rest) = rest.split_once('.')?;
    let block: u32 = block_str.parse().ok()?;

    let rest = rest.strip_prefix("layer.")?;
    let (layer_str, suffix) = rest.split_once('.')?;
    let layer: u32 = layer_str.parse().ok()?;

    let gguf_suffix = match layer {
        0 => map_time_attn_suffix(suffix)?,
        1 => map_group_attn_suffix(suffix)?,
        2 => map_ffn_suffix(suffix)?,
        _ => return None,
    };

    Some(format!("blk.{block}.{gguf_suffix}"))
}

fn map_time_attn_suffix(suffix: &str) -> Option<&'static str> {
    Some(match suffix {
        "self_attention.q.weight" => "time_attn.q.weight",
        "self_attention.k.weight" => "time_attn.k.weight",
        "self_attention.v.weight" => "time_attn.v.weight",
        "self_attention.o.weight" => "time_attn.o.weight",
        "layer_norm.weight"       => "time_attn_norm.weight",
        _ => return None,
    })
}

fn map_group_attn_suffix(suffix: &str) -> Option<&'static str> {
    Some(match suffix {
        "self_attention.q.weight" => "group_attn.q.weight",
        "self_attention.k.weight" => "group_attn.k.weight",
        "self_attention.v.weight" => "group_attn.v.weight",
        "self_attention.o.weight" => "group_attn.o.weight",
        "layer_norm.weight"       => "group_attn_norm.weight",
        _ => return None,
    })
}

fn map_ffn_suffix(suffix: &str) -> Option<&'static str> {
    Some(match suffix {
        "mlp.wi.weight"    => "ffn.wi.weight",
        "mlp.wo.weight"    => "ffn.wo.weight",
        "layer_norm.weight" => "ffn_norm.weight",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_embd() {
        assert_eq!(map_tensor_name("shared.weight"), Some("token_embd.weight".into()));
    }

    #[test]
    fn input_patch() {
        assert_eq!(
            map_tensor_name("input_patch_embedding.hidden_layer.weight"),
            Some("input_patch.hidden.weight".into())
        );
        assert_eq!(
            map_tensor_name("input_patch_embedding.residual_layer.bias"),
            Some("input_patch.skip.bias".into())
        );
    }

    #[test]
    fn output_patch() {
        assert_eq!(
            map_tensor_name("output_patch_embedding.output_layer.weight"),
            Some("output_patch.output.weight".into())
        );
    }

    #[test]
    fn enc_norm() {
        assert_eq!(
            map_tensor_name("encoder.final_layer_norm.weight"),
            Some("enc_norm.weight".into())
        );
    }

    #[test]
    fn block_time_attn() {
        assert_eq!(
            map_tensor_name("encoder.block.0.layer.0.self_attention.q.weight"),
            Some("blk.0.time_attn.q.weight".into())
        );
        assert_eq!(
            map_tensor_name("encoder.block.5.layer.0.layer_norm.weight"),
            Some("blk.5.time_attn_norm.weight".into())
        );
    }

    #[test]
    fn block_group_attn() {
        assert_eq!(
            map_tensor_name("encoder.block.3.layer.1.self_attention.o.weight"),
            Some("blk.3.group_attn.o.weight".into())
        );
        assert_eq!(
            map_tensor_name("encoder.block.3.layer.1.layer_norm.weight"),
            Some("blk.3.group_attn_norm.weight".into())
        );
    }

    #[test]
    fn block_ffn() {
        assert_eq!(
            map_tensor_name("encoder.block.2.layer.2.mlp.wi.weight"),
            Some("blk.2.ffn.wi.weight".into())
        );
        assert_eq!(
            map_tensor_name("encoder.block.2.layer.2.mlp.wo.weight"),
            Some("blk.2.ffn.wo.weight".into())
        );
        assert_eq!(
            map_tensor_name("encoder.block.2.layer.2.layer_norm.weight"),
            Some("blk.2.ffn_norm.weight".into())
        );
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(map_tensor_name("some.unknown.tensor"), None);
    }
}
