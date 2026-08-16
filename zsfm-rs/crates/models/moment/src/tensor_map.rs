/// Map MOMENT safetensors tensor names to GGUF naming convention.
pub fn map_tensor_name(name: &str) -> Option<String> {
    match name {
        "patch_embedding.value_embedding.weight"    => return Some("patch_embed.weight".into()),
        "patch_embedding.position_embedding.pe"     => return Some("pos_embed.pe".into()),
        "patch_embedding.mask_embedding"            => return Some("mask_embed".into()),
        "encoder.embed_tokens.weight"               => return Some("token_embed.weight".into()),
        "encoder.final_layer_norm.weight"           => return Some("norm_f.weight".into()),
        "head.linear.weight"                        => return Some("head.weight".into()),
        "head.linear.bias"                          => return Some("head.bias".into()),
        _ => {}
    }

    if let Some(rest) = name.strip_prefix("encoder.block.") {
        let (n_str, rest) = rest.split_once('.')?;
        let n: u32 = n_str.parse().ok()?;

        let suffix = match rest {
            "layer.0.SelfAttention.q.weight"                         => "attn_q.weight",
            "layer.0.SelfAttention.k.weight"                         => "attn_k.weight",
            "layer.0.SelfAttention.v.weight"                         => "attn_v.weight",
            "layer.0.SelfAttention.o.weight"                         => "attn_o.weight",
            "layer.0.SelfAttention.relative_attention_bias.weight"   => "attn_rel_bias.weight",
            "layer.0.layer_norm.weight"                              => "attn_norm.weight",
            "layer.1.DenseReluDense.wi_0.weight"                     => "ffn_wi0.weight",
            "layer.1.DenseReluDense.wi_1.weight"                     => "ffn_wi1.weight",
            "layer.1.DenseReluDense.wo.weight"                       => "ffn_wo.weight",
            "layer.1.layer_norm.weight"                              => "ffn_norm.weight",
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
    fn patch_embed() {
        assert_eq!(
            map_tensor_name("patch_embedding.value_embedding.weight"),
            Some("patch_embed.weight".into())
        );
    }

    #[test]
    fn block_attn() {
        assert_eq!(
            map_tensor_name("encoder.block.0.layer.0.SelfAttention.q.weight"),
            Some("blk.0.attn_q.weight".into())
        );
        assert_eq!(
            map_tensor_name("encoder.block.23.layer.1.DenseReluDense.wo.weight"),
            Some("blk.23.ffn_wo.weight".into())
        );
    }

    #[test]
    fn rel_bias_only_in_block0() {
        assert_eq!(
            map_tensor_name("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
            Some("blk.0.attn_rel_bias.weight".into())
        );
    }
}
