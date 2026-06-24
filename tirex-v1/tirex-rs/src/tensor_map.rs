/// Map a TiRex state_dict tensor name (from PyTorch Lightning .ckpt) to GGUF naming.
///
/// The checkpoint stores block weights under "block_stack.blocks.N.*" and the
/// non-block embeddings / output norm without a "block_stack." prefix.
pub fn map_tensor_name(name: &str) -> Option<String> {
    // Non-block tensors
    match name {
        "block_stack.out_norm.weight" => return Some("out_norm".into()),
        _ => {}
    }

    // input/output patch embedding (no block_stack prefix)
    if let Some(rest) = name.strip_prefix("input_patch_embedding.") {
        let suffix = match rest {
            "hidden_layer.weight"   => "in_emb.hidden.weight",
            "hidden_layer.bias"     => "in_emb.hidden.bias",
            "output_layer.weight"   => "in_emb.output.weight",
            "output_layer.bias"     => "in_emb.output.bias",
            "residual_layer.weight" => "in_emb.residual.weight",
            "residual_layer.bias"   => "in_emb.residual.bias",
            _ => return None,
        };
        return Some(suffix.into());
    }
    if let Some(rest) = name.strip_prefix("output_patch_embedding.") {
        let suffix = match rest {
            "hidden_layer.weight"   => "out_emb.hidden.weight",
            "hidden_layer.bias"     => "out_emb.hidden.bias",
            "output_layer.weight"   => "out_emb.output.weight",
            "output_layer.bias"     => "out_emb.output.bias",
            "residual_layer.weight" => "out_emb.residual.weight",
            "residual_layer.bias"   => "out_emb.residual.bias",
            _ => return None,
        };
        return Some(suffix.into());
    }

    // block_stack.blocks.N.*
    if let Some(rest) = name.strip_prefix("block_stack.blocks.") {
        let (n_str, rest) = rest.split_once('.')?;
        let n: u32 = n_str.parse().ok()?;
        let gguf_suffix = match rest {
            "norm_slstm.weight"                   => "norm_slstm",
            "slstm_layer.fgate.weight"             => "fgate.weight",
            "slstm_layer.igate.weight"             => "igate.weight",
            "slstm_layer.zgate.weight"             => "zgate.weight",
            "slstm_layer.ogate.weight"             => "ogate.weight",
            "slstm_layer.slstm_cell._recurrent_kernel_" => "slstm_kernel",
            "slstm_layer.slstm_cell._bias_"        => "slstm_bias",
            "slstm_layer.group_norm.weight"        => "group_norm",
            "norm_ffn.weight"                      => "norm_ffn",
            "ffn.proj_up_gate.weight"              => "ffn_gate.weight",
            "ffn.proj_up.weight"                   => "ffn_up.weight",
            "ffn.proj_down.weight"                 => "ffn_down.weight",
            _ => return None,
        };
        return Some(format!("blk.{n}.{gguf_suffix}"));
    }

    None
}

/// Whether this tensor needs bias permutation ([NH, NG, DH] → [NG, NH, DH]).
pub fn needs_bias_permute(gguf_name: &str) -> bool {
    gguf_name.ends_with(".slstm_bias")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_block() {
        assert_eq!(
            map_tensor_name("block_stack.out_norm.weight"),
            Some("out_norm".into())
        );
        assert_eq!(
            map_tensor_name("input_patch_embedding.hidden_layer.weight"),
            Some("in_emb.hidden.weight".into())
        );
        assert_eq!(
            map_tensor_name("output_patch_embedding.residual_layer.bias"),
            Some("out_emb.residual.bias".into())
        );
    }

    #[test]
    fn block() {
        assert_eq!(
            map_tensor_name("block_stack.blocks.0.norm_slstm.weight"),
            Some("blk.0.norm_slstm".into())
        );
        assert_eq!(
            map_tensor_name("block_stack.blocks.11.slstm_layer.slstm_cell._bias_"),
            Some("blk.11.slstm_bias".into())
        );
        assert_eq!(
            map_tensor_name("block_stack.blocks.3.ffn.proj_down.weight"),
            Some("blk.3.ffn_down.weight".into())
        );
    }
}
