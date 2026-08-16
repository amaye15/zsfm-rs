/// Map a HuggingFace TimesFM tensor name to its GGUF blk.N.* equivalent.
///
/// HF parameter naming comes from PyTorch nn.Module traversal of
/// `TimesFM_2p5_200M_torch_module`. Returns `None` for any name that
/// should not be included in the GGUF (currently none are skipped).
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // Tokenizer (ResidualBlock, with bias)
    match hf_name {
        "tokenizer.hidden_layer.weight"   => return Some("tokenizer.hidden.weight".into()),
        "tokenizer.hidden_layer.bias"     => return Some("tokenizer.hidden.bias".into()),
        "tokenizer.output_layer.weight"   => return Some("tokenizer.output.weight".into()),
        "tokenizer.output_layer.bias"     => return Some("tokenizer.output.bias".into()),
        "tokenizer.residual_layer.weight" => return Some("tokenizer.skip.weight".into()),
        "tokenizer.residual_layer.bias"   => return Some("tokenizer.skip.bias".into()),
        // Output projection — point head (ResidualBlock, no bias)
        "output_projection_point.hidden_layer.weight"   => return Some("out_point.hidden.weight".into()),
        "output_projection_point.output_layer.weight"   => return Some("out_point.output.weight".into()),
        "output_projection_point.residual_layer.weight" => return Some("out_point.skip.weight".into()),
        // Output projection — quantile head (ResidualBlock, no bias; stored but not used in basic infer)
        "output_projection_quantiles.hidden_layer.weight"   => return Some("out_quantile.hidden.weight".into()),
        "output_projection_quantiles.output_layer.weight"   => return Some("out_quantile.output.weight".into()),
        "output_projection_quantiles.residual_layer.weight" => return Some("out_quantile.skip.weight".into()),
        _ => {}
    }

    // Transformer blocks: stacked_xf.{N}.*
    let rest = hf_name.strip_prefix("stacked_xf.")?;
    let (block_str, rest) = rest.split_once('.')?;
    let block: u32 = block_str.parse().ok()?;

    let gguf_suffix = match rest {
        "pre_attn_ln.scale"                       => "pre_attn_norm.weight",
        "post_attn_ln.scale"                      => "post_attn_norm.weight",
        "attn.qkv_proj.weight"                    => "attn_qkv.weight",
        "attn.out.weight"                         => "attn_out.weight",
        "attn.query_ln.scale"                     => "attn_q_norm.weight",
        "attn.key_ln.scale"                       => "attn_k_norm.weight",
        "attn.per_dim_scale.per_dim_scale"        => "attn_q_scale.weight",
        "pre_ff_ln.scale"                         => "pre_ff_norm.weight",
        "post_ff_ln.scale"                        => "post_ff_norm.weight",
        "ff0.weight"                              => "ffn_up.weight",
        "ff1.weight"                              => "ffn_down.weight",
        _ => return None,
    };

    Some(format!("blk.{block}.{gguf_suffix}"))
}
