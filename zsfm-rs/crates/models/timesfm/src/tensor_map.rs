/// Map a HuggingFace TimesFM tensor name to its GGUF blk.N.* equivalent.
///
/// HF parameter naming comes from PyTorch nn.Module traversal of
/// `TimesFM_2p5_200M_torch_module`. Returns `None` for any name that
/// should not be included in the GGUF (currently none are skipped).
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    const HEAD: &[(&str, &str)] = &[
        ("tokenizer.hidden_layer.weight", "tokenizer.hidden.weight"),
        ("tokenizer.hidden_layer.bias", "tokenizer.hidden.bias"),
        ("tokenizer.output_layer.weight", "tokenizer.output.weight"),
        ("tokenizer.output_layer.bias", "tokenizer.output.bias"),
        ("tokenizer.residual_layer.weight", "tokenizer.skip.weight"),
        ("tokenizer.residual_layer.bias", "tokenizer.skip.bias"),
        (
            "output_projection_point.hidden_layer.weight",
            "out_point.hidden.weight",
        ),
        (
            "output_projection_point.output_layer.weight",
            "out_point.output.weight",
        ),
        (
            "output_projection_point.residual_layer.weight",
            "out_point.skip.weight",
        ),
        (
            "output_projection_quantiles.hidden_layer.weight",
            "out_quantile.hidden.weight",
        ),
        (
            "output_projection_quantiles.output_layer.weight",
            "out_quantile.output.weight",
        ),
        (
            "output_projection_quantiles.residual_layer.weight",
            "out_quantile.skip.weight",
        ),
    ];
    if let Some(mapped) = zsfm_gguf::map_with_table(hf_name, HEAD) {
        return Some(mapped);
    }

    // Transformer blocks: stacked_xf.{N}.*
    let rest = hf_name.strip_prefix("stacked_xf.")?;
    let (block_str, rest) = rest.split_once('.')?;
    let block: u32 = block_str.parse().ok()?;

    let gguf_suffix = match rest {
        "pre_attn_ln.scale" => "pre_attn_norm.weight",
        "post_attn_ln.scale" => "post_attn_norm.weight",
        "attn.qkv_proj.weight" => "attn_qkv.weight",
        "attn.out.weight" => "attn_out.weight",
        "attn.query_ln.scale" => "attn_q_norm.weight",
        "attn.key_ln.scale" => "attn_k_norm.weight",
        "attn.per_dim_scale.per_dim_scale" => "attn_q_scale.weight",
        "pre_ff_ln.scale" => "pre_ff_norm.weight",
        "post_ff_ln.scale" => "post_ff_norm.weight",
        "ff0.weight" => "ffn_up.weight",
        "ff1.weight" => "ffn_down.weight",
        _ => return None,
    };

    Some(format!("blk.{block}.{gguf_suffix}"))
}
