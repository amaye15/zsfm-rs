/// Map a HuggingFace Sundial tensor name to its GGUF equivalent.
/// Returns None for tensors that should be skipped.
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // Patch embedding (tokenizer)
    match hf_name {
        "model.embed_layer.hidden_layer.weight"   => return Some("embed.hidden.weight".into()),
        "model.embed_layer.hidden_layer.bias"     => return Some("embed.hidden.bias".into()),
        "model.embed_layer.output_layer.weight"   => return Some("embed.output.weight".into()),
        "model.embed_layer.output_layer.bias"     => return Some("embed.output.bias".into()),
        "model.embed_layer.residual_layer.weight" => return Some("embed.skip.weight".into()),
        "model.embed_layer.residual_layer.bias"   => return Some("embed.skip.bias".into()),
        // Final backbone norm
        "model.norm.weight" => return Some("norm.weight".into()),
        "model.norm.bias"   => return Some("norm.bias".into()),
        // Flow head — timestep embedder
        "flow_loss.net.time_embed.mlp.0.weight" => return Some("flow.t_proj1.weight".into()),
        "flow_loss.net.time_embed.mlp.0.bias"   => return Some("flow.t_proj1.bias".into()),
        "flow_loss.net.time_embed.mlp.2.weight" => return Some("flow.t_proj2.weight".into()),
        "flow_loss.net.time_embed.mlp.2.bias"   => return Some("flow.t_proj2.bias".into()),
        // Flow head — condition/input projections
        "flow_loss.net.cond_embed.weight" => return Some("flow.cond.weight".into()),
        "flow_loss.net.cond_embed.bias"   => return Some("flow.cond.bias".into()),
        "flow_loss.net.input_proj.weight" => return Some("flow.in_proj.weight".into()),
        "flow_loss.net.input_proj.bias"   => return Some("flow.in_proj.bias".into()),
        // Flow head — final layer
        "flow_loss.net.final_layer.linear.weight"             => return Some("flow.out_linear.weight".into()),
        "flow_loss.net.final_layer.linear.bias"               => return Some("flow.out_linear.bias".into()),
        "flow_loss.net.final_layer.adaLN_modulation.1.weight" => return Some("flow.out_adaln.weight".into()),
        "flow_loss.net.final_layer.adaLN_modulation.1.bias"   => return Some("flow.out_adaln.bias".into()),
        _ => {}
    }

    // Transformer blocks
    if let Some(rest) = hf_name.strip_prefix("model.layers.") {
        if let Some((n_str, rest)) = rest.split_once('.') {
            if let Ok(n) = n_str.parse::<u32>() {
                let gguf = match rest {
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.q_proj.bias"   => "attn_q.bias",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.k_proj.bias"   => "attn_k.bias",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.v_proj.bias"   => "attn_v.bias",
                    "self_attn.o_proj.weight" => "attn_out.weight",
                    "norm1.weight"            => "attn_norm.weight",
                    "norm1.bias"              => "attn_norm.bias",
                    "norm2.weight"            => "ffn_norm.weight",
                    "norm2.bias"              => "ffn_norm.bias",
                    "ffn_layer.gate_proj.weight" => "ffn_gate.weight",
                    "ffn_layer.up_proj.weight"   => "ffn_up.weight",
                    "ffn_layer.down_proj.weight" => "ffn_down.weight",
                    _ => return None,
                };
                return Some(format!("blk.{n}.{gguf}"));
            }
        }
    }

    // Flow residual blocks
    if let Some(rest) = hf_name.strip_prefix("flow_loss.net.res_blocks.") {
        if let Some((k_str, rest)) = rest.split_once('.') {
            if let Ok(k) = k_str.parse::<u32>() {
                let gguf = match rest {
                    "in_ln.weight"              => "ln.weight",
                    "in_ln.bias"                => "ln.bias",
                    "mlp.0.weight"              => "mlp1.weight",
                    "mlp.0.bias"                => "mlp1.bias",
                    "mlp.2.weight"              => "mlp2.weight",
                    "mlp.2.bias"                => "mlp2.bias",
                    "adaLN_modulation.1.weight" => "adaln.weight",
                    "adaLN_modulation.1.bias"   => "adaln.bias",
                    _ => return None,
                };
                return Some(format!("flow.res.{k}.{gguf}"));
            }
        }
    }

    None
}
