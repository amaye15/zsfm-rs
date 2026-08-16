//! Maps `tabfm/src/pytorch/model.py` state_dict keys (from `torch.load(pytorch_model.bin)`,
//! after conversion to safetensors) to canonical GGUF tensor names.
//!
//! There's no pre-existing GGUF naming convention for this architecture family, so this module
//! defines one. Stack prefixes mirror the five weight-sharing sub-networks in `TabFM.forward`:
//!   - `cell`     <- `cell_embedder`            (per-cell Fourier embedding)
//!   - `colenc1`  <- `col_embedder`             (SetTransformer, stage 1)
//!   - `colenc2`  <- `col_embedder_2`           (SetTransformer, stage 2)
//!   - `rowenc1`  <- `row_interactor`           (RoPE cross-column attention, stage 1)
//!   - `rowenc2`  <- `row_interactor_2`         (RoPE cross-column attention, stage 2)
//!   - `icl`      <- `icl_predictor`            (24-block in-context-learning attention)
//! `cls_tokens` is a top-level parameter with no stack.
//!
//! Column stacks nest two `MultiheadAttentionBlock`s per transformer block (`mab1`, `mab2` —
//! induced-attention query/apply pair); row and ICL stacks have one attention+FFN sublayer per
//! block. Both share the same leaf naming for a block's attention/FFN/norm weights.

pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    if hf_name == "cls_tokens" {
        return Some("cls_tokens".into());
    }
    if let Some(rest) = hf_name.strip_prefix("cell_embedder.") {
        return map_cell(rest);
    }
    for (prefix, stack) in [
        ("col_embedder_2.", "colenc2"),
        ("col_embedder.", "colenc1"),
        ("row_interactor_2.", "rowenc2"),
        ("row_interactor.", "rowenc1"),
    ] {
        if let Some(rest) = hf_name.strip_prefix(prefix) {
            return map_stack(stack, rest, prefix.starts_with("col"));
        }
    }
    if let Some(rest) = hf_name.strip_prefix("icl_predictor.") {
        return map_icl(rest);
    }
    None
}

fn map_cell(rest: &str) -> Option<String> {
    match rest {
        "fourier_frequencies" => Some("cell.fourier_freq".into()),
        "fourier_frequencies_cat" => Some("cell.fourier_freq_cat".into()),
        "in_linear.weight" => Some("cell.in_linear.weight".into()),
        "in_linear.bias" => Some("cell.in_linear.bias".into()),
        "in_linear_cat.weight" => Some("cell.in_linear_cat.weight".into()),
        "in_linear_cat.bias" => Some("cell.in_linear_cat.bias".into()),
        // classification: y_embedder_lookup is a plain nn.Embedding
        "y_embedder_lookup.weight" => Some("cell.y_embed.weight".into()),
        _ => {
            // regression: y_embedder_lookup is an MLP (layers.0, layers.1)
            rest.strip_prefix("y_embedder_lookup.layers.")
                .and_then(|r| map_mlp_leaf("cell.y_embed", r))
        }
    }
}

/// `rest` is `"{idx}.{weight|bias}"` (a torch `nn.ModuleList` of plain `Linear`s inside an MLP).
fn map_mlp_leaf(base: &str, rest: &str) -> Option<String> {
    let dot = rest.find('.')?;
    let idx: usize = rest[..dot].parse().ok()?;
    let field = &rest[dot + 1..];
    if field != "weight" && field != "bias" {
        return None;
    }
    Some(format!("{base}.mlp.{idx}.{field}"))
}

fn map_stack(stack: &str, rest: &str, is_col: bool) -> Option<String> {
    if is_col {
        if let Some(r) = rest.strip_prefix("tf_col.blocks.") {
            return map_block(stack, r, true);
        }
        match rest {
            "out_w.weight" => Some(format!("{stack}.out_w.weight")),
            "out_w.bias" => Some(format!("{stack}.out_w.bias")),
            "ln_w.weight" => Some(format!("{stack}.out_norm.weight")),
            _ => None,
        }
    } else {
        if rest == "tf_row.rope.freqs" {
            return Some(format!("{stack}.rope_freqs"));
        }
        if let Some(r) = rest.strip_prefix("tf_row.blocks.") {
            return map_block(stack, r, false);
        }
        if rest == "out_ln.weight" {
            return Some(format!("{stack}.out_norm.weight"));
        }
        None
    }
}

/// `rest` is `"{block_idx}.{suffix}"`. `has_mab` selects col-stack blocks (two nested
/// `MultiheadAttentionBlock`s, `mab1`/`mab2`) vs row/ICL blocks (one, unnested).
fn map_block(stack: &str, rest: &str, has_mab: bool) -> Option<String> {
    let dot = rest.find('.')?;
    let n: usize = rest[..dot].parse().ok()?;
    let suffix = &rest[dot + 1..];
    let blk_prefix = format!("{stack}.blk.{n}");

    if has_mab {
        if suffix == "ind_vectors" {
            return Some(format!("{blk_prefix}.ind_vectors"));
        }
        for mab in ["mab1", "mab2"] {
            if let Some(leaf) = suffix.strip_prefix(&format!("{mab}.")) {
                let mapped = map_mab_leaf(leaf)?;
                return Some(format!("{blk_prefix}.{mab}.{mapped}"));
            }
        }
        None
    } else {
        let mapped = map_mab_leaf(suffix)?;
        Some(format!("{blk_prefix}.{mapped}"))
    }
}

/// Maps a `MultiheadAttentionBlock`'s direct-child parameter suffix (attention weights, the
/// four RMSNorms, and the SwiGLU FFN) to its canonical leaf name.
fn map_mab_leaf(leaf: &str) -> Option<String> {
    if let Some(r) = leaf.strip_prefix("attn.") {
        return match r {
            "q_proj.weight" => Some("attn_q.weight".into()),
            "q_proj.bias" => Some("attn_q.bias".into()),
            "k_proj.weight" => Some("attn_k.weight".into()),
            "k_proj.bias" => Some("attn_k.bias".into()),
            "v_proj.weight" => Some("attn_v.weight".into()),
            "v_proj.bias" => Some("attn_v.bias".into()),
            "out_proj.weight" => Some("attn_o.weight".into()),
            "out_proj.bias" => Some("attn_o.bias".into()),
            "query_ln.weight" => Some("q_norm.weight".into()),
            "key_ln.weight" => Some("k_norm.weight".into()),
            "per_dim_scale" => Some("per_dim_scale".into()),
            _ => None,
        };
    }
    match leaf {
        "pre_attn_ln.weight" => Some("pre_attn_norm.weight".into()),
        "post_attn_ln.weight" => Some("post_attn_norm.weight".into()),
        "pre_ff_ln.weight" => Some("pre_ff_norm.weight".into()),
        "post_ff_ln.weight" => Some("post_ff_norm.weight".into()),
        "linear1.weight" => Some("ffn_up.weight".into()),
        "linear1.bias" => Some("ffn_up.bias".into()),
        "linear1_gate.weight" => Some("ffn_gate.weight".into()),
        "linear1_gate.bias" => Some("ffn_gate.bias".into()),
        "linear2.weight" => Some("ffn_down.weight".into()),
        "linear2.bias" => Some("ffn_down.bias".into()),
        _ => None,
    }
}

fn map_icl(rest: &str) -> Option<String> {
    if let Some(r) = rest.strip_prefix("tf_icl.blocks.") {
        return map_block("icl", r, false);
    }
    match rest {
        "ln.weight" => return Some("icl.out_norm.weight".into()),
        "y_encoder.projection.weight" => return Some("icl.y_encoder.projection.weight".into()),
        "y_encoder.projection.bias" => return Some("icl.y_encoder.projection.bias".into()),
        _ => {}
    }
    if let Some(r) = rest.strip_prefix("y_encoder.layers.") {
        return map_mlp_leaf("icl.y_encoder", r);
    }
    if let Some(r) = rest.strip_prefix("decoder.layers.") {
        return map_mlp_leaf("icl.decoder", r);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cls_tokens() {
        assert_eq!(map_tensor_name("cls_tokens"), Some("cls_tokens".into()));
    }

    #[test]
    fn test_cell_embedder_fourier() {
        assert_eq!(
            map_tensor_name("cell_embedder.fourier_frequencies"),
            Some("cell.fourier_freq".into())
        );
        assert_eq!(
            map_tensor_name("cell_embedder.fourier_frequencies_cat"),
            Some("cell.fourier_freq_cat".into())
        );
    }

    #[test]
    fn test_cell_embedder_in_linear() {
        assert_eq!(
            map_tensor_name("cell_embedder.in_linear.weight"),
            Some("cell.in_linear.weight".into())
        );
        assert_eq!(
            map_tensor_name("cell_embedder.in_linear_cat.bias"),
            Some("cell.in_linear_cat.bias".into())
        );
    }

    #[test]
    fn test_cell_embedder_y_embed_classification() {
        assert_eq!(
            map_tensor_name("cell_embedder.y_embedder_lookup.weight"),
            Some("cell.y_embed.weight".into())
        );
    }

    #[test]
    fn test_cell_embedder_y_embed_regression_mlp() {
        assert_eq!(
            map_tensor_name("cell_embedder.y_embedder_lookup.layers.0.weight"),
            Some("cell.y_embed.mlp.0.weight".into())
        );
        assert_eq!(
            map_tensor_name("cell_embedder.y_embedder_lookup.layers.1.bias"),
            Some("cell.y_embed.mlp.1.bias".into())
        );
    }

    #[test]
    fn test_col_stack_ind_vectors_and_mab() {
        assert_eq!(
            map_tensor_name("col_embedder.tf_col.blocks.0.ind_vectors"),
            Some("colenc1.blk.0.ind_vectors".into())
        );
        assert_eq!(
            map_tensor_name("col_embedder_2.tf_col.blocks.2.mab2.attn.q_proj.weight"),
            Some("colenc2.blk.2.mab2.attn_q.weight".into())
        );
        assert_eq!(
            map_tensor_name("col_embedder.tf_col.blocks.1.mab1.attn.per_dim_scale"),
            Some("colenc1.blk.1.mab1.per_dim_scale".into())
        );
        assert_eq!(
            map_tensor_name("col_embedder.tf_col.blocks.0.mab1.pre_ff_ln.weight"),
            Some("colenc1.blk.0.mab1.pre_ff_norm.weight".into())
        );
        assert_eq!(
            map_tensor_name("col_embedder.tf_col.blocks.0.mab2.linear1_gate.weight"),
            Some("colenc1.blk.0.mab2.ffn_gate.weight".into())
        );
    }

    #[test]
    fn test_col_stack_out_projection() {
        assert_eq!(
            map_tensor_name("col_embedder.out_w.weight"),
            Some("colenc1.out_w.weight".into())
        );
        assert_eq!(
            map_tensor_name("col_embedder_2.ln_w.weight"),
            Some("colenc2.out_norm.weight".into())
        );
    }

    #[test]
    fn test_row_stack_rope_and_block() {
        assert_eq!(
            map_tensor_name("row_interactor.tf_row.rope.freqs"),
            Some("rowenc1.rope_freqs".into())
        );
        assert_eq!(
            map_tensor_name("row_interactor_2.tf_row.blocks.2.attn.out_proj.bias"),
            Some("rowenc2.blk.2.attn_o.bias".into())
        );
        assert_eq!(
            map_tensor_name("row_interactor.tf_row.blocks.0.post_attn_ln.weight"),
            Some("rowenc1.blk.0.post_attn_norm.weight".into())
        );
        assert_eq!(
            map_tensor_name("row_interactor.out_ln.weight"),
            Some("rowenc1.out_norm.weight".into())
        );
    }

    #[test]
    fn test_icl_block() {
        assert_eq!(
            map_tensor_name("icl_predictor.tf_icl.blocks.23.linear2.weight"),
            Some("icl.blk.23.ffn_down.weight".into())
        );
        assert_eq!(
            map_tensor_name("icl_predictor.ln.weight"),
            Some("icl.out_norm.weight".into())
        );
    }

    #[test]
    fn test_icl_y_encoder_classification() {
        assert_eq!(
            map_tensor_name("icl_predictor.y_encoder.projection.weight"),
            Some("icl.y_encoder.projection.weight".into())
        );
    }

    #[test]
    fn test_icl_y_encoder_regression_and_decoder() {
        assert_eq!(
            map_tensor_name("icl_predictor.y_encoder.layers.0.weight"),
            Some("icl.y_encoder.mlp.0.weight".into())
        );
        assert_eq!(
            map_tensor_name("icl_predictor.decoder.layers.1.bias"),
            Some("icl.decoder.mlp.1.bias".into())
        );
    }

    #[test]
    fn test_unrecognized_returns_none() {
        assert_eq!(map_tensor_name("some.random.optimizer.state"), None);
    }
}
