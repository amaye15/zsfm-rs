/// Convert a Toto HuggingFace tensor name to the GGUF blk.N convention.
/// Returns `None` for unrecognised names (caller will warn and skip them).
///
/// Actual tensor names were confirmed by running `inspect-tensors` on the
/// downloaded model.safetensors checkpoint.
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // --- patch projection (two-layer MLP with skip connection) ---
    match hf_name {
        "patch_proj.linear1.weight" => return Some("patch_proj.linear1.weight".into()),
        "patch_proj.linear1.bias"   => return Some("patch_proj.linear1.bias".into()),
        "patch_proj.linear2.weight" => return Some("patch_proj.linear2.weight".into()),
        "patch_proj.linear2.bias"   => return Some("patch_proj.linear2.bias".into()),
        "patch_proj.skip_proj.weight" => return Some("patch_proj.skip_proj.weight".into()),
        "patch_proj.skip_proj.bias"   => return Some("patch_proj.skip_proj.bias".into()),
        _ => {}
    }

    // --- output head projection (two-layer MLP with skip connection) ---
    match hf_name {
        "output_head.param_projection.proj.linear1.weight" =>
            return Some("output_head.linear1.weight".into()),
        "output_head.param_projection.proj.linear1.bias" =>
            return Some("output_head.linear1.bias".into()),
        "output_head.param_projection.proj.linear2.weight" =>
            return Some("output_head.linear2.weight".into()),
        "output_head.param_projection.proj.linear2.bias" =>
            return Some("output_head.linear2.bias".into()),
        "output_head.param_projection.proj.skip_proj.weight" =>
            return Some("output_head.skip_proj.weight".into()),
        "output_head.param_projection.proj.skip_proj.bias" =>
            return Some("output_head.skip_proj.bias".into()),
        _ => {}
    }

    // --- per-block tensors ---
    // Pattern: transformer.layers.{N}.<suffix>
    let rest = hf_name.strip_prefix("transformer.layers.")?;
    let (block_str, suffix) = rest.split_once('.')?;
    let block: u32 = block_str.parse().ok()?;

    let gguf_suffix = map_block_suffix(suffix)?;
    Some(format!("blk.{block}.{gguf_suffix}"))
}

fn map_block_suffix(suffix: &str) -> Option<&'static str> {
    Some(match suffix {
        // Fused QKV projection
        "attn.in_proj.weight"  => "attn_qkv.weight",
        "attn.in_proj.bias"    => "attn_qkv.bias",
        // Output projection
        "attn.out_proj.weight" => "attn_output.weight",
        "attn.out_proj.bias"   => "attn_output.bias",
        // Per-dimension scale (Toto-specific learned scaling)
        "attn._pds.per_dim_scale" => "attn_pds.weight",
        // Attention temperature (learned scalar per block)
        "attn_tau" => "attn_tau",
        // Feed-forward
        "ffn.fc1.weight" => "ffn_up.weight",
        "ffn.fc1.bias"   => "ffn_up.bias",
        "ffn.fc2.weight" => "ffn_down.weight",
        "ffn.fc2.bias"   => "ffn_down.bias",
        // MLP temperature (learned scalar per block)
        "mlp_tau" => "mlp_tau",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_proj() {
        assert_eq!(
            map_tensor_name("patch_proj.linear1.weight"),
            Some("patch_proj.linear1.weight".into())
        );
        assert_eq!(
            map_tensor_name("patch_proj.skip_proj.bias"),
            Some("patch_proj.skip_proj.bias".into())
        );
    }

    #[test]
    fn block_attn_qkv() {
        assert_eq!(
            map_tensor_name("transformer.layers.0.attn.in_proj.weight"),
            Some("blk.0.attn_qkv.weight".into())
        );
    }

    #[test]
    fn block_attn_pds() {
        assert_eq!(
            map_tensor_name("transformer.layers.3.attn._pds.per_dim_scale"),
            Some("blk.3.attn_pds.weight".into())
        );
    }

    #[test]
    fn block_attn_tau() {
        assert_eq!(
            map_tensor_name("transformer.layers.47.attn_tau"),
            Some("blk.47.attn_tau".into())
        );
    }

    #[test]
    fn block_ffn() {
        assert_eq!(
            map_tensor_name("transformer.layers.12.ffn.fc1.weight"),
            Some("blk.12.ffn_up.weight".into())
        );
        assert_eq!(
            map_tensor_name("transformer.layers.12.ffn.fc2.weight"),
            Some("blk.12.ffn_down.weight".into())
        );
    }

    #[test]
    fn output_head() {
        assert_eq!(
            map_tensor_name("output_head.param_projection.proj.linear2.weight"),
            Some("output_head.linear2.weight".into())
        );
        assert_eq!(
            map_tensor_name("output_head.param_projection.proj.skip_proj.bias"),
            Some("output_head.skip_proj.bias".into())
        );
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(map_tensor_name("some.unknown.tensor"), None);
    }
}
