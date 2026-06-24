/// Map a FlowState HuggingFace tensor name to the GGUF canonical name.
/// Returns `None` for tensors unused at inference time (caller skips them).
///
/// HF prefix in safetensors:  (none — model is saved as FlowStateForPrediction,
/// which wraps FlowStateModel in `self.model`, but the checkpoint stores
/// FlowStateModel weights directly without the `.model.` prefix.)
///
/// Verified tensor names from model.safetensors header:
///   embed.embed.weight / embed.embed.bias
///   encoder.layers.{N}.ssm.{log_Lambda_real, Lambda_imag, B_tilde_r, B_tilde_i,
///                                C_tilde_r, C_tilde_i, D, log_Delta}
///   encoder.layers.{N}.out.{weight, bias}
///   encoder.layers.{N}.norm.{weight, bias}
///   decoder.lin.{weight, bias}
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // Embedding
    match hf_name {
        "embed.embed.weight" => return Some("embed.weight".into()),
        "embed.embed.bias"   => return Some("embed.bias".into()),
        "decoder.lin.weight" => return Some("decoder.weight".into()),
        "decoder.lin.bias"   => return Some("decoder.bias".into()),
        _ => {}
    }

    // encoder.layers.{N}.ssm.*  and  encoder.layers.{N}.{out,norm}.*
    let rest = hf_name.strip_prefix("encoder.layers.")?;
    let dot = rest.find('.')?;
    let n: usize = rest[..dot].parse().ok()?;
    let suffix = &rest[dot + 1..];

    let name = match suffix {
        "ssm.log_Lambda_real" => format!("blk.{n}.ssm.log_lambda_real"),
        "ssm.Lambda_imag"     => format!("blk.{n}.ssm.lambda_imag"),
        "ssm.B_tilde_r"       => format!("blk.{n}.ssm.b_r"),
        "ssm.B_tilde_i"       => format!("blk.{n}.ssm.b_i"),
        "ssm.C_tilde_r"       => format!("blk.{n}.ssm.c_r"),
        "ssm.C_tilde_i"       => format!("blk.{n}.ssm.c_i"),
        "ssm.D"               => format!("blk.{n}.ssm.d"),
        "ssm.log_Delta"       => format!("blk.{n}.ssm.log_delta"),
        "out.weight"          => format!("blk.{n}.out.weight"),
        "out.bias"            => format!("blk.{n}.out.bias"),
        "norm.weight"         => format!("blk.{n}.norm.weight"),
        "norm.bias"           => format!("blk.{n}.norm.bias"),
        _ => return None,
    };
    Some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed() {
        assert_eq!(map_tensor_name("embed.embed.weight"), Some("embed.weight".into()));
        assert_eq!(map_tensor_name("embed.embed.bias"),   Some("embed.bias".into()));
    }

    #[test]
    fn test_decoder() {
        assert_eq!(map_tensor_name("decoder.lin.weight"), Some("decoder.weight".into()));
        assert_eq!(map_tensor_name("decoder.lin.bias"),   Some("decoder.bias".into()));
    }

    #[test]
    fn test_ssm_params() {
        assert_eq!(
            map_tensor_name("encoder.layers.0.ssm.log_Lambda_real"),
            Some("blk.0.ssm.log_lambda_real".into())
        );
        assert_eq!(
            map_tensor_name("encoder.layers.3.ssm.B_tilde_r"),
            Some("blk.3.ssm.b_r".into())
        );
        assert_eq!(
            map_tensor_name("encoder.layers.5.ssm.C_tilde_i"),
            Some("blk.5.ssm.c_i".into())
        );
        assert_eq!(
            map_tensor_name("encoder.layers.2.ssm.log_Delta"),
            Some("blk.2.ssm.log_delta".into())
        );
    }

    #[test]
    fn test_layer_mlp_and_norm() {
        assert_eq!(
            map_tensor_name("encoder.layers.1.out.weight"),
            Some("blk.1.out.weight".into())
        );
        assert_eq!(
            map_tensor_name("encoder.layers.4.norm.bias"),
            Some("blk.4.norm.bias".into())
        );
    }

    #[test]
    fn test_unknown_returns_none() {
        assert_eq!(map_tensor_name("something.unknown"), None);
    }
}
