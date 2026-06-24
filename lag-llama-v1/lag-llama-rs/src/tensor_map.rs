/// Map a Lag-Llama state_dict tensor name to GGUF naming convention.
pub fn map_tensor_name(name: &str) -> Option<String> {
    match name {
        "model.transformer.wte.weight" => return Some("enc.wte.weight".into()),
        "model.transformer.wte.bias"   => return Some("enc.wte.bias".into()),
        "model.transformer.ln_f.scale" => return Some("norm_f.weight".into()),
        "model.param_proj.proj.0.weight" => return Some("head.mu.weight".into()),
        "model.param_proj.proj.0.bias"   => return Some("head.mu.bias".into()),
        "model.param_proj.proj.1.weight" => return Some("head.sigma.weight".into()),
        "model.param_proj.proj.1.bias"   => return Some("head.sigma.bias".into()),
        "model.param_proj.proj.2.weight" => return Some("head.nu.weight".into()),
        "model.param_proj.proj.2.bias"   => return Some("head.nu.bias".into()),
        _ => {}
    }

    if let Some(rest) = name.strip_prefix("model.transformer.h.") {
        let (n_str, rest) = rest.split_once('.')?;
        let n: u32 = n_str.parse().ok()?;
        let gguf_suffix = match rest {
            "rms_1.scale"          => "rms1.weight",
            "rms_2.scale"          => "rms2.weight",
            "attn.q_proj.weight"   => "attn_q.weight",
            "attn.kv_proj.weight"  => "attn_kv.weight",
            "attn.c_proj.weight"   => "attn_c.weight",
            "mlp.c_fc1.weight"     => "mlp_fc1.weight",
            "mlp.c_fc2.weight"     => "mlp_fc2.weight",
            "mlp.c_proj.weight"    => "mlp_proj.weight",
            _ => return None,
        };
        return Some(format!("blk.{n}.{gguf_suffix}"));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wte() {
        assert_eq!(map_tensor_name("model.transformer.wte.weight"), Some("enc.wte.weight".into()));
    }

    #[test]
    fn block_attn() {
        assert_eq!(
            map_tensor_name("model.transformer.h.0.attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".into())
        );
        assert_eq!(
            map_tensor_name("model.transformer.h.7.mlp.c_proj.weight"),
            Some("blk.7.mlp_proj.weight".into())
        );
    }

    #[test]
    fn head() {
        assert_eq!(
            map_tensor_name("model.param_proj.proj.0.weight"),
            Some("head.mu.weight".into())
        );
    }
}
