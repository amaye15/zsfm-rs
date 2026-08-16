/// Map a TinyTimeMixer HuggingFace tensor name to the GGUF naming convention.
/// Returns `None` for unrecognised names (caller will warn and skip them).
pub fn map_tensor_name(hf_name: &str) -> Option<String> {
    // Patcher (Linear patch_length → d_model)
    match hf_name {
        "backbone.encoder.patcher.weight" => return Some("enc.patcher.weight".into()),
        "backbone.encoder.patcher.bias"   => return Some("enc.patcher.bias".into()),
        "decoder.adapter.weight"          => return Some("dec.adapter.weight".into()),
        "decoder.adapter.bias"            => return Some("dec.adapter.bias".into()),
        "head.base_forecast_block.weight" => return Some("head.weight".into()),
        "head.base_forecast_block.bias"   => return Some("head.bias".into()),
        _ => {}
    }

    // Encoder adaptive patching mixers:
    // backbone.encoder.mlp_mixer_encoder.mixers.{L}.mixer_layers.{N}.<part>
    if let Some(rest) = hf_name.strip_prefix("backbone.encoder.mlp_mixer_encoder.mixers.") {
        return map_mixer_layer(rest, "enc.blk");
    }

    // Decoder block mixers:
    // decoder.decoder_block.mixers.{N}.<part>
    if let Some(rest) = hf_name.strip_prefix("decoder.decoder_block.mixers.") {
        return map_decoder_mixer_layer(rest);
    }

    None
}

/// Map `{L}.mixer_layers.{N}.<part>` → `{prefix}.{L}.layer.{N}.<gguf_suffix>`
fn map_mixer_layer(rest: &str, prefix: &str) -> Option<String> {
    let (level_str, rest) = rest.split_once('.')?;
    let level: u32 = level_str.parse().ok()?;

    let rest = rest.strip_prefix("mixer_layers.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let layer: u32 = layer_str.parse().ok()?;

    let suffix = map_mixer_suffix(rest)?;
    Some(format!("{prefix}.{level}.layer.{layer}.{suffix}"))
}

/// Map `{N}.<part>` → `dec.blk.{N}.<gguf_suffix>`
fn map_decoder_mixer_layer(rest: &str) -> Option<String> {
    let (n_str, rest) = rest.split_once('.')?;
    let n: u32 = n_str.parse().ok()?;
    let suffix = map_mixer_suffix(rest)?;
    Some(format!("dec.blk.{n}.{suffix}"))
}

fn map_mixer_suffix(suffix: &str) -> Option<&'static str> {
    Some(match suffix {
        "patch_mixer.norm.norm.weight"            => "patch_norm.weight",
        "patch_mixer.norm.norm.bias"              => "patch_norm.bias",
        "patch_mixer.mlp.fc1.weight"              => "patch_fc1.weight",
        "patch_mixer.mlp.fc1.bias"                => "patch_fc1.bias",
        "patch_mixer.mlp.fc2.weight"              => "patch_fc2.weight",
        "patch_mixer.mlp.fc2.bias"                => "patch_fc2.bias",
        "patch_mixer.gating_block.attn_layer.weight" => "patch_gate.weight",
        "patch_mixer.gating_block.attn_layer.bias"   => "patch_gate.bias",
        "feature_mixer.norm.norm.weight"          => "feat_norm.weight",
        "feature_mixer.norm.norm.bias"            => "feat_norm.bias",
        "feature_mixer.mlp.fc1.weight"            => "feat_fc1.weight",
        "feature_mixer.mlp.fc1.bias"              => "feat_fc1.bias",
        "feature_mixer.mlp.fc2.weight"            => "feat_fc2.weight",
        "feature_mixer.mlp.fc2.bias"              => "feat_fc2.bias",
        "feature_mixer.gating_block.attn_layer.weight" => "feat_gate.weight",
        "feature_mixer.gating_block.attn_layer.bias"   => "feat_gate.bias",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patcher() {
        assert_eq!(map_tensor_name("backbone.encoder.patcher.weight"), Some("enc.patcher.weight".into()));
        assert_eq!(map_tensor_name("backbone.encoder.patcher.bias"), Some("enc.patcher.bias".into()));
    }

    #[test]
    fn enc_mixer() {
        assert_eq!(
            map_tensor_name("backbone.encoder.mlp_mixer_encoder.mixers.0.mixer_layers.1.patch_mixer.mlp.fc1.weight"),
            Some("enc.blk.0.layer.1.patch_fc1.weight".into())
        );
        assert_eq!(
            map_tensor_name("backbone.encoder.mlp_mixer_encoder.mixers.2.mixer_layers.0.feature_mixer.gating_block.attn_layer.bias"),
            Some("enc.blk.2.layer.0.feat_gate.bias".into())
        );
    }

    #[test]
    fn dec_mixer() {
        assert_eq!(
            map_tensor_name("decoder.decoder_block.mixers.1.patch_mixer.norm.norm.weight"),
            Some("dec.blk.1.patch_norm.weight".into())
        );
    }

    #[test]
    fn head() {
        assert_eq!(map_tensor_name("head.base_forecast_block.weight"), Some("head.weight".into()));
    }

    #[test]
    fn unknown() {
        assert_eq!(map_tensor_name("something.unknown"), None);
    }
}
