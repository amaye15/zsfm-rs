/// Map Mitra (Tab2D) safetensors tensor names to GGUF naming convention.
pub fn map_tensor_name(name: &str) -> Option<String> {
    match name {
        "final_layer.weight" => return Some("head.weight".into()),
        "final_layer.bias" => return Some("head.bias".into()),
        "final_layer_norm.weight" => return Some("norm_f.weight".into()),
        "final_layer_norm.bias" => return Some("norm_f.bias".into()),
        "x_embedding.x_embedding.weight" => return Some("x_embed.weight".into()),
        "x_embedding.x_embedding.bias" => return Some("x_embed.bias".into()),
        // Classifier: y_embedding.y_embedding is an nn.Embedding (weight only).
        // Regressor: y_embedding.y_embedding is an nn.Linear (weight + bias).
        "y_embedding.y_embedding.weight" => return Some("y_embed.weight".into()),
        "y_embedding.y_embedding.bias" => return Some("y_embed.bias".into()),
        "y_embedding.y_mask.weight" => return Some("y_mask.weight".into()),
        _ => {}
    }

    if let Some(rest) = name.strip_prefix("layers.") {
        let (n_str, rest) = rest.split_once('.')?;
        let n: u32 = n_str.parse().ok()?;

        let suffix = match rest {
            "layer_norm1.weight" => "ln1.weight",
            "layer_norm1.bias" => "ln1.bias",
            "layer_norm2.weight" => "ln2.weight",
            "layer_norm2.bias" => "ln2.bias",
            "layer_norm3.weight" => "ln3.weight",
            "layer_norm3.bias" => "ln3.bias",
            "layer_norm4.weight" => "ln4.weight",
            "layer_norm4.bias" => "ln4.bias",
            "attention1.q.weight" => "attn_row.q.weight",
            "attention1.q.bias" => "attn_row.q.bias",
            "attention1.k.weight" => "attn_row.k.weight",
            "attention1.k.bias" => "attn_row.k.bias",
            "attention1.v.weight" => "attn_row.v.weight",
            "attention1.v.bias" => "attn_row.v.bias",
            "attention1.o.weight" => "attn_row.o.weight",
            "attention1.o.bias" => "attn_row.o.bias",
            "attention2.q.weight" => "attn_feat.q.weight",
            "attention2.q.bias" => "attn_feat.q.bias",
            "attention2.k.weight" => "attn_feat.k.weight",
            "attention2.k.bias" => "attn_feat.k.bias",
            "attention2.v.weight" => "attn_feat.v.weight",
            "attention2.v.bias" => "attn_feat.v.bias",
            "attention2.o.weight" => "attn_feat.o.weight",
            "attention2.o.bias" => "attn_feat.o.bias",
            "linear1.weight" => "mlp1_fc1.weight",
            "linear1.bias" => "mlp1_fc1.bias",
            "linear2.weight" => "mlp1_fc2.weight",
            "linear2.bias" => "mlp1_fc2.bias",
            "linear3.weight" => "mlp2_fc1.weight",
            "linear3.bias" => "mlp2_fc1.bias",
            "linear4.weight" => "mlp2_fc2.weight",
            "linear4.bias" => "mlp2_fc2.bias",
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
    fn top_level() {
        assert_eq!(map_tensor_name("final_layer.weight"), Some("head.weight".into()));
        assert_eq!(map_tensor_name("x_embedding.x_embedding.bias"), Some("x_embed.bias".into()));
        assert_eq!(map_tensor_name("y_embedding.y_mask.weight"), Some("y_mask.weight".into()));
    }

    #[test]
    fn block_attn_and_mlp() {
        assert_eq!(
            map_tensor_name("layers.0.attention1.q.weight"),
            Some("blk.0.attn_row.q.weight".into())
        );
        assert_eq!(
            map_tensor_name("layers.11.attention2.o.bias"),
            Some("blk.11.attn_feat.o.bias".into())
        );
        assert_eq!(map_tensor_name("layers.3.linear4.weight"), Some("blk.3.mlp2_fc2.weight".into()));
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(map_tensor_name("optimizer.state"), None);
    }
}
