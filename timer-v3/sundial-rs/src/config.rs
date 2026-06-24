use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SundialConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub input_token_len: usize,
    #[serde(default = "default_output_token_lens")]
    pub output_token_lens: Vec<usize>,
    pub rope_theta: f64,
    #[serde(default = "default_flow_depth")]
    pub flow_loss_depth: usize,
    #[serde(default = "default_sampling_steps")]
    pub num_sampling_steps: usize,
}

fn default_output_token_lens() -> Vec<usize> { vec![720] }
fn default_flow_depth() -> usize { 3 }
fn default_sampling_steps() -> usize { 50 }

impl SundialConfig {
    pub fn head_dim(&self) -> usize { self.hidden_size / self.num_attention_heads }
    pub fn output_token_len(&self) -> usize { self.output_token_lens[0] }
}

impl Default for SundialConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            input_token_len: 16,
            output_token_lens: vec![720],
            rope_theta: 10000.0,
            flow_loss_depth: 3,
            num_sampling_steps: 50,
        }
    }
}
