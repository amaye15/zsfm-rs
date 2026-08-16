//! Shared HTTP client construction for the download-side of this crate
//! (`download.rs`, `discover.rs`). `upload.rs` builds its own client since its
//! auth requirements differ (token is mandatory, not optional).

use anyhow::Context;

pub(crate) const HF_BASE: &str = "https://huggingface.co";

pub(crate) fn build_client(hf_token: Option<&str>) -> anyhow::Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::USER_AGENT,
        "zsfm-hub/0.1".parse().unwrap(),
    );
    if let Some(token) = hf_token {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {token}").parse().context("invalid HF token")?,
        );
    }
    Ok(reqwest::Client::builder()
        .default_headers(headers)
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?)
}
