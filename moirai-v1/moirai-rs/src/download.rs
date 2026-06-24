use std::path::{Path, PathBuf};

use anyhow::Context;
use futures_util::StreamExt;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

const HF_BASE: &str = "https://huggingface.co";

pub struct ModelFiles {
    pub safetensors_shards: Vec<PathBuf>,
}

pub async fn download_model(
    repo_id: &str,
    hf_token: Option<&str>,
    model_dir: &Path,
) -> anyhow::Result<ModelFiles> {
    let client = build_client(hf_token)?;
    std::fs::create_dir_all(model_dir).context("create model dir")?;

    let shards =
        match fetch_file(&client, repo_id, "model.safetensors.index.json", model_dir).await {
            Ok(index_path) => {
                println!("Found sharded model — reading index …");
                resolve_shards(&client, repo_id, &index_path, model_dir).await?
            }
            Err(_) => {
                println!("Fetching model.safetensors …");
                let shard =
                    fetch_file(&client, repo_id, "model.safetensors", model_dir).await?;
                vec![shard]
            }
        };

    Ok(ModelFiles { safetensors_shards: shards })
}

fn build_client(hf_token: Option<&str>) -> anyhow::Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::USER_AGENT,
        "moirai-rs/0.1".parse().unwrap(),
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

async fn fetch_file(
    client: &reqwest::Client,
    repo_id: &str,
    filename: &str,
    dest_dir: &Path,
) -> anyhow::Result<PathBuf> {
    let dest = dest_dir.join(filename.replace('/', "_"));
    if dest.exists() {
        println!("  (cached) {filename}");
        return Ok(dest);
    }

    let dest_tmp = dest.with_extension("tmp");
    let already = if dest_tmp.exists() {
        dest_tmp.metadata()?.len()
    } else {
        0
    };

    let url = format!("{HF_BASE}/{repo_id}/resolve/main/{filename}");

    let mut req = client.get(&url);
    if already > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={already}-"));
        println!("  Resuming {filename} from {} MB …", already / 1_000_000);
    }

    let response = req.send().await.with_context(|| format!("GET {url}"))?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("HTTP {status} fetching {filename} from {repo_id}");
    }

    let (file, resume_offset) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        let f = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&dest_tmp)
            .await
            .with_context(|| format!("open tmp {}", dest_tmp.display()))?;
        (f, already)
    } else {
        let f = tokio::fs::File::create(&dest_tmp)
            .await
            .with_context(|| format!("create tmp {}", dest_tmp.display()))?;
        (f, 0)
    };

    let total = response.content_length().map(|n| n + resume_offset).unwrap_or(0);

    let pb = indicatif::ProgressBar::new(total);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "  {msg} [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(filename.to_string());
    pb.set_position(resume_offset);

    {
        let mut file = file;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| format!("stream chunk of {filename}"))?;
            pb.inc(chunk.len() as u64);
            file.write_all(&chunk)
                .await
                .with_context(|| format!("write chunk to {}", dest_tmp.display()))?;
        }
    }

    pb.finish_and_clear();
    std::fs::rename(&dest_tmp, &dest)
        .with_context(|| format!("rename tmp → {}", dest.display()))?;

    Ok(dest)
}

async fn resolve_shards(
    client: &reqwest::Client,
    repo_id: &str,
    index_path: &Path,
    cache_dir: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    #[derive(Deserialize)]
    struct Index {
        weight_map: std::collections::HashMap<String, String>,
    }

    let raw = std::fs::read_to_string(index_path).context("read index json")?;
    let index: Index = serde_json::from_str(&raw).context("parse index json")?;

    let mut shard_names: Vec<String> = index.weight_map.into_values().collect();
    shard_names.sort();
    shard_names.dedup();

    let mut paths = Vec::with_capacity(shard_names.len());
    for name in &shard_names {
        println!("  Fetching {name} …");
        let p = fetch_file(client, repo_id, name, cache_dir).await?;
        paths.push(p);
    }
    Ok(paths)
}
