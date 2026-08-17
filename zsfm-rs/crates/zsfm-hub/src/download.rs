use std::path::{Path, PathBuf};

use anyhow::Context;
use futures_util::{stream, StreamExt};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

use crate::http::{build_client, HF_BASE};

/// How many shard files to fetch concurrently. HF's CDN comfortably serves this many
/// parallel range/GET requests per client; higher offers diminishing returns and risks
/// the server throttling or the local connection pool thrashing.
pub(crate) const SHARD_DOWNLOAD_CONCURRENCY: usize = 4;

/// Paths to all local files needed for conversion.
pub struct ModelFiles {
    pub config_json: PathBuf,
    /// Ordered list of safetensors shard paths, already downloaded locally.
    pub safetensors_shards: Vec<PathBuf>,
}

/// Download (or locate from cache) `config.json` + `model.safetensors[.index.json]`
/// for `repo_id` into `model_dir`.
pub async fn download_model(
    repo_id: &str,
    hf_token: Option<&str>,
    model_dir: &Path,
) -> anyhow::Result<ModelFiles> {
    download_model_prefixed(repo_id, "", hf_token, model_dir).await
}

/// Download (or locate from cache) a single arbitrary file from `repo_id`, resuming a partial
/// download if one exists. For models that don't fit the `config.json` + `model.safetensors`
/// shape — e.g. Lag-Llama's raw PyTorch Lightning `.ckpt` checkpoint.
pub async fn download_file(
    repo_id: &str,
    relpath: &str,
    hf_token: Option<&str>,
    dest_dir: &Path,
) -> anyhow::Result<PathBuf> {
    let client = build_client(hf_token)?;
    std::fs::create_dir_all(dest_dir).context("create dest dir")?;
    fetch_file(&client, repo_id, relpath, dest_dir, None).await
}

/// Same as [`download_model`], but every remote path is joined under `prefix` first.
///
/// Pass `""` for a flat repo layout (`config.json`, `model.safetensors`).
/// Pass e.g. `"classification"` for a repo that keeps multiple task variants
/// in subfolders (`classification/config.json`, `classification/model.safetensors`),
/// as TabFM does.
pub async fn download_model_prefixed(
    repo_id: &str,
    prefix: &str,
    hf_token: Option<&str>,
    model_dir: &Path,
) -> anyhow::Result<ModelFiles> {
    let client = build_client(hf_token)?;
    // Namespace by repo (matching the `owner__name` convention the generic `zsfm convert
    // --repo` path already uses) so two different repos sharing the generic `config.json` +
    // `model.safetensors` naming — which is most of them — never collide when a caller passes
    // the same (often default) `model_dir` for both, whether run sequentially or concurrently.
    // Without this, the second download's `config.json`/`model.safetensors` would either
    // silently overwrite the first's files, or worse, get "(cached)" hit on the *wrong* model's
    // bytes.
    let model_dir = model_dir.join(repo_id.replace('/', "__"));
    let model_dir = model_dir.as_path();
    std::fs::create_dir_all(model_dir).context("create model dir")?;

    let config_rel = joined(prefix, "config.json");
    println!("Fetching {config_rel} …");
    let config_json = fetch_file(&client, repo_id, &config_rel, model_dir, None).await?;

    // Detect sharded model by fetching the index file.
    let index_rel = joined(prefix, "model.safetensors.index.json");
    let single_rel = joined(prefix, "model.safetensors");

    let shards = match fetch_file(&client, repo_id, &index_rel, model_dir, None).await {
        Ok(index_path) => {
            println!("Found sharded model — reading index …");
            resolve_shards(&client, repo_id, &index_path, model_dir, prefix).await?
        }
        Err(_) => {
            println!("Fetching {single_rel} …");
            let shard = fetch_file(&client, repo_id, &single_rel, model_dir, None).await?;
            vec![shard]
        }
    };

    Ok(ModelFiles {
        config_json,
        safetensors_shards: shards,
    })
}

fn joined(prefix: &str, filename: &str) -> String {
    if prefix.is_empty() {
        filename.to_string()
    } else {
        format!("{prefix}/{filename}")
    }
}

/// Download `relpath` (may include subfolders, e.g. `"classification/config.json"`)
/// from `repo_id` into `dest_dir`, resuming if a partial `.tmp` file already exists.
/// Returns the local path of the completed file.
///
/// `mp`: when several `fetch_file` calls run concurrently (see `resolve_shards`), pass a
/// shared [`indicatif::MultiProgress`] so their bars stack cleanly instead of each fighting
/// to redraw the same terminal line; `None` draws a lone standalone bar.
pub(crate) async fn fetch_file(
    client: &reqwest::Client,
    repo_id: &str,
    relpath: &str,
    dest_dir: &Path,
    mp: Option<&indicatif::MultiProgress>,
) -> anyhow::Result<PathBuf> {
    let dest = dest_dir.join(relpath.replace('/', "_"));
    if dest.exists() {
        println!("  (cached) {relpath}");
        return Ok(dest);
    }

    let dest_tmp = dest.with_extension("tmp");
    let already = if dest_tmp.exists() {
        dest_tmp.metadata()?.len()
    } else {
        0
    };

    let url = format!("{HF_BASE}/{repo_id}/resolve/main/{relpath}");

    let mut req = client.get(&url);
    if already > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={already}-"));
        println!("  Resuming {relpath} from {} MB …", already / 1_000_000);
    }

    let response = req
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;

    let status = response.status();
    // 206 = partial content (resume accepted), 200 = full content
    if !status.is_success() {
        anyhow::bail!("HTTP {status} fetching {relpath} from {repo_id}");
    }

    // If server ignored the Range header and sent 200, truncate the tmp file.
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

    let total = response
        .content_length()
        .map(|n| n + resume_offset)
        .unwrap_or(0);

    let pb = indicatif::ProgressBar::new(total);
    let pb = match mp {
        Some(mp) => mp.add(pb),
        None => pb,
    };
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "  {msg} [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(relpath.to_string());
    pb.set_position(resume_offset);

    {
        let mut file = file;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| format!("stream chunk of {relpath}"))?;
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

/// Parse the shard index JSON and download every unique shard (joined under `prefix`).
async fn resolve_shards(
    client: &reqwest::Client,
    repo_id: &str,
    index_path: &Path,
    cache_dir: &Path,
    prefix: &str,
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

    // Fetch up to SHARD_DOWNLOAD_CONCURRENCY shards at once — each is an independent file,
    // so there's no reason to serialize what's fundamentally a bandwidth-bound operation.
    // `buffer_unordered` completes them in whatever order the network delivers them; each
    // result is tagged with its original index so the returned `Vec<PathBuf>` still matches
    // `shard_names`' sorted order (some downstream converters iterate shards positionally).
    let multi = indicatif::MultiProgress::new();
    let mut results: Vec<(usize, PathBuf)> = stream::iter(shard_names.iter().enumerate())
        .map(|(i, name)| {
            let rel = joined(prefix, name);
            let multi = &multi;
            async move {
                println!("  Fetching {rel} …");
                fetch_file(client, repo_id, &rel, cache_dir, Some(multi))
                    .await
                    .map(|p| (i, p))
            }
        })
        .buffer_unordered(SHARD_DOWNLOAD_CONCURRENCY)
        .collect::<Vec<anyhow::Result<(usize, PathBuf)>>>()
        .await
        .into_iter()
        .collect::<anyhow::Result<Vec<_>>>()?;
    results.sort_by_key(|(i, _)| *i);
    Ok(results.into_iter().map(|(_, p)| p).collect())
}
