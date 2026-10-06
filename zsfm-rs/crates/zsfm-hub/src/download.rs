use std::path::{Path, PathBuf};

use anyhow::Context;
use futures_util::{stream, StreamExt};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

use crate::http::{build_client, HF_BASE};
use crate::log::{log_status, HubError};

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

impl ModelFiles {
    /// Delete the (potentially large) weight shards once they've been converted into a
    /// GGUF, leaving `config_json` in place — several `infer` commands read it directly
    /// (architecture parameters not embedded in the GGUF), and it's tiny, so there's no
    /// disk-space reason to remove it. Best-effort: a failed delete is not fatal, since
    /// the GGUF conversion has already succeeded by the time this is called.
    pub fn cleanup_weights(&self) {
        for f in &self.safetensors_shards {
            if let Err(e) = std::fs::remove_file(f) {
                eprintln!("warning: could not remove {}: {e}", f.display());
            }
        }
    }
}

/// Where the canonical, always-F32 GGUF for `repo_id` lives once converted once. Every
/// per-model `convert` command checks this path first: if present, it recasts straight
/// from this cached GGUF to whatever dtype was requested instead of re-downloading and
/// re-converting from HuggingFace.
pub fn canonical_gguf_path(model_dir: &Path, repo_id: &str) -> PathBuf {
    model_dir
        .join(repo_id.replace('/', "__"))
        .join("model-f32.gguf")
}

/// Variant-aware cache path for multi-task repos (Mitra, TabFM):
/// `<model_dir>/<variant>-<task>/<owner>__<name>/model-f32.gguf`.
/// Centralizes the `format!("mitra-{task}")` logic duplicated in CLI + Python.
pub fn variant_gguf_path(
    model_dir: &Path,
    variant_prefix: &str,
    task: &str,
    repo_id: &str,
) -> PathBuf {
    canonical_gguf_path(&model_dir.join(format!("{variant_prefix}-{task}")), repo_id)
}

/// Shared cache-hit path for CLI + Python: recast from canonical F32 GGUF when
/// present. Returns `true` when the caller can return early.
pub fn try_recast_from_cache(
    canonical: &Path,
    output: &Path,
    dtype: zsfm_gguf::GGMLType,
    redownload: bool,
) -> anyhow::Result<bool> {
    if canonical.exists() && !redownload {
        log_status(&format!(
            "Using cached F32 GGUF at {} …",
            canonical.display()
        ));
        zsfm_checkpoint::recast(canonical, output, dtype)?;
        log_status(&format!("Wrote {}", output.display()));
        return Ok(true);
    }
    Ok(false)
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
        // Validate legacy cache hits against the `.size` sidecar written on
        // first download; a mismatch means corruption → re-download.
        let sidecar = dest.with_extension("size");
        if let Ok(expect) = std::fs::read_to_string(&sidecar) {
            if let Ok(expect) = expect.trim().parse::<u64>() {
                if let Ok(actual) = std::fs::metadata(&dest).map(|m| m.len()) {
                    if actual != expect {
                        eprintln!(
                            "  (cached {relpath} size mismatch: got {actual}, want {expect} → re-downloading)"
                        );
                        std::fs::remove_file(&dest)?;
                    } else {
                        log_status(&format!("  (cached) {relpath}"));
                        return Ok(dest);
                    }
                }
            }
        } else {
            log_status(&format!("  (cached) {relpath}"));
            return Ok(dest);
        }
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
        log_status(&format!(
            "  Resuming {relpath} from {} MB …",
            already / 1_000_000
        ));
    }

    let response = req.send().await.with_context(|| format!("GET {url}"))?;

    let status = response.status();
    // 206 = partial content (resume accepted), 200 = full content
    if !status.is_success() {
        anyhow::bail!("HTTP {status} fetching {relpath} from {repo_id}");
    }

    // Verify the server honored the Range offset when it claims 206.
    if status == reqwest::StatusCode::PARTIAL_CONTENT && already > 0 {
        if let Some(range) = response.headers().get(reqwest::header::CONTENT_RANGE) {
            let range_str = range.to_str().unwrap_or("");
            // Expected form: `bytes <start>-<end>/<total>`.
            if let Some(start) = range_str
                .strip_prefix("bytes ")
                .and_then(|s| s.split('-').next())
                .and_then(|s| s.parse::<u64>().ok())
            {
                if start != already {
                    return Err(HubError::ResumeMismatch {
                        relpath: relpath.to_string(),
                        local: already,
                        server: start,
                    }
                    .into());
                }
            }
        }
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

    let content_len = response.content_length();
    let total = content_len.map(|n| n + resume_offset).unwrap_or(0);

    let pb = indicatif::ProgressBar::new(total);
    let pb = match mp {
        Some(mp) => mp.add(pb),
        None => pb,
    };
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "  {msg} [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
        )
        .expect("valid progress template")
        .progress_chars("=>-"),
    );
    pb.set_message(relpath.to_string());
    pb.set_position(resume_offset);

    {
        let mut file = file;
        let mut stream = response.bytes_stream();
        let mut written: u64 = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| format!("stream chunk of {relpath}"))?;
            pb.inc(chunk.len() as u64);
            written += chunk.len() as u64;
            file.write_all(&chunk)
                .await
                .with_context(|| format!("write chunk to {}", dest_tmp.display()))?;
        }
        // When the server reports a length, the byte count must match.
        if let Some(len) = content_len {
            if written != len {
                return Err(HubError::SizeMismatch {
                    relpath: relpath.to_string(),
                    expected: len,
                    actual: written,
                }
                .into());
            }
        }
    }

    pb.finish_and_clear();
    std::fs::rename(&dest_tmp, &dest)
        .with_context(|| format!("rename tmp → {}", dest.display()))?;
    // Record final size for future cache validation + log SHA256 so users can
    // compare against the Hub file page when debugging corruption.
    if let Ok(meta) = std::fs::metadata(&dest) {
        let _ = std::fs::write(dest.with_extension("size"), meta.len().to_string());
        if let Ok(sha) = sha256_file(&dest) {
            log_status(&format!("  sha256({relpath}) = {sha}"));
        }
    }

    Ok(dest)
}

/// Streaming SHA256 without loading multi-GB files into memory.
fn sha256_file(path: &Path) -> anyhow::Result<String> {
    use sha2::Digest;
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = sha2::Sha256::new();
    let mut buf = [0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
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
    anyhow::ensure!(
        raw.len() <= 16 << 20,
        "shard index too large: {} bytes (max 16 MiB)",
        raw.len()
    );
    let index: Index = serde_json::from_str(&raw).context("parse index json")?;

    let mut shard_names: Vec<String> = index.weight_map.into_values().collect();
    shard_names.sort();
    shard_names.dedup();
    anyhow::ensure!(
        shard_names.len() <= 512,
        "too many shards: {} (max 512)",
        shard_names.len()
    );
    // Flattened filenames must stay unique — `a/b.safetensors` and `a_b.safetensors`
    // would otherwise overwrite each other in the cache dir.
    {
        let mut flat: Vec<String> = shard_names
            .iter()
            .map(|n| joined(prefix, n).replace('/', "_"))
            .collect();
        flat.sort();
        for w in flat.windows(2) {
            if w[0] == w[1] {
                return Err(HubError::ShardCollision(w[0].clone()).into());
            }
        }
    }

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
                log_status(&format!("  Fetching {rel} …"));
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
