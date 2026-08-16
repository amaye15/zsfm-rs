//! Format-agnostic HuggingFace Hub download: list whatever files a repo
//! actually has, pick a checkpoint format (auto or requested), and fetch it —
//! without assuming `config.json` + `model.safetensors` like [`crate::download_model`].
//!
//! Feeds straight into `zsfm-checkpoint::load_checkpoint`, which reads every
//! format this module can select.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use futures_util::{stream, StreamExt};
use serde::Deserialize;

use crate::download::{fetch_file, SHARD_DOWNLOAD_CONCURRENCY};
use crate::http::{build_client, HF_BASE};

/// Checkpoint file extensions this tool knows how to read, in the order
/// preferred when a repo ships more than one format and the caller doesn't
/// request a specific one.
pub const FORMAT_PRIORITY: &[&str] = &[
    "safetensors", "bin", "pt", "pth", "ckpt", "onnx", "h5", "hdf5", "keras", "npz", "npy", "gguf",
];

pub struct DownloadedRepo {
    pub dir: PathBuf,
    /// Extension of the format that was selected (e.g. "safetensors").
    pub format: String,
    /// Local paths of every checkpoint file downloaded, ready to pass to
    /// `zsfm-checkpoint::load_checkpoint`.
    pub checkpoint_files: Vec<PathBuf>,
    /// `config.json`, if the repo has one (best-effort, not required).
    pub config_json: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

#[derive(Deserialize)]
struct RepoInfo {
    siblings: Vec<Sibling>,
}

/// List every file path in a repo at `revision` (default branch: `"main"`).
pub async fn list_repo_files(
    repo_id: &str,
    revision: &str,
    hf_token: Option<&str>,
) -> anyhow::Result<Vec<String>> {
    let client = build_client(hf_token)?;
    let url = if revision == "main" {
        format!("{HF_BASE}/api/models/{repo_id}")
    } else {
        format!("{HF_BASE}/api/models/{repo_id}/revision/{revision}")
    };
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!(
            "HTTP {status} listing files for {repo_id}@{revision} — check the repo id, \
             revision, and (for gated/private repos) that --token / HF_TOKEN is set"
        );
    }
    let info: RepoInfo = resp.json().await.context("parse repo file listing")?;
    Ok(info.siblings.into_iter().map(|s| s.rfilename).collect())
}

fn ext_of(path: &str) -> Option<String> {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
}

/// Download a repo's checkpoint in whatever format it's stored in (or a
/// specific one, if `format` is given). Resolves HF's standard sharding
/// indices for `safetensors` and `bin`; for every other format, if more than
/// one file matches, `file_filter` (a case-insensitive substring) is required
/// to pick which ones — this avoids silently pulling every alternative
/// quantization/variant of a multi-GB checkpoint.
pub async fn download_any_format(
    repo_id: &str,
    format: Option<&str>,
    file_filter: Option<&str>,
    revision: &str,
    hf_token: Option<&str>,
    dest_dir: &Path,
) -> anyhow::Result<DownloadedRepo> {
    let client = build_client(hf_token)?;
    std::fs::create_dir_all(dest_dir).context("create dest dir")?;

    let files = list_repo_files(repo_id, revision, hf_token).await?;
    anyhow::ensure!(
        !files.is_empty(),
        "repo {repo_id}@{revision} has no files (wrong repo id/revision, or empty repo)"
    );

    let chosen_format = match format {
        Some(f) => {
            let f = f.trim_start_matches('.').to_ascii_lowercase();
            anyhow::ensure!(
                files.iter().any(|p| ext_of(p).as_deref() == Some(f.as_str())),
                "repo {repo_id} has no .{f} files. Found extensions: {}",
                describe_extensions(&files)
            );
            f
        }
        None => FORMAT_PRIORITY
            .iter()
            .find(|ext| files.iter().any(|p| ext_of(p).as_deref() == Some(**ext)))
            .map(|s| s.to_string())
            .with_context(|| {
                format!(
                    "repo {repo_id} has no recognized checkpoint format. Found extensions: {}",
                    describe_extensions(&files)
                )
            })?,
    };

    let mut matches: Vec<&String> = files
        .iter()
        .filter(|p| ext_of(p).as_deref() == Some(chosen_format.as_str()))
        .collect();
    if let Some(filter) = file_filter {
        let filter_lower = filter.to_ascii_lowercase();
        matches.retain(|p| p.to_ascii_lowercase().contains(&filter_lower));
        anyhow::ensure!(
            !matches.is_empty(),
            "no .{chosen_format} file in {repo_id} matches --file {filter}"
        );
    }

    let checkpoint_files = if chosen_format == "safetensors"
        && files.iter().any(|f| f == "model.safetensors.index.json")
    {
        download_sharded(&client, repo_id, "model.safetensors.index.json", dest_dir).await?
    } else if (chosen_format == "bin" || chosen_format == "pt")
        && files.iter().any(|f| f == "pytorch_model.bin.index.json")
    {
        download_sharded(&client, repo_id, "pytorch_model.bin.index.json", dest_dir).await?
    } else if matches.len() == 1 {
        vec![fetch_file(&client, repo_id, matches[0], dest_dir, None).await?]
    } else {
        anyhow::ensure!(
            file_filter.is_some(),
            "repo {repo_id} has {} .{chosen_format} files and none is HF-indexed sharding — \
             pass --file <substring> to pick which one(s) to download. Candidates:\n  {}",
            matches.len(),
            matches
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n  ")
        );
        let multi = indicatif::MultiProgress::new();
        let mut results: Vec<(usize, PathBuf)> = stream::iter(matches.iter().enumerate())
            .map(|(i, m)| {
                let multi = &multi;
                let client = client.clone();
                async move {
                    fetch_file(&client, repo_id, m, dest_dir, Some(multi))
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
        results.into_iter().map(|(_, p)| p).collect()
    };

    let config_json = if files.iter().any(|f| f == "config.json") {
        fetch_file(&client, repo_id, "config.json", dest_dir, None).await.ok()
    } else {
        None
    };

    Ok(DownloadedRepo {
        dir: dest_dir.to_path_buf(),
        format: chosen_format,
        checkpoint_files,
        config_json,
    })
}

/// Download an HF sharding index (`*.index.json`) and every shard it names.
async fn download_sharded(
    client: &reqwest::Client,
    repo_id: &str,
    index_relpath: &str,
    dest_dir: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    let index_path = fetch_file(client, repo_id, index_relpath, dest_dir, None).await?;
    #[derive(Deserialize)]
    struct Index {
        weight_map: HashMap<String, String>,
    }
    let raw = std::fs::read_to_string(&index_path).context("read sharding index")?;
    let index: Index = serde_json::from_str(&raw).context("parse sharding index")?;

    let mut shard_names: Vec<String> = index.weight_map.into_values().collect();
    shard_names.sort();
    shard_names.dedup();
    anyhow::ensure!(!shard_names.is_empty(), "{index_relpath} lists no shards");

    // See `download::resolve_shards` — same bounded-concurrency rationale.
    let multi = indicatif::MultiProgress::new();
    let mut results: Vec<(usize, PathBuf)> = stream::iter(shard_names.iter().enumerate())
        .map(|(i, name)| {
            let multi = &multi;
            async move {
                fetch_file(client, repo_id, name, dest_dir, Some(multi))
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

fn describe_extensions(files: &[String]) -> String {
    let mut exts: Vec<String> = files.iter().filter_map(|f| ext_of(f)).collect();
    exts.sort();
    exts.dedup();
    if exts.is_empty() {
        "(no file extensions found)".to_string()
    } else {
        exts.join(", ")
    }
}
