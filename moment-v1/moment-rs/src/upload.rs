use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::Context;
use base64::Engine as _;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use tokio_util::io::ReaderStream;

const HF_BASE: &str = "https://huggingface.co";
const PREUPLOAD_SAMPLE: usize = 512;

const SKIP_DIRS: &[&str] = &["target", ".git", "__pycache__", ".venv", "models"];

// HF manages these files automatically — never delete them
const KEEP_ALWAYS: &[&str] = &[".gitattributes", ".gitignore"];

struct RegularFile {
    remote: String,
    content_b64: String,
}

struct LfsFile {
    local: PathBuf,
    remote: String,
    size: u64,
    oid: String, // sha256 hex
}

pub async fn run(repo_id: &str, token: &str, root: &Path) -> anyhow::Result<()> {
    let client = build_client(token)?;

    ensure_repo(&client, repo_id).await?;

    // List existing repo files so we can delete stale ones in the same commit
    let existing_files = list_repo_files(&client, repo_id).await?;
    if !existing_files.is_empty() {
        println!("  {} file(s) currently in repo", existing_files.len());
    }

    let files = gather_files(root)?;
    println!("Gathered {} file(s) to upload …", files.len());

    let (regular, lfs) = classify_files(&client, repo_id, &files).await?;
    println!("  {} regular, {} LFS", regular.len(), lfs.len());

    if !lfs.is_empty() {
        upload_lfs(&client, repo_id, &lfs).await?;
    }

    // Delete anything currently in the repo that we're not re-uploading
    let upload_paths: std::collections::HashSet<&str> = regular
        .iter()
        .map(|f| f.remote.as_str())
        .chain(lfs.iter().map(|f| f.remote.as_str()))
        .collect();
    let to_delete: Vec<String> = existing_files
        .into_iter()
        .filter(|p| !upload_paths.contains(p.as_str()) && !KEEP_ALWAYS.contains(&p.as_str()))
        .collect();
    if !to_delete.is_empty() {
        println!("  Deleting {} stale file(s)", to_delete.len());
    }

    make_commit(&client, repo_id, &regular, &lfs, &to_delete).await?;

    println!("Uploaded → https://huggingface.co/{repo_id}");
    Ok(())
}

fn build_client(token: &str) -> anyhow::Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        format!("Bearer {token}").parse().context("invalid HF token")?,
    );
    headers.insert(
        reqwest::header::USER_AGENT,
        "moment-rs/0.1".parse().unwrap(),
    );
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

async fn ensure_repo(client: &reqwest::Client, repo_id: &str) -> anyhow::Result<()> {
    let name = repo_id.split('/').nth(1).context("repo_id must be owner/name")?;
    let resp: reqwest::Response = client
        .post(format!("{HF_BASE}/api/repos/create"))
        .json(&serde_json::json!({ "name": name, "type": "model", "private": false }))
        .send()
        .await
        .context("create repo")?;

    let status = resp.status();
    if status.is_success() {
        println!("Created repo {repo_id}");
    } else if status.as_u16() == 409 {
        // already exists — fine
    } else {
        let body: String = resp.text().await.unwrap_or_default();
        anyhow::bail!("create repo: HTTP {status}: {body}");
    }
    Ok(())
}

/// List all file (blob) paths currently in the repo, following Link header pagination.
async fn list_repo_files(client: &reqwest::Client, repo_id: &str) -> anyhow::Result<Vec<String>> {
    let mut files = Vec::new();
    let mut url = format!("{HF_BASE}/api/models/{repo_id}/tree/main?recursive=true&limit=1000");

    loop {
        let resp: reqwest::Response = client.get(&url).send().await.context("list repo tree")?;
        let status = resp.status();
        if status.as_u16() == 404 {
            return Ok(files); // repo is new / empty
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("list repo tree: HTTP {status}: {body}");
        }

        // Extract next-page URL from Link header before consuming body
        let next_url = resp
            .headers()
            .get("link")
            .and_then(|v| v.to_str().ok())
            .and_then(parse_next_link);

        #[derive(serde::Deserialize)]
        struct TreeEntry { r#type: String, path: String }
        let entries: Vec<TreeEntry> = resp.json().await.context("parse tree response")?;
        for e in entries {
            if e.r#type == "file" {
                files.push(e.path);
            }
        }

        match next_url {
            Some(next) => url = next,
            None => break,
        }
    }

    Ok(files)
}

/// Parse `<url>; rel="next"` from a Link header value.
fn parse_next_link(header: &str) -> Option<String> {
    for part in header.split(',') {
        let part = part.trim();
        if part.contains(r#"rel="next""#) {
            if let Some(url_part) = part.split(';').next() {
                let url = url_part.trim().trim_start_matches('<').trim_end_matches('>');
                return Some(url.to_string());
            }
        }
    }
    None
}

/// Ask HF which files need LFS vs regular inline upload, then prepare both lists.
async fn classify_files(
    client: &reqwest::Client,
    repo_id: &str,
    files: &[(PathBuf, String)],
) -> anyhow::Result<(Vec<RegularFile>, Vec<LfsFile>)> {
    let mut preupload_entries: Vec<serde_json::Value> = Vec::new();
    for (local, remote) in files {
        let size = std::fs::metadata(local)
            .with_context(|| format!("stat {}", local.display()))?
            .len();
        let sample = {
            let bytes = std::fs::read(local)
                .with_context(|| format!("read {}", local.display()))?;
            let n = bytes.len().min(PREUPLOAD_SAMPLE);
            base64::engine::general_purpose::STANDARD.encode(&bytes[..n])
        };
        preupload_entries.push(serde_json::json!({
            "path": remote,
            "size": size,
            "sample": sample,
        }));
    }

    let url = format!("{HF_BASE}/api/models/{repo_id}/preupload/main");
    let resp: reqwest::Response = client
        .post(&url)
        .json(&serde_json::json!({ "files": preupload_entries }))
        .send()
        .await
        .context("preupload request")?;

    let status = resp.status();
    if !status.is_success() {
        let body: String = resp.text().await.unwrap_or_default();
        anyhow::bail!("preupload: HTTP {status}: {body}");
    }

    #[derive(serde::Deserialize)]
    struct PreuploadFile {
        path: String,
        #[serde(rename = "uploadMode")]
        upload_mode: String,
        #[serde(rename = "shouldIgnore", default)]
        _should_ignore: bool,
    }
    #[derive(serde::Deserialize)]
    struct PreuploadResp { files: Vec<PreuploadFile> }

    let preupload: PreuploadResp = resp.json().await.context("parse preupload response")?;
    let modes: std::collections::HashMap<String, String> = preupload.files
        .into_iter()
        .map(|f| (f.path, f.upload_mode))
        .collect();

    let mut regular: Vec<RegularFile> = Vec::new();
    let mut lfs: Vec<LfsFile> = Vec::new();

    for (local, remote) in files {
        let mode = modes.get(remote).map(|s| s.as_str()).unwrap_or("regular");
        if mode == "lfs" {
            let size = std::fs::metadata(local)?.len();
            let oid = sha256_file(local)?;
            lfs.push(LfsFile { local: local.clone(), remote: remote.clone(), size, oid });
        } else {
            let bytes = std::fs::read(local)
                .with_context(|| format!("read {}", local.display()))?;
            let content_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
            regular.push(RegularFile { remote: remote.clone(), content_b64 });
        }
    }

    Ok((regular, lfs))
}

fn gather_files(root: &Path) -> anyhow::Result<Vec<(PathBuf, String)>> {
    let mut out = Vec::new();
    walk_dir(root, "", &mut out)?;

    // Promote the README.md from the first-level crate subdirectory to the repo root
    // so it renders on the HF repo page (e.g. "toto-rs/README.md" → "README.md").
    let has_root_readme = out.iter().any(|(_, r)| r == "README.md");
    if !has_root_readme {
        for (_, remote) in &mut out {
            let parts: Vec<&str> = remote.splitn(3, '/').collect();
            if parts.len() == 2 && parts[1] == "README.md" {
                *remote = "README.md".to_string();
                break;
            }
        }
    }

    Ok(out)
}

fn walk_dir(dir: &Path, prefix: &str, out: &mut Vec<(PathBuf, String)>) -> anyhow::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("non-UTF-8 filename"))?;
        if SKIP_DIRS.contains(&name.as_str()) {
            continue;
        }
        let remote = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
        if path.is_file() {
            out.push((path, remote));
        } else if path.is_dir() {
            walk_dir(&path, &remote, out)?;
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut f =
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

async fn upload_lfs(
    client: &reqwest::Client,
    repo_id: &str,
    files: &[LfsFile],
) -> anyhow::Result<()> {
    let objects: Vec<_> = files
        .iter()
        .map(|f| serde_json::json!({ "oid": f.oid, "size": f.size }))
        .collect();

    // Request both multipart (for files >5 GB) and basic transfers
    let url = format!("{HF_BASE}/{repo_id}.git/info/lfs/objects/batch");
    let lfs_body = serde_json::to_string(&serde_json::json!({
        "operation": "upload",
        "transfers": ["multipart", "basic"],
        "objects": objects,
    }))?;
    let resp: reqwest::Response = client
        .post(&url)
        .header("Content-Type", "application/vnd.git-lfs+json")
        .header("Accept", "application/vnd.git-lfs+json")
        .body(lfs_body)
        .send()
        .await
        .context("LFS batch request")?;

    let status = resp.status();
    if !status.is_success() {
        let body: String = resp.text().await.unwrap_or_default();
        anyhow::bail!("LFS batch: HTTP {status}: {body}");
    }

    #[derive(serde::Deserialize)]
    struct BatchResp { objects: Vec<serde_json::Value> }
    let batch: BatchResp = resp.json().await.context("parse LFS batch response")?;

    for (file, obj) in files.iter().zip(batch.objects.iter()) {
        let Some(upload_href) =
            obj.pointer("/actions/upload/href").and_then(|v: &serde_json::Value| v.as_str())
        else {
            println!("  (already on LFS) {}", file.remote);
            continue;
        };

        // Multipart if the server provided chunk_size in the header
        let is_multipart = obj.pointer("/actions/upload/header/chunk_size").is_some();

        if is_multipart {
            upload_lfs_object_multipart(file, upload_href, obj).await
                .with_context(|| format!("upload (multipart) {}", file.remote))?;
        } else {
            upload_lfs_object(file, upload_href, obj).await
                .with_context(|| format!("upload {}", file.remote))?;
        }

        // Verify step (optional but recommended by Git LFS spec)
        if let Some(verify_href) =
            obj.pointer("/actions/verify/href").and_then(|v: &serde_json::Value| v.as_str())
        {
            let verify_headers = obj
                .pointer("/actions/verify/header")
                .and_then(|v: &serde_json::Value| v.as_object())
                .cloned()
                .unwrap_or_default();

            let mut vreq: reqwest::RequestBuilder = reqwest::Client::new()
                .post(verify_href)
                .header("Content-Type", "application/vnd.git-lfs+json")
                .json(&serde_json::json!({ "oid": file.oid, "size": file.size }));
            for (k, v) in &verify_headers {
                if let Some(val) = v.as_str() {
                    vreq = vreq.header(k.as_str(), val);
                }
            }
            let vresp: reqwest::Response = vreq.send().await.context("LFS verify")?;
            if !vresp.status().is_success() {
                eprintln!("  Warning: LFS verify returned {}", vresp.status());
            }
        }
    }

    Ok(())
}

async fn upload_lfs_object(
    file: &LfsFile,
    href: &str,
    obj: &serde_json::Value,
) -> anyhow::Result<()> {
    let pb = ProgressBar::new(file.size);
    pb.set_style(
        ProgressStyle::with_template(
            "  {msg} [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(file.remote.clone());

    let f = tokio::fs::File::open(&file.local)
        .await
        .with_context(|| format!("open {}", file.local.display()))?;

    let pb2 = pb.clone();
    let stream = ReaderStream::new(f).map(move |chunk| {
        if let Ok(ref b) = chunk {
            pb2.inc(b.len() as u64);
        }
        chunk
    });

    // LFS upload goes to S3 / Azure — use a plain client (no HF auth header)
    let mut req = reqwest::Client::new()
        .put(href)
        .header("Content-Length", file.size.to_string());

    if let Some(extra) = obj.pointer("/actions/upload/header").and_then(|v| v.as_object()) {
        for (k, v) in extra {
            if let Some(val) = v.as_str() {
                req = req.header(k.as_str(), val);
            }
        }
    }

    let resp = req
        .body(reqwest::Body::wrap_stream(stream))
        .send()
        .await
        .context("PUT LFS object")?;

    pb.finish_and_clear();

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("LFS PUT HTTP {status}: {body}");
    }

    Ok(())
}

async fn upload_lfs_object_multipart(
    file: &LfsFile,
    complete_href: &str,
    obj: &serde_json::Value,
) -> anyhow::Result<()> {
    use tokio::io::AsyncReadExt;

    let header = obj
        .pointer("/actions/upload/header")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow::anyhow!("no header in multipart LFS response"))?;

    let chunk_size: usize = header
        .get("chunk_size")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow::anyhow!("missing chunk_size in multipart header"))?;

    // Collect part URLs sorted numerically by key ("00001", "00002", …)
    let mut parts: Vec<(u32, String)> = header
        .iter()
        .filter_map(|(k, v)| {
            let n: u32 = k.parse().ok()?;
            Some((n, v.as_str()?.to_string()))
        })
        .collect();
    parts.sort_by_key(|(n, _)| *n);

    let pb = ProgressBar::new(file.size);
    pb.set_style(
        ProgressStyle::with_template(
            "  {msg} [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(file.remote.clone());

    let mut f = tokio::fs::File::open(&file.local)
        .await
        .with_context(|| format!("open {}", file.local.display()))?;

    let s3 = reqwest::Client::new(); // plain client — S3 parts use pre-signed URLs
    let mut etags: Vec<(u32, String)> = Vec::with_capacity(parts.len());

    for (part_num, url) in &parts {
        // Read up to chunk_size bytes for this part
        let mut buf = vec![0u8; chunk_size];
        let mut pos = 0;
        while pos < chunk_size {
            let n = f.read(&mut buf[pos..]).await?;
            if n == 0 { break; }
            pos += n;
        }
        if pos == 0 { break; }
        buf.truncate(pos);
        let len = buf.len();

        // Retry up to 3 times on transient connection errors
        const MAX_RETRIES: usize = 3;
        let mut last_err: Option<anyhow::Error> = None;
        let mut etag_opt: Option<String> = None;
        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                eprintln!("  Retrying part {part_num} (attempt {})…", attempt + 1);
            }
            match s3
                .put(url.as_str())
                .header("Content-Length", len.to_string())
                .body(buf.clone())
                .send()
                .await
            {
                Err(e) => {
                    last_err = Some(anyhow::anyhow!("PUT part {part_num}: {e}"));
                }
                Ok(resp) => {
                    let status = resp.status();
                    let tag = resp.headers().get("etag")
                        .and_then(|v| v.to_str().ok())
                        .map(|s| s.to_string());
                    if status.is_success() {
                        if let Some(t) = tag {
                            etag_opt = Some(t);
                            last_err = None;
                            break;
                        } else {
                            last_err = Some(anyhow::anyhow!("PUT part {part_num}: no ETag in response"));
                        }
                    } else {
                        let body = resp.text().await.unwrap_or_default();
                        last_err = Some(anyhow::anyhow!("PUT part {part_num}: HTTP {status}: {body}"));
                    }
                }
            }
        }
        let etag = etag_opt.ok_or_else(|| {
            last_err.unwrap_or_else(|| anyhow::anyhow!("PUT part {part_num}: exhausted retries"))
        })?;

        pb.inc(len as u64);
        etags.push((*part_num, etag));
    }

    pb.finish_and_clear();

    // Tell HF to assemble the parts on S3
    let parts_json: Vec<serde_json::Value> = etags
        .iter()
        .map(|(n, e)| serde_json::json!({ "partNumber": n, "etag": e }))
        .collect();

    let resp = s3
        .post(complete_href)
        .json(&serde_json::json!({ "oid": file.oid, "parts": parts_json }))
        .send()
        .await
        .context("complete multipart")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("complete multipart: HTTP {status}: {body}");
    }

    Ok(())
}

async fn make_commit(
    client: &reqwest::Client,
    repo_id: &str,
    regular: &[RegularFile],
    lfs: &[LfsFile],
    to_delete: &[String],
) -> anyhow::Result<()> {
    let mut lines: Vec<String> = Vec::new();

    lines.push(serde_json::to_string(&serde_json::json!({
        "key": "header",
        "value": { "summary": "Upload model files", "description": "" },
    }))?);

    for rf in regular {
        lines.push(serde_json::to_string(&serde_json::json!({
            "key": "file",
            "value": {
                "path": rf.remote,
                "encoding": "base64",
                "content": rf.content_b64,
            },
        }))?);
    }

    for lf in lfs {
        lines.push(serde_json::to_string(&serde_json::json!({
            "key": "lfsFile",
            "value": {
                "path": lf.remote,
                "algo": "sha256",
                "oid": lf.oid,
                "size": lf.size,
            },
        }))?);
    }

    for path in to_delete {
        lines.push(serde_json::to_string(&serde_json::json!({
            "key": "deletedEntry",
            "value": { "path": path },
        }))?);
    }

    let body = lines.join("\n");

    let url = format!("{HF_BASE}/api/models/{repo_id}/commit/main");
    let resp = client
        .post(&url)
        .header("Content-Type", "application/x-ndjson")
        .body(body)
        .send()
        .await
        .context("POST commit")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("commit: HTTP {status}: {body}");
    }

    Ok(())
}
