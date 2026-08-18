//! Model-agnostic `zsfm convert` / `zsfm inspect` / `zsfm pull`: turn ANY supported
//! checkpoint (safetensors, sharded-safetensors index, PyTorch pickle, npy/npz,
//! ONNX, HDF5/Keras, GGUF) into a GGUF file with tensor names passed through
//! unchanged, list its contents, or just fetch a HuggingFace repo's checkpoint in
//! whatever format it happens to be published in.
//!
//! Unlike the per-model subcommands (which map tensor names to each inference
//! engine's schema and assume `config.json` + `model.safetensors`), this path
//! does no renaming beyond an optional `--strip-prefix` and downloads whatever
//! format a repo actually has (via `zsfm_hub::download_any_format`), so it works
//! for arbitrary models.

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Args, ValueEnum};

use zsfm_checkpoint::{cast, load_checkpoint, LoadOptions};
use zsfm_gguf::{GGMLType, GGUFMetaValue, GGUFWriter};

#[derive(Clone, ValueEnum)]
pub enum DtypeArg { F32, F16, Bf16, Q8 }

impl From<DtypeArg> for GGMLType {
    fn from(d: DtypeArg) -> Self {
        match d {
            DtypeArg::F32  => GGMLType::F32,
            DtypeArg::F16  => GGMLType::F16,
            DtypeArg::Bf16 => GGMLType::BF16,
            DtypeArg::Q8   => GGMLType::Q8_0,
        }
    }
}

#[derive(Args)]
pub struct ConvertArgs {
    /// Input checkpoint file(s): .safetensors (one or more shards),
    /// model.safetensors.index.json, .pt/.pth/.bin/.ckpt, .npy/.npz, .onnx,
    /// .h5/.hdf5, .keras, or .gguf (GGUF input re-quantizes and carries
    /// metadata through). TF SavedModel/checkpoints: see tools/export-weights.py.
    /// Omit if using --repo.
    #[arg(required_unless_present = "repo")]
    pub inputs: Vec<PathBuf>,
    /// Download and convert directly from a HuggingFace repo (owner/name) instead
    /// of local files — whatever checkpoint format it's published in.
    #[arg(long, conflicts_with = "inputs", value_name = "OWNER/NAME")]
    pub repo: Option<String>,
    /// With --repo: which checkpoint format to fetch (safetensors, bin, pt, pth,
    /// ckpt, onnx, h5, hdf5, keras, npz, npy, gguf). Default: the highest-priority
    /// format actually present in the repo.
    #[arg(long, requires = "repo")]
    pub format: Option<String>,
    /// With --repo: case-insensitive substring to pick among multiple files of the
    /// chosen format — e.g. a specific GGUF quant variant, or a safetensors file
    /// when the repo has no sharding index and ships several unrelated ones.
    #[arg(long, requires = "repo")]
    pub file: Option<String>,
    /// With --repo: branch, tag, or commit to fetch from.
    #[arg(long, default_value = "main", requires = "repo")]
    pub revision: String,
    /// With --repo: HuggingFace API token, for gated/private repos.
    #[arg(long, env = "HF_TOKEN", requires = "repo")]
    pub token: Option<String>,
    /// With --repo: directory to download into. Default: models/`<owner>__<name>`.
    #[arg(long, requires = "repo")]
    pub model_dir: Option<PathBuf>,
    /// Output GGUF path.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Output dtype. Tensors too small for Q8_0 blocks fall back to F32.
    #[arg(long, default_value = "f16")]
    pub dtype: DtypeArg,
    /// Value for the `general.architecture` metadata key.
    #[arg(long)]
    pub arch: Option<String>,
    /// Value for the `general.name` metadata key (default: first input's file stem).
    #[arg(long)]
    pub name: Option<String>,
    /// Extra metadata entries, `key=value` (repeatable). Values are parsed as
    /// bool / u32 / i32 / f32 when they look like one, else stored as strings.
    #[arg(long = "metadata", value_name = "KEY=VALUE")]
    pub metadata: Vec<String>,
    /// For pickle inputs: descend into this top-level dict (e.g. `state_dict`).
    #[arg(long)]
    pub pickle_key: Option<String>,
    /// Strip this prefix from tensor names that carry it (e.g. `model.`).
    #[arg(long)]
    pub strip_prefix: Option<String>,
}

#[derive(Args)]
pub struct PullArgs {
    /// HuggingFace repo to download from (owner/name).
    #[arg(value_name = "OWNER/NAME")]
    pub repo: String,
    /// Which checkpoint format to fetch (safetensors, bin, pt, pth, ckpt, onnx,
    /// h5, hdf5, keras, npz, npy, gguf). Default: the highest-priority format
    /// actually present in the repo.
    #[arg(long)]
    pub format: Option<String>,
    /// Case-insensitive substring to pick among multiple files of the chosen
    /// format — e.g. a specific GGUF quant variant.
    #[arg(long)]
    pub file: Option<String>,
    /// Branch, tag, or commit to fetch from.
    #[arg(long, default_value = "main")]
    pub revision: String,
    /// HuggingFace API token, for gated/private repos.
    #[arg(long, env = "HF_TOKEN")]
    pub token: Option<String>,
    /// Directory to download into. Default: models/`<owner>__<name>`.
    #[arg(long)]
    pub model_dir: Option<PathBuf>,
}

#[derive(Args)]
pub struct InspectArgs {
    /// Checkpoint file to inspect (any supported format, including .gguf).
    pub path: PathBuf,
    /// Print the first N values of --tensor (or of every tensor if --tensor is
    /// omitted and N is given).
    #[arg(long, value_name = "N")]
    pub values: Option<usize>,
    /// Restrict --values output to this tensor.
    #[arg(long)]
    pub tensor: Option<String>,
    /// For pickle inputs: descend into this top-level dict (e.g. `state_dict`).
    #[arg(long)]
    pub pickle_key: Option<String>,
}

pub async fn run_convert(args: ConvertArgs) -> anyhow::Result<()> {
    // --repo downloads first, then falls through to the same local-file path
    // below using the downloaded checkpoint files as `inputs`.
    let (inputs, repo_default_name) = if let Some(repo) = &args.repo {
        let model_dir = args
            .model_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("models").join(repo.replace('/', "__")));
        println!("Looking up files in {repo}@{} …", args.revision);
        let downloaded = zsfm_hub::download_any_format(
            repo,
            args.format.as_deref(),
            args.file.as_deref(),
            &args.revision,
            args.token.as_deref(),
            &model_dir,
        )
        .await
        .with_context(|| format!("download {repo}"))?;
        println!(
            "Downloaded {} {} file(s) into {} …",
            downloaded.checkpoint_files.len(),
            downloaded.format,
            model_dir.display()
        );
        let default_name = repo.rsplit('/').next().unwrap_or(repo).to_string();
        (downloaded.checkpoint_files, Some(default_name))
    } else {
        (args.inputs.clone(), None)
    };

    let opts = LoadOptions {
        pickle_key: args.pickle_key.clone(),
        strip_prefix: args.strip_prefix.clone(),
    };
    println!("Loading {} input file(s) …", inputs.len());
    let ckpt = load_checkpoint(&inputs, &opts)?;
    println!("Loaded {} tensors.", ckpt.tensors.len());
    anyhow::ensure!(!ckpt.tensors.is_empty(), "checkpoint contains no tensors");

    let mut writer = GGUFWriter::new();

    // Metadata: carried-over (GGUF inputs) first, then general.* defaults /
    // overrides, then explicit --metadata pairs. Later entries replace earlier
    // ones with the same key.
    let mut meta: Vec<(String, GGUFMetaValue)> = ckpt.metadata;
    let default_name = repo_default_name.unwrap_or_else(|| {
        inputs[0]
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string()
    });
    let set = |meta: &mut Vec<(String, GGUFMetaValue)>, key: &str, value: GGUFMetaValue| {
        meta.retain(|(k, _)| k != key);
        meta.push((key.to_string(), value));
    };
    if args.arch.is_some() || !meta.iter().any(|(k, _)| k == "general.architecture") {
        let arch = args.arch.clone().unwrap_or_else(|| "unknown".to_string());
        set(&mut meta, "general.architecture", GGUFMetaValue::String(arch));
    }
    if args.name.is_some() || !meta.iter().any(|(k, _)| k == "general.name") {
        let name = args.name.clone().unwrap_or(default_name);
        set(&mut meta, "general.name", GGUFMetaValue::String(name));
    }
    for pair in &args.metadata {
        let (key, value) = pair
            .split_once('=')
            .with_context(|| format!("--metadata {pair}: expected key=value"))?;
        set(&mut meta, key, parse_meta_value(value));
    }
    for (key, value) in meta {
        writer.add_metadata(key, value);
    }

    let out_dtype: GGMLType = args.dtype.into();
    let mut fallback_count = 0usize;
    for t in &ckpt.tensors {
        let n_elems: u64 = t.shape.iter().product();
        let innermost = t.shape.last().copied().unwrap_or(1);

        let dst = if out_dtype == GGMLType::Q8_0 && (innermost % 32 != 0 || n_elems % 32 != 0) {
            fallback_count += 1;
            GGMLType::F32
        } else {
            out_dtype
        };
        let data = cast::cast_data(&t.data, t.dtype, dst)
            .with_context(|| format!("tensor {}: cast failed", t.name))?;
        let gguf_shape: Vec<u64> = t.shape.iter().rev().copied().collect();
        writer.add_tensor(t.name.clone(), gguf_shape, dst, data);
    }
    if fallback_count > 0 {
        eprintln!("note: {fallback_count} tensor(s) fell back to F32 (too small for Q8_0 blocks)");
    }

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    println!("Writing {} tensors to {} …", ckpt.tensors.len(), args.output.display());
    let out_file = File::create(&args.output)
        .with_context(|| format!("create {}", args.output.display()))?;
    writer.write_to(&mut BufWriter::new(out_file))?;
    println!("Done.");
    Ok(())
}

/// Download a repo's checkpoint in whatever format it's published in, without
/// converting — for inspecting raw files, or when the target format is one
/// `zsfm convert` doesn't need to touch (already GGUF).
pub async fn run_pull(args: PullArgs) -> anyhow::Result<()> {
    let model_dir = args
        .model_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("models").join(args.repo.replace('/', "__")));
    println!("Looking up files in {}@{} …", args.repo, args.revision);
    let downloaded = zsfm_hub::download_any_format(
        &args.repo,
        args.format.as_deref(),
        args.file.as_deref(),
        &args.revision,
        args.token.as_deref(),
        &model_dir,
    )
    .await
    .with_context(|| format!("download {}", args.repo))?;

    println!(
        "Downloaded {} file(s) in {} format:",
        downloaded.checkpoint_files.len(),
        downloaded.format
    );
    for f in &downloaded.checkpoint_files {
        println!("  {}", f.display());
    }
    if let Some(cfg) = &downloaded.config_json {
        println!("  {} (config.json)", cfg.display());
    }
    println!(
        "\nTo convert: zsfm convert {} -o model.gguf --dtype f16",
        downloaded
            .checkpoint_files
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}

fn parse_meta_value(s: &str) -> GGUFMetaValue {
    match s {
        "true" => return GGUFMetaValue::Bool(true),
        "false" => return GGUFMetaValue::Bool(false),
        _ => {}
    }
    if let Ok(v) = s.parse::<u32>() {
        return GGUFMetaValue::Uint32(v);
    }
    if let Ok(v) = s.parse::<i32>() {
        return GGUFMetaValue::Int32(v);
    }
    if let Ok(v) = s.parse::<f32>() {
        return GGUFMetaValue::Float32(v);
    }
    GGUFMetaValue::String(s.to_string())
}

pub fn run_inspect(args: InspectArgs) -> anyhow::Result<()> {
    let ext = args
        .path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    // GGUF: show stored (possibly quantized) dtypes and metadata directly.
    if ext == "gguf" {
        let mut file = std::fs::File::open(&args.path)
            .with_context(|| format!("open {}", args.path.display()))?;
        // Prefer the in-crate reader (knows BF16); fall back to candle for
        // foreign quantization formats it doesn't cover.
        {
            use std::io::Seek;
            if let Ok(gguf) = zsfm_gguf::GGUFFile::read(&mut file) {
                println!("Metadata in {}:", args.path.display());
                for (key, value) in &gguf.metadata {
                    println!("  {key:48} {value:?}");
                }
                println!("Tensors:");
                let mut infos: Vec<&zsfm_gguf::GGUFTensorInfo> = gguf.tensors.iter().collect();
                infos.sort_by(|a, b| a.name.cmp(&b.name));
                for info in &infos {
                    let py_shape: Vec<u64> = info.shape.iter().rev().copied().collect();
                    println!("  {:64} {:?} {py_shape:?}", info.name, info.dtype);
                }
                if let Some(n) = args.values {
                    for info in infos {
                        if let Some(only) = &args.tensor {
                            if info.name != *only {
                                continue;
                            }
                        }
                        let vals = gguf.tensor_f32(&mut file, info)?;
                        let shown: Vec<String> =
                            vals.iter().take(n).map(|v| v.to_string()).collect();
                        println!("  {} values[..{}]: [{}]", info.name, shown.len(), shown.join(", "));
                    }
                }
                return Ok(());
            }
            file.seek(std::io::SeekFrom::Start(0))?;
        }
        let content =
            candle_core::quantized::gguf_file::Content::read(&mut file).context("read GGUF")?;

        println!("Metadata in {}:", args.path.display());
        let mut keys: Vec<&String> = content.metadata.keys().collect();
        keys.sort();
        for key in keys {
            println!("  {key:48} {:?}", content.metadata[key]);
        }
        println!("Tensors:");
        let mut names: Vec<&String> = content.tensor_infos.keys().collect();
        names.sort();
        for name in &names {
            let ti = &content.tensor_infos[name.as_str()];
            println!("  {name:64} {:?} {:?}", ti.ggml_dtype, ti.shape);
        }
        if let Some(n) = args.values {
            let device = candle_core::Device::Cpu;
            for name in names {
                if let Some(only) = &args.tensor {
                    if *name != *only {
                        continue;
                    }
                }
                let qt = content.tensor(&mut file, name, &device)?;
                let vals: Vec<f32> = qt
                    .dequantize(&device)?
                    .flatten_all()?
                    .to_dtype(candle_core::DType::F32)?
                    .to_vec1()?;
                let shown: Vec<String> = vals.iter().take(n).map(|v| v.to_string()).collect();
                println!("  {name} values[..{}]: [{}]", shown.len(), shown.join(", "));
            }
        }
        return Ok(());
    }

    let opts = LoadOptions { pickle_key: args.pickle_key.clone(), strip_prefix: None };
    let ckpt = load_checkpoint(&[args.path.clone()], &opts)?;
    println!("Tensors in {}:", args.path.display());
    let mut tensors: Vec<_> = ckpt.tensors.iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    for t in &tensors {
        println!("  {:64} {} {:?}", t.name, t.dtype.name(), t.shape);
    }
    if let Some(n) = args.values {
        for t in tensors {
            if let Some(only) = &args.tensor {
                if &t.name != only {
                    continue;
                }
            }
            let vals = cast::decode_to_f32(&t.data, t.dtype)?;
            let shown: Vec<String> = vals.iter().take(n).map(|v| v.to_string()).collect();
            println!("  {} values[..{}]: [{}]", t.name, shown.len(), shown.join(", "));
        }
    }
    Ok(())
}
