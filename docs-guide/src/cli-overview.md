# The `zsfm` CLI

Every model gets its own subcommand: `zsfm <model> <action>`. Most models support four actions:

| Action | What it does |
|---|---|
| `convert` | Download the model from HuggingFace and produce a GGUF file |
| `infer` | Load a GGUF file and run zero-shot inference on a JSON request from stdin |
| `upload` | Push the source + GGUF files to a HuggingFace repo you control |
| `inspect-tensors` | Print every tensor name/shape/dtype in a local checkpoint file |

Run `zsfm <model> --help` or `zsfm <model> convert --help` for the full flag list of any specific model. Every `convert` command takes `--model-dir` (default `models/`, resolved relative to wherever you run `zsfm` from) for where downloaded weights and the canonical F32 GGUF cache are stored.

## Model caching

The first time you convert a model, `zsfm` does three things:

1. Downloads the raw weights from HuggingFace.
2. Converts them once to a **canonical F32 GGUF**, cached at `<model_dir>/<owner>__<repo-name>/model-f32.gguf` (default `model_dir` is `models/`).
3. Deletes the downloaded raw weight file(s) — they're fully redundant once the canonical GGUF exists. (`config.json`, where a model's `infer` command reads it separately, is kept — it's tiny.)

Every **later** `convert` for that same model — at any `--dtype` — recasts straight from that cached F32 GGUF instead of touching the network again:

```bash
zsfm chronos convert --dtype f32   # downloads, ~10s
zsfm chronos convert --dtype f16   # recasts from cache, <1s, no network
zsfm chronos convert --dtype q8    # recasts from cache, <1s, no network
```

Pass `--redownload` to force a fresh download anyway (e.g. the upstream repo was updated):

```bash
zsfm chronos convert --redownload
```

This applies to every model **except** TabICL and TabPFN-3, which don't have their own `convert` subcommand — see below.

## Output dtype

Most `convert` commands accept `--dtype f32|f16|q8`. `q8` (Q8_0 block quantization) roughly quarters file size; tensors too small for a 32-element block automatically fall back to F32 for that tensor rather than erroring. BF16 is available through the generic converter (next section) for interop with other GGUF consumers, but not through the per-model `--dtype` flag — candle's GGUF reader in this workspace can't load ggml dtype 30 (BF16) back in, so a BF16-converted file couldn't be used with this same crate's own `infer`.

## The generic converter

TabICL and TabPFN-3 ship a raw checkpoint whose tensor names can be used unchanged (no HuggingFace-specific renaming needed), so they're converted through a separate, model-agnostic command instead of a dedicated subcommand:

```bash
zsfm convert --repo jingang/TabICL --file classifier-v2 --format ckpt -o gguf/tabicl-v2-f32.gguf
zsfm convert --repo Prior-Labs/tabpfn_3 --file classifier-v3_default --format ckpt -o gguf/tabpfn-v3-f32.gguf
```

`--file` is a case-insensitive substring used to disambiguate when a repo publishes multiple checkpoints of the same format (both of the repos above do — see [Tabular foundation models](./models/tabular.md) for the exact substrings to use).

This same generic command works for **any** HuggingFace repo, not just the two above — `zsfm convert --repo <owner>/<name>` downloads whatever checkpoint format is published (safetensors, PyTorch pickle, ONNX, HDF5/Keras, npz/npy, or an existing GGUF) and converts it with tensor names passed through unchanged. It's also how you'd re-quantize an existing GGUF file: `zsfm convert existing.gguf -o smaller.gguf --dtype q8`. It does *not* get the F32-caching behavior described above, since it's meant to work with arbitrary repos, not just this workspace's known model list.

## Request/response format

`infer` always reads a single JSON object from stdin and writes a single JSON object to stdout. The shape depends on whether the model is a time-series forecaster or a tabular model — see the next two chapters for exact examples.
