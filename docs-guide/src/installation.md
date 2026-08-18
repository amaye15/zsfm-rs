# Installation

## Option 1: download a release binary

Pre-built binaries are published on the [GitHub Releases page](https://github.com/amaye15/zero-shot-forecasters-gguf/releases) for:

- Linux (`x86_64-unknown-linux-gnu`)
- macOS Apple Silicon (`aarch64-apple-darwin`)

> macOS Intel (`x86_64-apple-darwin`) isn't currently published — GitHub's free Intel Mac runner capacity has been cut to the point of being unusable for CI. Build from source instead; the workspace itself has no problem running on Intel Macs.

Download the tarball for your platform, extract it, and put `zsfm` on your `PATH`:

```bash
tar xzf zsfm-v0.1.0-aarch64-apple-darwin.tar.gz
sudo mv zsfm-v0.1.0-aarch64-apple-darwin/zsfm /usr/local/bin/
zsfm --help
```

## Option 2: build from source

Requires a recent stable Rust toolchain ([rustup.rs](https://rustup.rs)).

```bash
git clone https://github.com/amaye15/zero-shot-forecasters-gguf.git
cd zero-shot-forecasters-gguf/zsfm-rs
cargo build --release -p zsfm-cli
./target/release/zsfm --help
```

On macOS, the release profile links against Apple's Accelerate framework for faster BLAS operations automatically — no extra setup needed. On Linux/Windows it falls back to candle's portable default backend.

## Verifying it works

```bash
zsfm --help          # top-level: one subcommand per model
zsfm chronos --help  # per-model: convert / infer / upload / inspect-tensors
```

If both print usage text without errors, you're ready for the [Quick start](./quickstart.md).
