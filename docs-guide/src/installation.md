# Installation

## Option 1: `cargo install` (like `cargo install ripgrep`)

From [crates.io](https://crates.io/crates/zsfm) (requires a recent stable Rust toolchain from [rustup.rs](https://rustup.rs)):

```bash
cargo install zsfm --locked
zsfm --help
```

From git (latest `main`):

```bash
cargo install --git https://github.com/amaye15/zsfm-rs zsfm --locked
# or pin a tag:
cargo install --git https://github.com/amaye15/zsfm-rs --tag v0.1.0 zsfm --locked
```

From a local checkout:

```bash
git clone https://github.com/amaye15/zsfm-rs.git
cargo install --path zsfm-rs/crates/zsfm --locked
# or, build without installing:
cargo build --release -p zsfm --manifest-path zsfm-rs/Cargo.toml
./zsfm-rs/target/release/zsfm --help
```

> `--locked` uses the `Cargo.lock` tested in CI for reproducible builds. Omit it if you want the latest compatible dependencies.

## Option 2: download a release binary

Pre-built binaries are published on the [GitHub Releases page](https://github.com/amaye15/zsfm-rs/releases) for:

- Linux (`x86_64-unknown-linux-gnu`)
- macOS Apple Silicon (`aarch64-apple-darwin`)

> macOS Intel (`x86_64-apple-darwin`) isn't currently published — GitHub's free Intel Mac runner capacity has been cut to the point of being unusable for CI. Use `cargo install` above instead; the workspace builds fine on Intel Macs.

Download the tarball for your platform, extract it, and put `zsfm` on your `PATH`:

```bash
tar xzf zsfm-v0.1.0-aarch64-apple-darwin.tar.gz
sudo mv zsfm-v0.1.0-aarch64-apple-darwin/zsfm /usr/local/bin/
zsfm --help
```

## Option 3: build from source

Requires a recent stable Rust toolchain ([rustup.rs](https://rustup.rs)).

```bash
git clone https://github.com/amaye15/zsfm-rs.git
cd zsfm-rs
cargo build --release -p zsfm
./target/release/zsfm --help
# or via the inner workspace:
cargo build --release -p zsfm --manifest-path zsfm-rs/Cargo.toml
./zsfm-rs/target/release/zsfm --help
```

On macOS, the release profile links against Apple's Accelerate framework for faster BLAS operations automatically — no extra setup needed. On Linux/Windows it falls back to candle's portable default backend.

## Verifying it works

```bash
zsfm --help          # top-level: one subcommand per model
zsfm chronos --help  # per-model: convert / infer / upload / inspect-tensors / delete
zsfm chronos delete --help  # per-model cache deletion
```

If both print usage text without errors, you're ready for the [Quick start](./quickstart.md).
