# Tabular foundation models

These models do in-context classification or regression: you give them a small labeled **support set** (training rows) and a set of **query rows** to predict, and they run a single zero-shot forward pass — no fine-tuning, no training loop.

Four of the five (Mitra, TabDPT, TabICL, TabPFN-3) share one request format:

```json
{
  "x_support": [[0.1, 1.2], [0.9, -0.3], [-1.1, 0.4]],
  "y_support": [0, 1, 1],
  "x_query": [[0.2, 0.9], [-0.8, 0.1]],
  "n_classes": 2
}
```

`n_classes` is only used for classification (ignored — may be omitted — for regression). Classification responses look like:

```json
{"task": "classification", "probabilities": [[0.83, 0.17], [0.21, 0.79]]}
```

Regression responses:

```json
{"task": "regression", "predictions": [1.53, 0.22]}
```

TabFM uses a different shape — see its own section below.

Each row below is the original model, not a reimplementation — `zsfm convert` downloads the exact published weights and this workspace's inference code is verified bit-exact against the original PyTorch implementation. Links go to the original HuggingFace weights, the original authors' source repo, and the paper.

| Model | HF weights | Original code | License | Class. | Regr. |
|---|---|---|---|---|---|
| [Mitra](#mitra) | [autogluon/mitra-classifier](https://huggingface.co/autogluon/mitra-classifier) / [-regressor](https://huggingface.co/autogluon/mitra-regressor) | [autogluon/autogluon](https://github.com/autogluon/autogluon/tree/master/tabular/src/autogluon/tabular/models/mitra) | Apache-2.0 | ✅ | ✅ |
| [TabDPT](#tabdpt) | [Layer6/TabDPT](https://huggingface.co/Layer6/TabDPT) | [layer6ai-labs/TabDPT-inference](https://github.com/layer6ai-labs/TabDPT-inference) | Apache-2.0 | ✅ | ✅ |
| [TabICL](#tabicl) | [jingang/TabICL](https://huggingface.co/jingang/TabICL) | [soda-inria/tabicl](https://github.com/soda-inria/tabicl) | BSD-3-Clause | ✅ | ❌ |
| [TabPFN-3](#tabpfn-3) | [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3) | [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) | **Non-commercial** | ✅ | ❌ |
| [TabFM](#tabfm) | [google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch) | [google-research/tabfm](https://github.com/google-research/tabfm) | **Non-commercial** | ✅ | ✅ |

Bold licenses restrict use to research/internal evaluation — see [Licensing](../licensing.md) before relying on TabPFN-3 or TabFM for anything else.

## Mitra

Paper: [Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](https://arxiv.org/abs/2510.21204).

```bash
zsfm mitra convert --model autogluon/mitra-classifier --task classification
zsfm mitra convert --model autogluon/mitra-regressor --task regression

zsfm mitra infer --gguf gguf/mitra-classification-f32.gguf --task classification < request.json
zsfm mitra infer --gguf gguf/mitra-regression-f32.gguf --task regression < request.json
```

`--task` on `infer` must match which checkpoint you loaded — the GGUF doesn't self-describe which head it has. Zero-shot only (no fine-tuning path), no random-mirror augmentations (both scope decisions made deliberately to keep the port a single deterministic forward pass).

## TabDPT

Paper: [TabDPT: Scaling Tabular Foundation Models on Real Data](https://huggingface.co/papers/2410.18164).

```bash
zsfm tabdpt convert
zsfm tabdpt infer --gguf gguf/tabdpt-f32.gguf --task classification < request.json
zsfm tabdpt infer --gguf gguf/tabdpt-f32.gguf --task regression < request.json
```

One checkpoint serves both tasks — `--task` on `infer` just picks which output head to read. Single forward pass, no class-permutation ensembling.

## TabICL

Papers: [TabICL](https://arxiv.org/abs/2502.05564), [TabICLv2](https://arxiv.org/abs/2602.11139).

```bash
zsfm convert --repo jingang/TabICL --file classifier-v2 --format ckpt -o gguf/tabicl-v2-f32.gguf
zsfm tabicl infer --gguf gguf/tabicl-v2-f32.gguf < request.json
```

Classification only, `n_classes` must be ≤ 10 (the >10-class mixed-radix/hierarchical path from the original model isn't implemented). No dedicated `convert` subcommand — see [the generic converter](../cli-overview.md#the-generic-converter). The `jingang/TabICL` repo publishes 4 classifier checkpoints; `classifier-v2` picks the one this port was verified against.

## TabPFN-3

Paper: [TabPFN-3: Technical Report](https://arxiv.org/abs/2605.13986).

```bash
zsfm convert --repo Prior-Labs/tabpfn_3 --file classifier-v3_default --format ckpt -o gguf/tabpfn-v3-f32.gguf
zsfm tabpfn infer --gguf gguf/tabpfn-v3-f32.gguf < request.json
```

Classification only (the regression bar-distribution head isn't implemented). No dedicated `convert` subcommand. The repo publishes several checkpoint variants; `classifier-v3_default` is the one this port was verified against.

> Licensed under `tabpfn-3-license-v1.0` — **non-commercial**: research, testing, and internal benchmarking are explicitly fine, but the model, its derivatives, and its outputs can't be used for any commercial or production purpose. See [Licensing](../licensing.md).

## TabFM

Source: [Google Research blog post](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/).

TabFM's request shape is a single combined table rather than separate support/query arrays:

```bash
zsfm tabfm convert   # classification by default; --task regression for the other variant

cat > request.json <<'EOF'
{
  "x": [[0.1, 1.2], [0.9, -0.3], [-1.1, 0.4], [0.2, 0.9]],
  "y": [0, 1, 1, 0],
  "train_size": 3
}
EOF
zsfm tabfm infer --gguf gguf/tabfm-classification-f16.gguf < request.json
```

`x` is `[rows][columns]`, `y` is one label per row (any finite placeholder value at query-row positions is fine — it's ignored), `train_size` is how many leading rows are the training set. Optional fields: `cat_mask` (marks categorical columns, default all-false) and `d` (actual unpadded feature count, default = number of columns).

TabFM also has a second, heavier command, `ensemble-predict`, that reproduces the full sklearn-wrapper pipeline (feature scaling, categorical encoding, `n_estimators`-member ensembling, calibration) bit-compatible with the original `TabFMClassifier`/`TabFMRegressor`'s default RNG — see `zsfm tabfm ensemble-predict --help` for its (considerably larger) request shape.

> Licensed under the **TabFM Non-Commercial License v1.0** — testing, evaluation, and internal benchmarking are fine; any commercial or production use (including client deliverables or revenue-generating decisions) requires a separate license from Google. See [Licensing](../licensing.md).
