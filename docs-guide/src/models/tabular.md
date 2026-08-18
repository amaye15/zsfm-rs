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

| Model | HF repo | Classification | Regression | Notes |
|---|---|---|---|---|
| [Mitra](#mitra) | `autogluon/mitra-classifier` / `-regressor` | ✅ | ✅ | |
| [TabDPT](#tabdpt) | `Layer6/TabDPT` | ✅ | ✅ | one checkpoint, both tasks |
| [TabICL](#tabicl) | `jingang/TabICL` | ✅ | ❌ | `n_classes` ≤ 10, generic converter |
| [TabPFN-3](#tabpfn-3) | `Prior-Labs/tabpfn_3` | ✅ | ❌ | **non-commercial weights**, generic converter |
| [TabFM](#tabfm) | `google/tabfm-1.0.0-pytorch` | ✅ | ✅ | different request shape, full ensemble pipeline available |

## Mitra

```bash
zsfm mitra convert --model autogluon/mitra-classifier --task classification
zsfm mitra convert --model autogluon/mitra-regressor --task regression

zsfm mitra infer --gguf gguf/mitra-classification-f32.gguf --task classification < request.json
zsfm mitra infer --gguf gguf/mitra-regression-f32.gguf --task regression < request.json
```

`--task` on `infer` must match which checkpoint you loaded — the GGUF doesn't self-describe which head it has. Zero-shot only (no fine-tuning path), no random-mirror augmentations (both scope decisions made deliberately to keep the port a single deterministic forward pass).

## TabDPT

```bash
zsfm tabdpt convert
zsfm tabdpt infer --gguf gguf/tabdpt-f32.gguf --task classification < request.json
zsfm tabdpt infer --gguf gguf/tabdpt-f32.gguf --task regression < request.json
```

One checkpoint serves both tasks — `--task` on `infer` just picks which output head to read. Single forward pass, no class-permutation ensembling.

## TabICL

```bash
zsfm convert --repo jingang/TabICL --file classifier-v2 --format ckpt -o gguf/tabicl-v2-f32.gguf
zsfm tabicl infer --gguf gguf/tabicl-v2-f32.gguf < request.json
```

Classification only, `n_classes` must be ≤ 10 (the >10-class mixed-radix/hierarchical path from the original model isn't implemented). No dedicated `convert` subcommand — see [the generic converter](../cli-overview.md#the-generic-converter). The `jingang/TabICL` repo publishes 4 classifier checkpoints; `classifier-v2` picks the one this port was verified against.

## TabPFN-3

```bash
zsfm convert --repo Prior-Labs/tabpfn_3 --file classifier-v3_default --format ckpt -o gguf/tabpfn-v3-f32.gguf
zsfm tabpfn infer --gguf gguf/tabpfn-v3-f32.gguf < request.json
```

Classification only (the regression bar-distribution head isn't implemented). No dedicated `convert` subcommand. The repo publishes several checkpoint variants; `classifier-v3_default` is the one this port was verified against.

> **The weights are non-commercial.** See [Licensing](../licensing.md) before using this one for anything beyond research or internal evaluation.

## TabFM

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
