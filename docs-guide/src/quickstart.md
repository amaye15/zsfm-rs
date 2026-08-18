# Quick start

This walks through converting a small forecasting model and running one forecast, end to end. It uses Moirai-2.0-R-small since it's one of the smaller downloads (~45MB converted).

## 1. Convert

```bash
zsfm moirai2 convert
```

This downloads `Salesforce/moirai-2.0-R-small` from HuggingFace, converts it to a GGUF file at `gguf/moirai2-f32.gguf`, and caches a canonical F32 copy under `models/Salesforce__moirai-2.0-R-small/model-f32.gguf` — see [The `zsfm` CLI](./cli-overview.md#model-caching) for what that cache buys you on future conversions.

## 2. Build a request

Every time-series model reads a JSON request from stdin with a numeric `context` array and a `horizon` (how many steps to forecast). Moirai batches multiple series:

```bash
cat > request.json <<'EOF'
{
  "context": [[10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8, 12.0, 12.2]],
  "horizon": 4
}
EOF
```

## 3. Infer

```bash
zsfm moirai2 infer --gguf gguf/moirai2-f32.gguf < request.json
```

You'll get back an OpenAI-compatible forecast object:

```json
{
  "id": "forecast-...",
  "object": "forecast",
  "model": "moirai-2",
  "choices": [
    {
      "index": 0,
      "forecast": {
        "point": [12.35, 12.51, 12.64, 12.72]
      }
    }
  ]
}
```

## Trying a tabular model instead

Tabular models take a small labeled support set plus rows to predict, rather than a time series. Mitra is a good first one to try:

```bash
zsfm mitra convert --model autogluon/mitra-classifier --task classification

cat > request.json <<'EOF'
{
  "x_support": [[0.1, 1.2], [0.9, -0.3], [-1.1, 0.4], [1.5, 1.1]],
  "y_support": [0, 1, 1, 0],
  "x_query": [[0.2, 0.9], [-0.8, 0.1]],
  "n_classes": 2
}
EOF

zsfm mitra infer --gguf gguf/mitra-classification-f32.gguf < request.json
```

```json
{
  "task": "classification",
  "logits": [[...]],
  "probabilities": [[0.83, 0.17], [0.21, 0.79]]
}
```

For every model's exact request/response shape, scope, and any caveats, see [Time-series forecasters](./models/time-series.md) and [Tabular foundation models](./models/tabular.md).
