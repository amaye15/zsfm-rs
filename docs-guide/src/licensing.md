# Licensing

## This project

The code in this repository (converters, GGUF writer/reader, CLI, inference kernels) is MIT licensed — see [LICENSE](https://github.com/amaye15/zsfm-rs/blob/main/LICENSE) in the repo root.

## Model weights

Converting a model with `zsfm <model> convert` downloads the original weights from HuggingFace and re-encodes them as GGUF — it does not change who owns them or what license applies. Each model keeps the license its original authors published it under, and licenses vary quite a bit across this workspace's 16 models. Check the table below before using a GGUF for anything beyond local experimentation.

| Model | License | Commercial use |
|---|---|---|
| Toto-2, Chronos-2, TimesFM 2.5, Sundial, TTM, Lag-Llama, Mitra, TabDPT | Apache-2.0 | ✅ unrestricted |
| MOMENT | MIT | ✅ unrestricted |
| TabICL | BSD-3-Clause | ✅ unrestricted |
| FlowState-R1 | Apache-2.0 | ✅ unrestricted |
| TiRex | [NXAI Community License](https://huggingface.co/NX-AI/TiRex/blob/main/LICENSE) | ✅ unless your org's annual revenue exceeds €100M *and* you ship TiRex in a commercial product/service (then a separate commercial license from NXAI is required) |
| Moirai 1.0, Moirai 2.0 | [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) | ❌ non-commercial only |
| TabPFN-3 | [tabpfn-3-license-v1.0](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE) | ❌ non-commercial only |
| TabFM | [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE) | ❌ non-commercial only |

### The non-commercial models

Three models — **Moirai (both versions), TabPFN-3, and TabFM** — are licensed for research, testing, and internal evaluation only. None of them permit production deployment, revenue-generating use, or offering the model (or its outputs) as part of a paid product or service. If you need any of those for a commercial use case, the license text for each (linked above) explains how to request a commercial license from the original publisher — `zsfm` has no involvement in that process.

### TiRex's revenue threshold

TiRex's NXAI Community License is modeled on Meta's Llama community license: free to use, modify, and redistribute — including commercially — for everyone except organizations whose consolidated annual revenue exceeds €100M and who are incorporating TiRex into a commercial product or service, who need to request a license from NXAI directly.

### Everything else

The remaining 11 models ship under standard permissive open-source licenses (Apache-2.0, MIT, or BSD-3-Clause) with no commercial-use restriction.

If you're unsure whether your use case qualifies under any of these, read the actual license file in the model's HuggingFace repo (linked from each model's page in [Time-series forecasters](./models/time-series.md) / [Tabular foundation models](./models/tabular.md)) — `zsfm` doesn't attempt to interpret or enforce license terms for you.
