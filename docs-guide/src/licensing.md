# Licensing

## This project

The code in this repository (converters, GGUF writer/reader, CLI, inference kernels) is MIT licensed — see [LICENSE](https://github.com/amaye15/zero-shot-forecasters-gguf/blob/main/LICENSE) in the repo root.

## Model weights

Converting a model with `zsfm <model> convert` downloads the original weights from HuggingFace and re-encodes them as GGUF — it does not change who owns them or what license applies. Each model keeps the license its original authors published it under. Most of the models in this workspace ship permissively (Apache-2.0 or MIT), but one is not:

> **TabPFN-3's weights are licensed under the [TabPFN-3 Non-Commercial License v1.0](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE)**, not a standard open-source license. It permits research, personal, and internal evaluation use, but **not** production deployment, commercial use, or offering it as a hosted service. Read the actual license text before using the TabPFN-3 GGUF for anything beyond that.

If you're unsure whether your use case qualifies, check the license file in the model's HuggingFace repo directly — `zsfm` doesn't attempt to interpret or enforce license terms for you, for TabPFN-3 or any other model.
