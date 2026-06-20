# Supported Models

TensorFold Runtime targets MoE checkpoints where only a small active expert set
is needed for each token.

## Current Status

| Family | Status | Notes |
| --- | --- | --- |
| Built-in toy safetensors | Supported | Used by `tensorfold demo create`, `inspect`, `pack`, and `selftest`; no model inference. |
| Qwen-style MoE MLX checkpoints | Supported by the low-memory runtime path | Best fit for expert packing, concurrent pack reads, and bounded RSS experiments. |
| GPT-OSS MLX checkpoints | Supported by the exact-first streaming path | Useful for native/reference comparisons and memory accounting. |
| Nemotron-style sparse MoE | Experimental | Candidate family for the same expert-packing approach; requires model-specific validation. |
| Other safetensors/MLX MoE checkpoints | Experimental | Works when tensor naming, router behavior, and expert shapes match existing assumptions. |
| Dense-only transformers | Inspect/manifest only | Dense models can be inspected and planned, but TensorFold's strongest memory wins come from sparse expert activation. |

## Not A Blanket Claim

TensorFold does not promise that every model runs at native speed under a fixed
memory fraction. Each model/profile needs a fresh benchmark with exactness,
RSS, resident weight peak, Metal peak, and bytes-read telemetry.

## Not Yet General

TensorFold does not currently promise universal support for every transformer
architecture. Dense models can be inspected and planned, but the strongest
memory wins come from MoE sparsity and expert packing.

## Platform Support

TensorFold's low-memory serving path is Apple Silicon / MLX-first in this
public alpha. The measured Qwen profile used an MLX checkpoint and Apple
Metal-backed execution.

Linux is still useful for ordinary release tooling:

- `tensorfold doctor`
- `tensorfold selftest`
- `tensorfold inspect`
- `tensorfold manifest`
- `tensorfold pack`

The OpenAI-compatible low-memory serving path is not yet a proven Linux target.
A CUDA, ROCm, or CPU Linux backend would need its own implementation work and
benchmark artifacts.

## Compatibility Checklist

A model is a good candidate when:

- weights are available in local safetensors or MLX-compatible format
- the tokenizer can be loaded by the existing MLX stack
- MoE expert tensors have stable names and shapes
- selected experts can be packed or mapped into a bounded resident arena
- exactness can be checked against a native or direct-QMM reference
