# Model families

One package per model family. TensorFold picks the package whose `MODEL_TYPES` holds the checkpoint's
`model_type` (from `config.json`); nothing else registers a family, so a new model is a new folder here.

| Package | Model | Engine | Kernels | Tested checkpoint |
| --- | --- | --- | --- | --- |
| `glm5_next/` | GLM-5.3-Flash | CUDA only, two GPUs: MTP and DFlash2 drafts, CUDA graphs | `glm5_next/cuda/` | `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` with `incoai/GLM-5.3-Flash-DFlash2` |
| `nemotron_h/` | Nemotron 3.5 Lightning | serial engine, one step ahead on the GPU, MTP drafts | `nemotron/lightning/v1` | `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` (MTP head included) |
| `qwen3_5/` | Qwen3.8 dense (27B) | lane engine, DFlash2 draft trees; on CUDA the same trees, one or two GPUs | `qwen/dense/v1`; `qwen3_5/cuda/` | `Vontra/Qwen3.8-27B-MLX-4bit` with `z-lab/Qwen3.8-27B-DFlash2` |
| `qwen4_exp/` | Qwen3.8 Flash Next | serial engine, MTP drafts; on CUDA MTP chains in CUDA graphs, one or two GPUs | `qwen/flash_next/v1`; `qwen4_exp/cuda/` | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` |

What a package declares and what the engines call on its model is in
[adding a family](../../../docs/recipes/adding-a-family.md), and for NVIDIA GPUs in
[adding a CUDA family](../../../docs/recipes/adding-a-cuda-family.md). What we did for each of them, with the
numbers, is in [the recipe book](../../../docs/recipes/README.md).
The versioned kernel folders are listed in [the kernel layout](../kernels/README.md).
