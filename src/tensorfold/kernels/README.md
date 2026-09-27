# Kernel layout

| Family | Active Metal kernel package |
| --- | --- |
| Nemotron 3.5 Lightning | `nemotron/lightning/v1/` |
| Qwen3.8 dense | `qwen/dense/v1/` |
| Qwen3.8 Flash Next | `qwen/flash_next/v1/` |

The version names the implementation, not the model release. Each family imports its active package;
incompatible versions can occupy separate directories. CUDA kernels live in the family's `cuda/` package.

Snapshot fingerprints hash family code, active kernels and declared shared dependencies.
Nemotron declares dense-Qwen attention as a shared dependency. The snapshot key also includes the
prefill plan and the MLX, mlx-lm and TensorFold versions.

See [the family guide](../../../docs/recipes/adding-a-family.md) for arithmetic, state and verification
requirements. A multi-row kernel must match its own serial row before it can verify drafts.
