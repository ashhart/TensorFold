# Model families

The CLI discovers family packages by `MODEL_TYPES`, matching the checkpoint configuration.
All MLX families use the lane engine; CUDA families provide their own engine.

| Package | Model | MLX drafting | CUDA |
| --- | --- | --- | --- |
| `nemotron_h/` | Nemotron 3.5 Lightning | MTP and context copies | Not supported |
| `qwen3_5/` | Qwen3.8-27B | DFlash2 and context copies | One or two ranks; EXL3 or MLX 4-bit weights |
| `qwen4_exp/` | Qwen3.8 Flash Next | MTP and context copies | One or two ranks; EXL3 or MLX 4-bit weights |
| `glm5_next/` | GLM-5.3-Flash | MTP | Two ranks, MTP and optional DFlash2 |
| `gemma4/` | Gemma 4 26B-A4B | Context copies | Not supported |
| `bonsai/` | Ternary Bonsai 2 27B | DFlash2 and context copies | Not supported |

MLX load-time checks determine the usable window width and shared-forward support. Each stream has
independent state; shared execution must reproduce its solo output. CUDA requests in the HTTP server
are serialized. Backend support is declared by the package, not inferred from a model's name.

See [the recipe book](../../../docs/recipes/README.md) for checkpoints and limits,
[the MLX interface](../../../docs/recipes/adding-a-family.md),
[the CUDA interface](../../../docs/recipes/adding-a-cuda-family.md) and
[the kernel map](../kernels/README.md).
