# Checkpoint configuration fixtures

These are the unmodified `config.json` files from the four downloaded checkpoints in
`build/models`, retained so configuration rejection tests need no weights or GPU:

| Fixture | Checkpoint directory |
| --- | --- |
| `qwen.json` | `Qwen3.8-27B-MLX-4bit` |
| `dflash.json` | `Qwen3.8-27B-DFlash2` |
| `nemotron.json` | `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` |
| `flash.json` | `Qwen3.8-Flash-Next-MLX-4bit-MTP` |

The tests first accept each original file, then reject mutations of the recipe fields.
These metadata fixtures contain no model weights.
