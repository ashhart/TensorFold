# Kernel layout

| Model family | Active kernel package | Version |
| --- | --- | --- |
| GLM-5.3-Flash | `glm/flash/v1/` | `v1` |
| Qwen3.8 dense | `qwen/dense/v1/` | `v1` |
| Qwen3.8 Flash Next | `qwen/flash_next/v1/` | `v1` |
| NVIDIA Nemotron 3.5 Lightning | `nemotron/lightning/v1/` | `v1` |

Each folder contains the kernels the matching family imports. `v1` names this implementation, not the model
release. A later incompatible implementation gets its own `v2` folder, and the family changes its import to
select it. The CLI also hashes the active source code in its prompt-snapshot key, so editing code within a
version cannot reuse a snapshot computed by different kernels.

Nemotron's long-context attention uses `lane_sdpa` from Qwen dense `v1`; its snapshot fingerprint includes that
shared source too.
