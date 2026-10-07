# TensorFold

TensorFold 1.0.0 serves language models from a Zig binary on Apple Silicon and NVIDIA GPUs.
The engine reads checkpoints, tokenizes requests and runs Metal or CUDA kernels directly.
Serving needs no Python or MLX installation.

## Install and serve

The macOS Homebrew formula installs the native binary as `tensorfold`:

```sh
brew install ashhart/tensorfold/tensorfold
tensorfold --version
tensorfold serve "$HOME/models/nemotron-lightning" \
  --name local-model --parallel 1 --context 8192 --temperature 0 --no-thinking
```

Use a complete local model directory or an existing Hugging Face cache.
The binary in a release archive is `bin/tensorfold-native`; the [runbook](RUNBOOK.md) covers archive installation and CUDA runtime files.
Use `tensorfold-native --help` and `tensorfold-native capabilities --json` to inspect the installed binary.
Model weights have separate downloads and licenses.

The API listens at `http://127.0.0.1:8080/v1` by default.
It serves OpenAI chat completions, completions and Responses, plus Anthropic Messages.
Use the model ID returned by `/v1/models`, or set one with `--name`.

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128,"temperature":0}'
```

## Qualified models

Only these model and platform combinations are admitted to 1.0.0.
The platform column names the hardware tested for that model, rather than every GPU the binary can detect.

| Model | Checkpoint format | Qualified platform |
| --- | --- | --- |
| Nemotron 3.5 Lightning 30B-A3B | MLX affine 4-bit, group 64, included MTP head | Metal on M1 through M5; CUDA on GB10, greedy serving |
| Qwen3.8 Flash Next | MLX affine 6-bit, group 32 | Metal on M5 Ultra |
| GLM-5.3-Flash | MLX affine 4-bit, group 64 | Metal on two M5 Ultras |
| Qwen3.5-2B | Pinned MLX affine 4-bit, group 64, tied embeddings | Metal on M5 Max |

Nemotron's named checkpoint is `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`.
The 2B checkpoint is `mlx-community/Qwen3.5-2B-MLX-4bit`, revision `93760be4f1f69842a46bc13dbdc0f19e291392a3`.
Flash Next loads its checkpoint directly and builds its weight packs locally, without a recorded kernel directory.
GLM's two-Mac setup uses one settings file per rank and a separate MCDMA runtime.
The [release notes](RELEASE-NOTES-1.0.0.md) give qualification limits and credit the contributors.

The 27B's drafted output passes its native plain comparison, but its paired served speed is below Python 0.6.6.
Bonsai, Gemma 4, Qwen3.6 and DeepSeek-V4 are still under qualification for 1.0.x.
The Python 0.6.6 engine remains on the `python-0.6` line for those models and other backends.
On CUDA, 1.0.0 qualifies greedy GB10 serving; seeded sampling, `top_k: 0` and device-memory telemetry have known gaps.

## Exact decoding

Every accepted draft must equal the token the same native engine would produce with `"draft": false`.
A resumed request must equal fresh execution, and each concurrent stream must equal its solo run.
The comparison fixes the checkpoint, backend, settings and runtime.
Different quantizations and different backends can produce different outputs.
Flash Next and GLM currently run one active reply per engine; Nemotron and the 2B model use shared lane rounds.

## Serve flags

The binary's `capabilities --json` response lists its supported flags and platform-specific values.
`serve MODEL --help` prints usage without loading a model.

| Flag | Meaning |
| --- | --- |
| `--host HOST`, `--port PORT` | Listen address and port, default `127.0.0.1:8080`. |
| `--name NAME` | Model ID advertised to clients. |
| `--alias NAME` | Another accepted model ID; repeat for multiple aliases. |
| `--api-key KEY` | Require a bearer key; repeat for multiple keys. |
| `--api-key-file FILE` | Read keys from a restricted-permission file. |
| `--metrics-open` | Allow `/metrics` without a key when API authentication is enabled. |
| `--dashboard` | Enable the local `/dashboard` page and `/stats` endpoint. |
| `--context N` | Bound prompt plus reply tokens; the model's window and memory checks still apply. |
| `--speed-up FILE` | Rank and link settings for two-Mac Flash Next or GLM serving. |
| `--max-tokens N` | Default reply limit, 4096; requests can override it. |
| `--temperature T` | Sampling temperature; zero requests greedy decoding. |
| `--top-p P`, `--top-k K`, `--min-p P` | Sampling defaults, subject to the CUDA qualification limits above. |
| `--thinking`, `--no-thinking` | Choose whether the chat template opens a reasoning block. |
| `--reasoning-effort LEVEL` | Default template effort: `low`, `medium`, `high` or `xhigh`. |
| `--thinking-budget N` | Limit reasoning tokens where the engine supports closing the reasoning block. |
| `--loop-guard` | Close a short repeated reasoning cycle where the engine supports it. |
| `--no-drafts` | Produce the plain reference through the same engine. |
| `--keep-warm SECONDS` | Keep the Metal GPU active while idle for this long after a request, default 900; zero disables it. |
| `--parallel N` | Admit up to N requests where the engine shares lanes; `auto` is the default. |
| `--prompt-cache-gib GIB` | Flash Next retained-prefix budget; zero disables retention. |
| `--prompt-cache-over-cap` | Permit an explicit Flash Next prefix budget above its default allowance. |
| `--snapshot-dir none` | Keep prefix state in memory; `none` is the supported value. |
| `--max-snapshots 0` | Disable disk snapshots at startup; `0` is the supported value. |
| `--no-update-check` | Accepted compatibility switch; update the installed binary through its package manager. |
| `--backend auto` | Use the backend compiled for the platform; `mlx` selects native Metal on macOS and `cuda` selects CUDA on Linux. |

Sampling defaults come from the checkpoint's `generation_config.json`, then serve flags and request fields override them.
`--context 0` is family-specific; use a positive limit for GLM and inspect the capacity reported at startup.
The supported environment variables are listed in `capabilities --json`, including API keys, request logging and Hugging Face cache paths.

## Build and contribute

Use the Zig version pinned in `.zig-version`, currently 0.17.0.
On a Mac with Xcode's Metal toolchain:

```sh
zig build native -Dcpu=apple_m1 -Dversion=1.0.0
zig build test test-golden -Dcpu=apple_m1
```

The server is `zig-out/native/bin/tensorfold-native`.
Release archives include the native executable, runtime assets and license notices; [packaging](packaging/README.md) describes the qualified CUDA inputs.
Read [CONTRIBUTING.md](CONTRIBUTING.md) for exactness, precision and performance gates.
TensorFold is Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
