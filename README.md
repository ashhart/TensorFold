# TensorFold

TensorFold serves text models on Apple Silicon and NVIDIA GPUs through an OpenAI-compatible API.
Each model family supplies its own kernels and draft verification.

```bash
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold serve Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
```

Use `http://127.0.0.1:8080/v1` as the client base URL and the model ID from `/v1/models`.
Python 3.11 or newer is required, and MLX 0.32.2 or newer on a Mac (pip installs it). See the [runbook](RUNBOOK.md)
for installation and a first request.

## Models

| Model | Checkpoint | Backend | Drafting |
| --- | --- | --- | --- |
| Nemotron 3.5 Lightning | `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | MLX, CUDA | Included MTP head; context copies on MLX |
| Qwen3.8-27B | `Vontra/Qwen3.8-27B-MLX-4bit` | MLX, CUDA | `z-lab/Qwen3.8-27B-DFlash2` and context copies; DFlash2 is optional on MLX |
| Qwen3.8 Flash Next | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` | MLX, CUDA | Included MTP head and context copies |
| GLM-5.3-Flash | `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` | MLX on a 256 GB Mac, CUDA with two ranks | MTP; optional DFlash2 on CUDA |
| Gemma 4 26B-A4B | `mlx-community/gemma-4-26b-a4b-it-4bit` | MLX | Context copies |
| Qwen3.8-27B (EXL3, experimental) | `turboderp/Qwen3.8-27B-exl3` (branches `3.00bpw`, `4.00bpw`; any codebook, 1 to 8 bits per weight) | CUDA | `z-lab/Qwen3.8-27B-DFlash2` and context copies |
| Qwen3.8 Flash Next (EXL3, experimental) | `turboderp/Qwen3.8-Flash-Next-exl3` (branch `3.05bpw_h5_ng5`; any codebook, a width per tensor) | CUDA | Included MTP head and context copies |

`tensorfold models` lists families and checkpoints. `tensorfold info MODEL` checks configuration without
fetching weights. `serve` downloads a missing checkpoint; `pull` downloads it ahead of time.

```bash
tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit
```

Qwen3.8-27B's M5 tensor-unit path reads MLX affine 2-, 3-, 4-, 5-, 6- and 8-bit projections in groups of 64.
On M1 through M4, `row_forward` uses the row-exact `simd_qmm` decoder, with 4-bit/group-64 weights and
windows of up to 16 rows. Serial and drafted calls use the same decoder. On CUDA, pull DFlash2 before
serving; without it, explicitly choose `--no-drafts` for the serial reference.

Nemotron uses TensorFold projections and routed-expert kernels. Its load-time row check controls drafting;
keep the installed MLX version within the package requirements. The named checkpoint includes
`mtp-4bit.safetensors`, which `pull` and `serve` check for.

Flash Next requires 4-bit/group-32 weights. Without an MTP head it can run without MTP drafting on MLX;
on CUDA, explicitly pass `--no-drafts`. Nemotron CUDA requires 4-bit/group-64 weights and an MTP head
unless `--no-drafts` is set. GLM on MLX reads 4-bit/group-64 weights and mlx-lm's mixed-bit conversions,
whose 5-, 6- and 8-bit tensors take their own row kernels; it needs MLX 0.32.2 or later. GLM CUDA reads
MLX 4-bit/group-64 weights and the experimental `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` conversion. GLM's optional
`incoai/GLM-5.3-Flash-DFlash2` checkpoint has non-commercial license
terms, described in [third-party notices](THIRD_PARTY_NOTICES.md).

Gemma 4 has no draft head and drafts copies of its context. Its kernels read 4-bit weights in groups of 32 or 64
with an 8-bit router, as the mlx-community conversion stores them; `serve` refuses other Gemma 4 layouts
before downloading.

See the [recipes](docs/recipes/README.md) for supported formats and backend limits.

## Exact decoding

A draft is accepted only when it equals the token the same engine would produce serially.
Sampling depends on the prompt or explicit seed, absolute position and token ID. Verify kernels keep each
row's arithmetic independent of the other rows in the call. Compare a request with the same request using
`"draft": false` to check drafted versus serial output.

The MLX engine can share a round across requests. Each stream keeps its own state and sampling key, with
concurrent output required to match its solo output. Load-time checks restrict window width and shared
forwards where a family cannot reproduce its serial arithmetic. On CUDA, `--parallel N` with N greater
than one enables shared rounds for Qwen3.8-27B on one or two ranks and Flash Next on one rank.
Flash Next rejects concurrent two-rank execution. GLM and Nemotron CUDA serve one request at a time;
CUDA `--parallel auto` also means one request at a time.

Exactness is against the same engine, weights, runtime and settings. It does not imply identical output
between MLX and CUDA, different quantizations, or different tensor-parallel rank counts.

## Serve options

| Option | Meaning | Backend |
| --- | --- | --- |
| `--host`, `--port` | Listen address, default `127.0.0.1:8080` | Both |
| `--name` | Model ID advertised to clients | Both |
| `--alias` | Additional model IDs | MLX |
| `--context N` | Prompt plus reply capacity | Both |
| `--max-tokens N` | Default reply limit, 4096 | Both |
| `--temperature`, `--top-p`, `--top-k` | Sampling defaults; temperature zero is greedy | Both |
| `--thinking`, `--no-thinking` | Template thinking toggle | Both |
| `--reasoning-effort` | Template effort, default `medium` | MLX |
| `--thinking-budget N` | Token-count limit inside reasoning | MLX |
| `--backend auto`, `mlx`, `cuda` | Select backend; auto uses MLX on macOS | Both |
| `--parallel N` | MLX `auto` admits up to 8 within budget; CUDA `auto` is 1, explicit N enables supported shared rounds | Both |
| `--no-drafts` | Decode serially | Both |
| `--drafter auto`, `none`, or model ID | Select an optional draft model where the family supports it | Both |
| `--mtp-drafts N` | Family-specific cap on MTP drafts | Both |
| `--tp 2 --rank R --master HOST` | Two-rank CUDA execution | CUDA |
| `--prompt-cache-gib N` | Retained conversation-prefix budget; zero disables retention | MLX |
| `--mlx-cache-gib N` | Reusable freed-buffer cache, default 8 GiB | MLX |
| `--snapshot-dir DIR` | Persistent prefix snapshots; `none` disables them | MLX |
| `--no-update-check` | Disable the startup release check | Both |

The default sampling settings come from `generation_config.json`. Requests can override sampling and reply
length. CUDA does not implement the MLX-only options above. See [API fields](docs/api.md) for request scope.

<a id="memory"></a>

## Context and memory

On MLX, omitted `--context` targets the model's metadata window and reduces it to the startup memory
estimate when needed, allowing room to retain a prompt for the next turn. An explicit positive value
that cannot fit one request is refused at startup. `--context 0` removes the metadata cap; finite engine
capacity and memory admission still apply. Use the reported context when configuring client compaction.

On CUDA, Qwen defaults to the affordable native capacity. GLM targets a dense 2,051-token window,
and Nemotron targets 16,384 tokens; the capacity estimate can lower these defaults. Explicit `--context 0` targets the affordable native capacity for every CUDA family.
A positive CUDA value must fit both the native window and the capacity estimate on every rank;
otherwise startup refuses it with fitting guidance. Increasing GLM beyond its dense window enables
its sparse-attention path. The startup report distinguishes native and allocated capacity.

MLX uses a process budget capped by 70% of RAM and the GPU's recommended working set. A family can state a
larger share: GLM-5.3-Flash takes 85% on a Mac with 256 GB or less, with nothing else loaded. It reserves 3 GiB
for the rest of the process before setting the MLX allocator limit. Admission accounts for weights,
cache growth, reply tokens and prefill workspace. `TENSORFOLD_MEMORY_LIMIT_GB` can lower the budget in
GiB. Retained prefixes and reusable MLX buffers have separate limits. Admission can evict retained
prefixes or queue another stream; fitting weights alone does not establish a usable context size.

An explicit reply limit is reserved before prefill. A request that exceeds context or memory is refused
with fitting guidance; an omitted reply limit is capped by the remaining context. MLX reports a
context refusal as HTTP 400 for a non-streamed request or as an error event after opening a stream.
CUDA checks context before opening a stream.

The memory-class table below keeps the model combinations under qualification. Its GiB budget ceilings
emulate the listed RAM classes before the 3 GiB process reserve. The actual default budget uses
OS-reported physical memory; a smaller GPU working set or an explicit memory limit lowers it. Context and peak-memory results remain TBD until a public
prompt fixture, checkpoint revision, runtime, command and measurement output accompany each result.

| Nominal RAM class | Budget ceiling | Qwen3.8-27B + DFlash2 | Qwen3.8-27B, `--drafter none` | Nemotron 3.5 Lightning | Qwen3.8 Flash Next |
| --- | --- | --- | --- | --- | --- |
| 36 GB | 25.2 GiB | TBD | TBD | TBD | TBD |
| 48 GB | 33.6 GiB | TBD | TBD | TBD | TBD |
| 64 GB | 44.8 GiB | TBD | TBD | TBD | TBD |
| 96 GB | 67.2 GiB | TBD | TBD | TBD | TBD |
| 128 GB | 89.6 GiB | TBD | TBD | TBD | TBD |
| 192 GB | 134.4 GiB | TBD | TBD | TBD | TBD |
| 256 GB | 179.2 GiB | TBD | TBD | TBD | TBD |

Each model cell needs the fitted context and peak physical process footprint. An emulated budget on
a larger host is not a measurement on hardware with that RAM size. These are qualification slots,
not minimum-memory promises. Weights that exceed the MLX budget are refused before loading.

## Prompt caching

On MLX, chunk starts come from the rendered token sequence. Resume points are assistant-message
starts and the second message start, using markers discovered from the chat template. The planner
skips points less than 256 tokens from the previous chunk start and otherwise cuts at the first
eligible point or after the model family's chunk. Qwen3.8 Flash Next measures one chunk at startup and
takes the largest of 8,192, 4,096 and 2,048 tokens (4,096 and 2,048 on GPUs without tensor units) that
still leaves room for 128K tokens of context; other families use 2,048. Without recognized markers it
uses that grid. There is no configurable `--prefill-grid` option.

Fresh and resumed requests use the same chunk plan. Reuse stops at a matching token prefix and a valid
chunk boundary; the previous reply is prefilled again under the current prompt. A template that rewrites
an earlier turn can reduce reuse. A follow-up therefore need not reprocess a full grid cell, but short
messages or changed earlier text can make it reprocess more than the latest reply and new messages.

Snapshots include the model, runtime, kernel and chunk-plan identity. System prefixes and retained
conversations can survive restarts. CUDA engines keep their own prompt/reply states and do not use the
MLX disk-snapshot or retained-prefix options.

<a id="dgx-spark-and-other-nvidia-gpus"></a>

## NVIDIA GPUs

Use NVIDIA's PyTorch container for CUDA, PyTorch, Triton and the extension compiler; the package has no
`cuda` installation extra. Install TensorFold inside the container without replacing that toolchain.

```bash
docker run -it --gpus all --ipc=host --network host nvcr.io/nvidia/pytorch:26.07-py3
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --host 0.0.0.0
```

Qwen3.8-27B, Flash Next and Nemotron support one or two CUDA ranks; GLM requires two.
For two ranks, see the [CUDA runbook](RUNBOOK.md#nvidia-gpus). Each rank needs its checkpoint and any
optional drafter. Rank 0 serves HTTP. Unified GPU/host memory also holds runtime buffers and file-backed
model data; the startup estimate is not a measured maximum capacity.

## Measurements

Each release's notes give its measured decode, prompt and concurrency numbers against the previous release and the
standard servers, on the machines they name: see [CHANGELOG.md](CHANGELOG.md) and the GitHub releases. The
[recipe book](docs/recipes/README.md#measurements) gives the public prompts and benchmark command, and each family's
recipe keeps its own tables.

## Updating

`tensorfold update --check` checks for a release; `tensorfold update` installs it, then the server must
restart. A normal installation uses the same interpreter's pip. An editable clone must be clean and able
to fast-forward to the release tag; afterwards run `python -m pip install -e .` in the checkout to refresh
metadata and dependencies. `--no-update-check` or `TENSORFOLD_NO_UPDATE_CHECK=1` disables startup checks.

When the update finishes it prints what changed since your version, from [CHANGELOG.md](CHANGELOG.md), which lists
every release. The first time a new version serves, it prints one line linking to its notes.

## Development and license

Family interfaces, kernel layout and verification requirements are in the [recipe book](docs/recipes/README.md),
[family map](src/tensorfold/families/README.md) and [kernel map](src/tensorfold/kernels/README.md).
MIT; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
Model weights keep their own licenses.
