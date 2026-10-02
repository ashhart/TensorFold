# TensorFold

TensorFold serves language models on Apple Silicon and NVIDIA GPUs through an OpenAI-compatible API.
Each model family supplies its own kernels and draft verification.

```bash
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold serve Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
```

On a Mac, Homebrew installs it too: `brew install ashhart/tensorfold/tensorfold`.

Use `http://127.0.0.1:8080/v1` as the client base URL and the model ID from `/v1/models`. Both backends serve chat
completions, completions, OpenAI Responses (`/v1/responses`) and Anthropic Messages (`/v1/messages`); see the [API reference](docs/api.md).
Python 3.11 or newer is required, and MLX 0.32.2 or newer on a Mac (pip installs it). See the [runbook](RUNBOOK.md)
for installation and a first request. On NVIDIA GPUs the CUDA kernels need compute capability 8.9 or newer: Ada (RTX 40
series), Hopper and Blackwell, including the DGX Spark's GB10 and the RTX 50 series. NVFP4 and FP8 checkpoints run from
8.9: their own math where the GPU has each mma (FP4 on 12.x, FP8 from 8.9), W4A16 elsewhere; the RTX 40, Hopper and
B200 builds are compiled and bit-checked on Blackwell but not yet run on those cards. RTX 30 cards (8.6) aren't
supported, and the server refuses a GPU below 8.9 at startup.

## Image input

Install the vision extra, `python -m pip install 'tensorfold[vision] @ git+https://github.com/ashhart/TensorFold.git'`,
and start a supported GLM-5.3-Flash or Qwen3.5/3.8 dense checkpoint with `--vision` to accept image and text content
parts through the same lane engine; Flash Next CUDA also accepts images with `--vision --parallel 2` or more.
GLM-5.3-Flash images run on MLX; dense Qwen's run on MLX and CUDA. See
[image input](docs/vision.md) for the API, checkpoint requirements, cache behavior and qualification status.

## Models

| Model | Checkpoint | Backend | Drafting |
| --- | --- | --- | --- |
| Nemotron 3.5 Lightning | `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | MLX, CUDA | Included MTP head; context copies on MLX |
| Qwen3.8-27B | `Vontra/Qwen3.8-27B-MLX-4bit` | MLX, CUDA | `z-lab/Qwen3.8-27B-DFlash2` and context copies; DFlash2 is optional on MLX |
| Qwen3.8 Flash Next | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` | MLX, CUDA | Included MTP head and context copies |
| GLM-5.3-Flash | `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` | MLX on a 256 GB Mac, CUDA with two ranks | MTP; optional DFlash2 on CUDA |
| Gemma 4 26B-A4B | `mlx-community/gemma-4-26b-a4b-it-4bit` | MLX | Context copies; `z-lab/gemma-4-26B-A4B-it-DFlash` is optional |
| DeepSeek-V4-Flash | `mlx-community/DeepSeek-V4-Flash-4bit` | MLX on a 256 GB Mac | `Vontra/DeepSeek-V4-Flash-DSpark-MLX` or `Vontra/DeepSeek-V4-Flash-MTP-MLX` |
| Qwen3.8-27B (NVFP4) | `nvidia/Qwen3.8-27B-NVFP4` (ModelOpt: NVFP4 MLP, FP8 attention) | CUDA, one GPU | `z-lab/Qwen3.8-27B-DFlash2` and context copies |
| Qwen3.8-27B (EXL3, experimental) | `turboderp/Qwen3.8-27B-exl3` (branches `3.00bpw`, `4.00bpw`; any codebook, 1 to 8 bits per weight) | CUDA | `z-lab/Qwen3.8-27B-DFlash2` and context copies |
| Qwen3.8 Flash Next (EXL3, experimental) | `turboderp/Qwen3.8-Flash-Next-exl3` (branch `3.05bpw_h5_ng5`; any codebook, a width per tensor) | CUDA | Included MTP head and context copies |
| Ternary Bonsai 2 27B | `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` | MLX | `z-lab/Qwen3.8-27B-DFlash2` and context copies |
| Qwen3.8 Flash Next (NVFP4) | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (ModelOpt: NVFP4 experts, MXFP8 attention and DeltaNet); `RadixArk/Qwen3.8-Flash-Next-NVFP4` (bf16 besides the experts) | CUDA, one GPU | Included MTP head and context copies |

`tensorfold models` lists families and checkpoints. `tensorfold info MODEL` checks configuration without
fetching weights. `serve` downloads a missing checkpoint; `pull` downloads it ahead of time.

```bash
tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit
```

Qwen3.8-27B reads MLX affine 2-, 3-, 4-, 5-, 6- and 8-bit checkpoints, including mixed layer formats.
It reads packed rows in groups of 32, 64 and 128 on Apple Silicon and CUDA, with hardware qualification still
pending for the newer paths. M5 keeps its native tensor-unit kernels for compatible formats,
and other formats use the row decoder. See [quantized checkpoints](docs/quantization.md) for the exact scope.
On CUDA, pull DFlash2 before serving; without it, explicitly choose `--no-drafts` for the serial reference.

Nemotron uses TensorFold projections and routed-expert kernels. Its load-time row check controls drafting;
keep the installed MLX version within the package requirements. The named checkpoint includes
`mtp-4bit.safetensors`, which `pull` and `serve` check for.

Flash Next requires 4-bit/group-32 weights. Without an MTP head it can run without MTP drafting on MLX;
on CUDA, explicitly pass `--no-drafts`. On one CUDA GPU it also reads the two NVFP4 exports in the table as they
ship, and block-scaled FP8 (ModelOpt `FP8_PB_WO`) linears in such exports; see [the recipe](docs/recipes/qwen3.8-flash-next.md#nvfp4-checkpoints) for their formats, the checks they
passed and what is not supported. Nemotron CUDA requires 4-bit/group-64 weights and an MTP head
unless `--no-drafts` is set. GLM on MLX reads 4-bit/group-64 weights and mlx-lm's mixed-bit conversions,
whose 5-, 6- and 8-bit tensors take their own row kernels; it needs MLX 0.32.2 or later. GLM CUDA reads
MLX 4-bit/group-64 weights and Brandon M. Music's experimental EXL3/TR3 checkpoint
(`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, also re-hosted as `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`). GLM's optional
`incoai/GLM-5.3-Flash-DFlash2` checkpoint has non-commercial license
terms, described in [third-party notices](THIRD_PARTY_NOTICES.md).

Gemma 4 has no draft head. It drafts copies of its context, and chains from z-lab's DFlash model when served with
`--drafter z-lab/gemma-4-26B-A4B-it-DFlash` (pulled once). Its kernels read 4-bit weights in groups of 32 or 64
with an 8-bit router, as the mlx-community conversion stores them; `serve` refuses other Gemma 4 layouts
before downloading.

DeepSeek-V4-Flash reads the mlx-community conversion (affine 4-bit/group-64 weights, mxfp4 routed experts) and
needs MLX 0.32.2 or later. Its draft heads are DeepSeek's DSpark blocks and MTP layer (MIT), converted:
`tensorfold pull Vontra/DeepSeek-V4-Flash-DSpark-MLX` once and `serve` drafts with it; see
[its recipe](docs/recipes/deepseek-v4-flash.md).

See the [recipes](docs/recipes/README.md) for supported formats and backend limits.

## Exact decoding

A draft is accepted only when it equals the token the same engine would produce serially.
Sampling depends on the prompt or explicit seed, absolute position and token ID. Verify kernels keep each
row's arithmetic independent of the other rows in the call. Compare a request with the same request using
`"draft": false` to check drafted versus serial output.

The MLX engine can share a round across requests. Each stream keeps its own state and sampling key, with
concurrent output required to match its solo output. Load-time checks restrict window width and shared
forwards where a family cannot reproduce its serial arithmetic. On CUDA, `--parallel N` with N greater
than one enables shared rounds for Qwen3.8-27B on one or two ranks and for Flash Next and Qwen3.6-35B-A3B on
one rank. Flash Next rejects concurrent two-rank execution. GLM and Nemotron CUDA serve one request at a time;
CUDA `--parallel auto` also means one request at a time.

Exactness is against the same engine, weights, runtime and settings. It does not imply identical output
between MLX and CUDA, different quantizations, or different tensor-parallel rank counts.

## Serve options

| Option | Meaning | Backend |
| --- | --- | --- |
| `--host`, `--port` | Listen address, default `127.0.0.1:8080` | Both |
| `--name` | Model ID advertised to clients | Both |
| `--vision` | Opt-in GLM-5.3-Flash, Qwen3.5/3.8 dense and Flash Next image input | MLX; dense Qwen also CUDA; Flash Next CUDA with `--parallel >=2` |
| `--vision-max-images N` | With `--vision`, images across the full request history (default 4); other image limits still apply | Both |
| `--alias` | Additional model IDs | MLX |
| `--context N` | Prompt plus reply capacity | Both |
| `--max-tokens N` | Default reply limit, 4096 | Both |
| `--temperature`, `--top-p`, `--top-k`, `--min-p` | Sampling defaults; temperature zero is greedy | Both |
| `--thinking`, `--no-thinking` | Template thinking toggle | Both |
| `--reasoning-effort` | Template effort when a request sets none; default: the template's own | Both |
| `--thinking-budget N` | Token-count limit inside reasoning | Both |
| `--backend auto`, `mlx`, `cuda` | Select backend; auto uses MLX on macOS | Both |
| `--parallel N` | MLX `auto` admits up to 8 within budget; CUDA `auto` is 1, explicit N enables supported shared rounds | Both |
| `--no-drafts` | Decode serially | Both |
| `--drafter auto`, `none`, or model ID | Select an optional draft model where the family supports it | Both |
| `--mtp-drafts N` | Family-specific cap on MTP drafts | Both |
| `--kv-dtype bf16`, `int8`, `int4` | Flash Next: `int8` or `int4` stores keys and values with one fp16 scale per 32 values. Other families and the MLX path refuse it | CUDA |
| `--mtp-confidence P` | Flash Next: stop a draft chain before a later draft under this probability, 0 to 1 (default 0.70) | CUDA |
| `--prefill-fp8` | Prompt matmuls take FP8 (e4m3) activations, one scale a row, where the checkpoint has an FP8 prompt kernel (Qwen3.8 27B and Qwen3.6 MLX 4-bit, FP8 and MXFP8 layers of NVFP4 checkpoints): faster prompts at lower precision ([measured](docs/recipes/cuda.md#prompt-precision)). Default: bf16 activations, as decode | CUDA |
| `--precision checkpoint`, `full` | NVFP4 checkpoints: `checkpoint` (default) runs their own math, FP4 x FP4 on SM 12.x and FP8 x FP8 from 8.9, W4A16 elsewhere; `full` runs bf16 activations against the stored weights ([measured](docs/recipes/cuda.md#nvfp4-precision)) | CUDA |
| `--tp 2 --rank R --master HOST` | Two-rank CUDA execution; `--master-port P` sets rank 0's rendezvous port (default 29551) | CUDA |
| `--decode-share F` | Mac: while prompts prefill, running replies keep moving for this share of each chunk's time; a new prompt starts at the next chunk, the fewest tokens left first (default 0.25; 0 prefills whole prompts first, in order, as 0.3.6.2). Flash Next on CUDA with `--parallel N`: replies decode inside each prompt pass, and the share sizes the passes so a round's decoding takes it (default 0: whole passes) | Both |
| `--prompt-cache-gib N` | Retained conversation-prefix budget; zero disables retention. Default: the memory the weights, a whole-window request and a shared round leave idle, at least an eighth of RAM up to 16 GiB, given back on demand | MLX |
| `--prefill-pass N` | Plan chunks one forward takes while a prompt fills alone, for families with a prompt pass (default 8; 1 as 0.5.0) | MLX |
| `--pass-cache-gib N` | Freed-buffer cache during such a pass where the memory budget has room, default 16 GiB | MLX |
| `--checkpoint-slots N` | Retained conversation prefixes (default 3 per lane, at least 8); long conversations hit this before the byte budget. On CUDA, the prompt states Qwen3.8-27B keeps under `--parallel` 2 or more (default 3) | Both |
| `--spill-gib N` | Write evicted conversation prefixes to disk (up to N GiB) and read them back instead of prefilling again; zero disables | MLX |
| `--mlx-cache-gib N` | Reusable freed-buffer cache, default 8 GiB | MLX |
| `--snapshot-dir DIR` | Persistent prefix snapshots; `none` disables them | MLX |
| `--max-snapshots N` | System-block snapshots loaded at start, default 3 | MLX |
| `--no-update-check` | Disable the startup release check | Both |

The default sampling settings come from `generation_config.json`. Requests can override sampling and reply
length. CUDA does not implement the MLX-only options above. See [API fields](docs/api.md) for request scope.

In a terminal, `tensorfold serve` keeps one live throughput line under its log; it is off when output is
redirected, and `TENSORFOLD_NO_LIVE=1` turns it off.

<a id="memory"></a>

## Context and memory

On MLX, omitted `--context` targets the model's metadata window and reduces it to the startup memory
estimate when needed, allowing room to retain a prompt for the next turn. An explicit positive value
that cannot fit one request is refused at startup. A larger explicit window may fit a request without
leaving room to retain its prompt, so a later turn can need a full prefill. `--context 0` removes the
metadata cap; finite engine capacity and memory admission still apply. Use the reported context when
configuring client compaction.

On CUDA, Qwen defaults to the affordable native capacity. GLM targets a dense 2,051-token window,
and Nemotron targets 16,384 tokens; the capacity estimate can lower these defaults. Flash Next's `--kv-dtype int8` or `int4` counts
its smaller cache, so the same memory admits a longer window. Explicit `--context 0` targets the affordable native capacity for every CUDA family.
A positive CUDA value must fit both the native window and the capacity estimate on every rank;
otherwise startup refuses it with fitting guidance. Increasing GLM beyond its dense window enables
its sparse-attention path. The startup report distinguishes native and allocated capacity.

MLX defaults to a process budget of 70% of RAM. A family can state a larger share: GLM-5.3-Flash takes 85%
on a Mac with 256 GB or less, with nothing else loaded. `TENSORFOLD_MEMORY_LIMIT_GB` replaces that default in
GiB, raising or lowering it; physical RAM and the GPU's recommended working set still cap the result.
The server reserves 3 GiB for the rest of the process before setting the MLX allocator limit, so a
110 GiB process budget allows 107 GiB of MLX buffers. Concurrent admission honors the raised budget
while accounting for memory held elsewhere on the machine. Admission accounts for weights, cache
growth, reply tokens and prefill workspace. Retained prefixes and reusable MLX buffers have separate
limits. Admission can evict retained prefixes or queue another stream; fitting weights alone does not
establish a usable context size. A larger budget leaves less RAM for other applications and cached file pages.
The startup line says how far the variable can raise the budget on this Mac, and a startup refusal says how far it
must go. On a 32 GB Mac, Qwen3.8-27B needs more than the default 22.4 GiB, with or without its draft model.

Flash Next's startup weight check excludes n-gram tensors when the loader keeps them in host file
mappings. The startup report shows resident and file-backed bytes separately. Cached file pages still
consume RAM and can be reclaimed by the OS; see [Flash Next memory](docs/recipes/qwen3.8-flash-next.md#mlx-execution).

An explicit reply limit is reserved before prefill. A request that exceeds context or memory is refused
with fitting guidance; an omitted reply limit is capped by the remaining context. MLX reports a
context refusal as HTTP 400 for a non-streamed request or as an error event after opening a stream.
CUDA checks context before opening a stream.

On MLX, streams that share rounds take memory as they grow. A stream beside others holds its next 2,048
tokens of growth, not its whole reply, so `--parallel` streams are admitted while their real contexts fit.
Before each round the server checks that the live streams' next growth fits. If it doesn't, it first frees
MLX's cached buffers and retained prefixes, but only when that makes room. Then the newest streams wait a
round, keeping their state, so their tokens don't change. If even the oldest stream can't grow, the newest one
ends with an error that says so; the tokens it already sent stay valid.

The memory-class table below keeps the model combinations under qualification. Its GiB budget ceilings
emulate the listed RAM classes before the 3 GiB process reserve. The actual default budget uses
OS-reported physical memory and the GPU working set; an explicit memory budget can lower or raise it.
Context and peak-memory results remain TBD until a public
prompt fixture, checkpoint revision, runtime, command and measurement output accompany each result.

| Nominal RAM class | Budget ceiling | Qwen3.8-27B + DFlash2 | Qwen3.8-27B, `--drafter none` | Nemotron 3.5 Lightning | Qwen3.8 Flash Next |
| --- | --- | --- | --- | --- | --- |
| 32 GB | 22.4 GiB | TBD | TBD | TBD | TBD |
| 36 GB | 25.2 GiB | TBD | TBD | TBD | TBD |
| 48 GB | 33.6 GiB | TBD | TBD | TBD | TBD |
| 64 GB | 44.8 GiB | 140,288 tokens, 43.7 GiB (0.3.5.1) | 152,576 tokens, 39.9 GiB (0.3.5.1) | 262,144 tokens, 42.2 GiB (0.3.5.1) | Not run: 4-bit weights exceed the budget |
| 96 GB | 67.2 GiB | TBD | TBD | TBD | TBD |
| 128 GB | 89.6 GiB | TBD | TBD | TBD | TBD |
| 192 GB | 134.4 GiB | TBD | TBD | TBD | TBD |
| 256 GB | 179.2 GiB | TBD | TBD | TBD | TBD |

Each model cell needs the fitted context and peak physical process footprint. An emulated budget on
a larger host is not a measurement on hardware with that RAM size. These are qualification slots,
not minimum-memory promises. Weights that exceed the MLX budget are refused before loading.

The 64 GB row comes from @benwilson's run on a 64 GB M5 Pro with TensorFold 0.3.5.1 and MLX 0.31.2 (#70),
not from this release. Each cell is the fitted context at the default budget and the process's lifetime peak
footprint over cold and resumed prompts at that context. The [Qwen3.8-27B recipe](docs/recipes/qwen3.8-27b.md#a-64-gb-m5-pro-on-0351)
lists the checkpoint revisions, fixture, commands and results. On 0.3.5.1 with DFlash2, prompts above about
100,000 tokens were served but not kept for the next turn (#71).

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

`tensorfold update --check` checks for a release; `tensorfold update` installs it (`--force` reinstalls the newest
release even when it is current), then the server must restart. A normal installation uses the same interpreter's pip. An editable clone must be clean and able
to fast-forward to the release tag; afterwards run `python -m pip install -e .` in the checkout to refresh
metadata and dependencies. A Homebrew install upgrades with `brew upgrade tensorfold` instead. `--no-update-check`
or `TENSORFOLD_NO_UPDATE_CHECK=1` disables startup checks.

When the update finishes it prints what changed since your version, from [CHANGELOG.md](CHANGELOG.md), which lists
every release. The first time a new version serves, it prints one line linking to its notes.

## Development and license

Family interfaces, kernel layout and verification requirements are in the [recipe book](docs/recipes/README.md),
[family map](src/tensorfold/families/README.md) and [kernel map](src/tensorfold/kernels/README.md).
Apache-2.0 from 0.6.0; see [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md).
Releases up to 0.5.0 were MIT, and code written before 0.6.0 keeps its [MIT notice](LICENSES/MIT.txt).
Model weights keep their own licenses.
