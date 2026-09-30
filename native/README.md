# Native Zig on macOS

This fork implements TensorFold's inference orchestration in Zig and runs the
upstream Metal kernels through MLX-C. MLX supplies tensors, graph execution,
memory management and GPU operations. The completion executable does not run
Python; Python supplies development dependencies and correctness oracles.
The native HTTP server provides raw and chat completions, including Qwen image
inputs. Serving parity with the upstream Python server is still in progress.

The Zig serving architecture and vendored Jinja integration draw on
[ddalcu's mlx-serve](https://github.com/ddalcu/mlx-serve). Jinja dependency
credits and licenses are retained in [vendor/jinja/NOTICE](vendor/jinja/NOTICE).

## Prerequisites

- An **Apple Silicon Mac**, running **macOS 26.2+**, in a native arm64 terminal.
  Intel Macs, Linux and CUDA are outside this build.
- **Full Xcode**, selected as the active developer directory, with the macOS
  26.2+ SDK and Metal compiler/toolchain installed. Command Line Tools alone are
  insufficient. Complete Xcode's first-launch setup yourself.
- **CMake 3.25+**, **Python 3.11+** with `venv` and `pip`, and Git on `PATH`.
  Python 3.14 is used on the development machine. The pinned packages must have
  wheels compatible with your Python/macOS combination.
- GitHub **SSH access** for the default dependency checkout mode. For environments
  without credentials, use `--source-archives` to download pinned source archives
  over HTTPS without Git operations. Existing Git remotes always remain SSH.
  HTTPS is also needed for Python wheels, Zig and CMake source archives.
- Free disk space for sources, native builds, Python packages and Zig, plus
  separate space for any models or test traces you choose to use.

Check your tools:

```sh
uname -m
sw_vers -productVersion
xcode-select -p
xcrun --sdk macosx --show-sdk-version
xcrun --sdk macosx metal --version
cmake --version
python3 --version
git ls-remote git@github.com:ml-explore/mlx.git HEAD
```

Expect `arm64`, the required versions and an available Metal compiler. Install
missing prerequisites using your normal development environment. Setup does not
install system software or request administrator access.

## Quick start

Clone the Zig branch, then run every command from the repository root:

```sh
git clone --branch feat/zig git@github.com:CerebralCoding/TensorFold.git
cd TensorFold
bash scripts/fetch-zig.sh
.zig-toolchain/zig run tools/setup_native.zig -- --dry-run
.zig-toolchain/zig run tools/setup_native.zig
zig-out/bin/tensorfold --help
```

The initial build can take several minutes. `fetch-zig.sh` stages the exact
nightly in [`.zig-version`](../.zig-version), verified before extraction against
[`.zig-archive.sha256`](../.zig-archive.sha256), matching the pin used by
`mlx-serve`. It refuses to silently replace an incompatible `.zig-toolchain`.
Use the staged compiler; a global stable Zig is not compatible.
**No neighboring repository or model download is required.**

The [setup tool](../tools/setup_native.zig):

1. Checks the architecture, macOS, SDK, Metal compiler, CMake, Python and Zig.
2. Creates `.venv` if absent and installs the editable project with `test` and
   `vision` extras, constrained by every Python pin in
   [`native/dependencies.json`](dependencies.json).
3. Fetches pinned MLX, MLX-C, fmt and libjpeg-turbo sources over SSH into
   `build/deps`. Existing checkouts must be clean and use the expected remote.
4. Builds MLX/MLX-C into `build/mlx` and static JPEG into `build/jpeg`, using
   the same CMake recipes as manual upstream sync.
5. Records source revisions and SHA-256 hashes for installed artifacts, verifies
   those receipts, loads MLX-C to check the linked runtime version, and checks
   Python/native dependency parity and committed Metal kernel exports.
6. Builds the executable with safety checks, then runs host unit tests,
   checkpoint-file corruption/allocation tests and setup/sync guard tests.

It stops at the first failure. It never downloads models, modifies dependency
pins, changes the branch, syncs upstream or pushes. Re-running setup reuses the
Python environment and incremental builds while restoring the recorded source
pins. It may install or downgrade packages inside `.venv` to match those pins.

```sh
.zig-toolchain/zig run tools/setup_native.zig -- --help
.zig-toolchain/zig run tools/setup_native.zig -- --python /path/to/python3 --jobs 2
.zig-toolchain/zig run tools/setup_native.zig -- --check
.zig-toolchain/zig run tools/setup_native.zig -- --source-archives
```

`--python` selects the interpreter only when creating `.venv`. `--jobs`
controls CMake parallelism (default 4); Zig uses `-j1`.
`--check` validates prerequisites, installed artifact hashes, source revisions and
runtime loading without installing. `--source-archives` uses separate
`build/deps-archives` and `build/*-archive-build` directories; it verifies source
file receipts before reusing an archive checkout and refuses modified sources.
`--dry-run` prints the plan without running setup commands.
Invoking `zig run` itself may populate Zig's compilation cache.

## Build and edit cycle

After setup, ordinary code changes only need:

```sh
.zig-toolchain/zig build -Doptimize=safe -j1
.zig-toolchain/zig build test test-checkpoint-files test-setup test-sync-upstream -Doptimize=safe -j1
.zig-toolchain/zig build check-dependencies -j1
```

The executable is `zig-out/bin/tensorfold`. Keep its linked libraries in
`build/mlx/lib`; copying only the executable to another machine is not a
standalone distribution. Rebuild after moving the checkout or changing pins.

Format edited files with `.zig-toolchain/zig fmt <files>`. This nightly uses
optimization names `debug`, `safe`, `fast` and `small`; use `safe` for
correctness work. The default is `fast`. List targets with
`.zig-toolchain/zig build --help`.

Already-built matching prefixes can be selected using
`-Dmlx-prefix=/absolute/path` and `-Djpeg-prefix=/absolute/path` on both the
build and `check-dependencies` commands. These options do not install libraries.
Setup intentionally uses the repository-local prefixes.

## Choose the right checks

A successful host build does not establish Metal correctness. Run GPU and model
tests with `-j1` to avoid loading multiple checkpoints concurrently.
The table entries are arguments to `.zig-toolchain/zig build`:

| Checks | Arguments | Requirements |
| --- | --- | --- |
| Host unit and checkpoint-file checks | `test test-checkpoint-files -Doptimize=safe -j1` | Native libraries; no weights or GPU execution |
| Flash checkpoint names and PLE loading | `test-flash-checkpoint -Doptimize=safe -j1` | Metal and Python; small synthetic PLE/MTP checkpoints |
| Flash affine row operators | `test-flash-affine -Doptimize=safe -j1` | Metal and Python; 2/3/4/5/6/8-bit projections, hyper-connections, experts, fused PLE and forced pre-M5 variants |
| Flash mixed-format checkpoint loading | `test-flash-weights -Doptimize=safe -j1` | Metal and Python; exact packed widening, regrouping, embedding lookup and configured projections |
| Setup, sync and dependency guards | `test-setup test-sync-upstream test-dependencies test-upstream-coverage -j1` | Python for dependency tests; no models |
| Runtime loading and hardware capabilities | `test-runtime -Doptimize=safe -j1` | CPU arithmetic always checked; unavailable Metal reported explicitly |
| Hardware-selected Metal smoke suite | `test-metal-smoke -Doptimize=safe -j1` | Synthetic SIMD checks, plus tensor/GLM checks when supported; fails if Metal is unavailable |
| SIMD attention parity | `test-simd-attention -Doptimize=safe -j1` | Metal and Python; synthetic data |
| Low-bit tensor and 5/6/8-bit SIMD projections | `test-tensor-quantization test-simd-bits -Doptimize=safe -j1` | M5 for tensor checks; SIMD checks include calibration, stacked reductions and affine fallback |
| Full synthetic Metal matrix | `test-metal -Doptimize=safe -j1` | Metal and Python; includes M5-specific paths |
| Additional M5 tensor-attention fixtures | `test-metal -Dmetal-tensors=true -Doptimize=safe -j1` | M5 Metal tensor support |
| GLM backbone/MTP/cache/generation | `test-glm-model -Doptimize=safe -j1` | Metal and Python; small synthetic checkpoints |
| DeepSeek backbone/MTP/cache/generation | `test-deepseek-model test-deepseek-wide test-deepseek-packed -Doptimize=safe -j1` | Synthetic checkpoints, including production hidden/attention widths and BF16 packed hyper-connections |
| DeepSeek calibrated dense arithmetic | `test-deepseek-dense -Doptimize=safe -j1` | Synthetic scalar/MMA calibration and physical threadgroup variants |
| DSpark block drafting | `test-dspark -Doptimize=safe -j1` | Synthetic layer taps, context caches, sorted experts, Markov draws and generation; includes production widths |
| Standard DFlash / Gemma drafting | `test-dflash` / `test-gemma-draft -Doptimize=safe -j1` | Float/quantized blocks and rotary layouts; full Gemma with synthetic drafter, taps, cache rollback and seeded generation |
| DeepSeek drafter conversion | `test-drafter-conversion -Doptimize=safe -j1` | Synthetic official FP8/FP4 shards; exact upstream tensor bytes and native draft generation |
| GLM/DeepSeek kernel components | `test-large-family-kernels -Doptimize=safe -j1` | Synthetic shapes; includes hardware-specific paths |
| Qwen, Nemotron and Flash model/cache parity | `test-models -Doptimize=safe -j1` | All three installed models; substantial unified memory |
| Bonsai memory layouts | `test-bonsai-pack test-bonsai-layouts -Doptimize=safe -j1` | Installed Bonsai; budget selection, packed/widened conversion, all layouts and mixed-layout prefill/continuation |
| Gemma text/cache parity | `test-gemma-model -Doptimize=safe -j1` | Installed Gemma checkpoint |
| Request state and interleaved generation | `test-request-state test-session-rounds test-session-images -Doptimize=safe -j1` | Synthetic ownership for all backends; Qwen/Gemma/Nemotron checkpoints and Qwen image inputs |
| Prefix cache policy and restoration | `test-prompt-cache test-session-rounds -Doptimize=safe -j1` | Python policy oracle; exact Qwen/Gemma/Nemotron continuation after prefix reuse and eviction |
| Adaptive prefill boundaries and markers | `test-prefill-plan test-chat -Doptimize=safe -j1` | Python plan oracle and all seven local tokenizers; no model weights loaded |
| HTTP prefix reuse and eviction | `test-server-prefixes -Doptimize=safe -j1` | Local Qwen; JSON/SSE parity, cancellation, LRU eviction and disabled caching |
| HTTP disk snapshots | `test-server-snapshots -Doptimize=safe -j1` | Local Qwen; eviction spill, startup/on-demand reuse and corrupt-file recovery with exact seeded outputs |
| Live serving status | `test-server-live test-server-live-http -Doptimize=safe -j1` | Python rate/redraw oracle; local Qwen for request counters, prefix reuse, cancellation, queue overflow and terminal modes |
| Memory accounting | `test-memory-budget test-memory-runtime -Doptimize=safe -j1` | Upstream admission/growth-gate parity, repeated probes and Qwen/Gemma/Nemotron pause/resume correctness |
| Memory admission | `test-server-memory -Doptimize=safe -j1` | Qwen rolling reservations, waiting, refusals and cancellation with a 70 GiB process budget on the 128 GiB development Mac |
| Background priority | `test-server-background -Doptimize=safe -j1` | Qwen repeated interruption, free-slot admission, JSON/SSE replay, reasoning, images, Responses/tools and queued cancellation |
| Gemma batched prefill | `test-gemma-prefill -Doptimize=safe -j1` | Hidden states, logits, sliding/full caches and continuation through 3,212 tokens |
| Image preprocessing, encoder and end-to-end | `test-images test-vision-encoder test-vision -Doptimize=safe -j1` | Installed Qwen checkpoint and image dependencies |
| Chat templates, tokenizers and required tools | `test-chat test-tool-calls -Doptimize=safe -j1` | All seven tokenizers, including DeepSeek's official encoder and upstream's rejection of required DeepSeek tool calls; no weights loaded |
| Tool structure and copy proposals | `test-tool-drafts test-server-tool-drafts -Doptimize=safe -j1` | Seven-tokenizer upstream oracle; Qwen HTTP acceptance/rejection, serial parity, sampling and streaming |
| Responses API | `test-responses test-server-responses -Doptimize=safe -j1` | Upstream translation/events/store oracle; Qwen HTTP text, tools, images, history chains and streaming |
| Neural request state and HTTP | `test-session-neural test-server-neural -Doptimize=safe -j1` | Qwen DFlash2, Nemotron/Flash MTP and Gemma DFlash; serial parity, prefix reuse, interleaving, JSON/SSE and cancellation. Add `-Dgemma-drafter=/absolute/checkpoint` for a trained Gemma drafter; otherwise uses the synthetic fixture |
| Neural images/tools and large-family HTTP | `test-server-neural-multimodal test-server-neural-synthetic -Doptimize=safe -j1` | Qwen image/text and tool controls; synthetic GLM MTP, DeepSeek MTP and DSpark. Full GLM/DeepSeek inference remains unverified |
| Checkpoint metadata rejection | `test-schema-failures -Doptimize=safe -j1` | Installed schema checkpoints; no GPU |

Start GPU verification with:

```sh
.zig-toolchain/zig build test-metal-smoke -Doptimize=safe -j1
```

Fixtures and oracle outputs go under `build/native-checks`. Long-context/layer
traces can consume tens or hundreds of GiB. Tests do not download missing models.
Metadata checks read safetensors headers and file lengths; they do not establish
that a model fits in memory or generates correct output.

HTTP checks use the server's memory admission policy. For Flash checks on a
128 GiB Mac, set `TENSORFOLD_MEMORY_LIMIT_GB=96` if the default budget is too small.
Hardware limits and memory occupied by other applications still apply.

The `Native macOS bootstrap` GitHub workflow starts with fresh sources and a fresh
Python environment, builds dependencies from archives, checks the installation,
repeats setup and probes runtime arithmetic. It has no SSH secrets or model
downloads. Hosted-runner CPU results do not count as physical GPU qualification.

Development verification uses an M5 Max with 128 GiB unified memory. Other physical
hardware is unverified. `--metal-simd` exercises a fallback on the current machine,
not physical M1–M4 coverage. Report unsupported/skipped checks separately.

## Existing models and first completion

Use `~/.models/<publisher>/<model>`. Pass an existing directory directly:

```sh
zig-out/bin/tensorfold check-model-schema qwen "$HOME/.models/Vontra/Qwen3.8-27B-MLX-4bit"
zig-out/bin/tensorfold run "$HOME/.models/Vontra/Qwen3.8-27B-MLX-4bit" --prompt "Explain why the sky is blue." --max-tokens 32 --temperature 0 --no-drafts
```

This is a raw completion CLI: it does not automatically build a conversation or
apply a chat template. Use `--tokens ID,ID,...` for controlled comparisons.
`tensorfold serve MODEL_DIR --host 127.0.0.1 --port 8080` exposes `/health`,
`/v1/models`, `/v1/completions`, `/v1/chat/completions` and `/v1/responses`, including SSE,
seeded sampling, stop strings, reasoning content and disconnect cancellation.
Responses support GET/DELETE by ID and `previous_response_id`; the in-memory
store keeps up to 1,024 responses with a 256 MiB limit on serialized data.
`store: false` disables retention.
JSON/schema output constraints are refused until grammar support is implemented.
`--batch-streams N` controls active requests (default `4`, range `1`–`8`). One
GPU worker interleaves their prefill chunks and decode steps with separate caches;
GPU forwards are still per request. Up to eight further requests can queue.
Serving follows upstream's RAM allowance and `TENSORFOLD_MEMORY_LIMIT_GB`, capped
by Metal's recommended working set, with 3 GiB reserved outside MLX. Checkpoints
that exceed the buffer budget are rejected before loading. Serving measures cache
growth with three startup probes, reserves unfinished prompts, up to 2,048 future
reply tokens per request, shared-prefix copies and image workspace, and waits for
memory before admitting another request. Requests that cannot fit
alone are rejected. Retained text prefixes are evicted when doing so can make
admission fit. Each prefill chunk is checked before allocation. Before decode
rounds, the growth gate reserves shared-prefix copies, reclaims buffers and prefixes,
pauses newer streams, or ends the newest when the oldest cannot grow. Requests with
`priority: "background"` and short title-generation chats yield their caches when
foreground work needs a slot or memory, then replay without repeating delivered
output. `/health` reports `background_preemptions`, memory, growth-gate counters,
memory-waiting requests and prompt-cache counters. Its `inference` object reports
all admitted/queued requests, waiting requests, token totals and rates. Interactive
stdout shows the same status every half second; `TENSORFOLD_NO_LIVE=1` disables it.
Redirected stdout receives no live display. Decode rates average the preceding
two seconds; prefill shows the latest chunk's rate for two seconds and excludes
reused prefix tokens.
`--prompt-cache-gib N` sets the retention target (default RAM/8, capped at 16 GiB;
`0` disables caching). As upstream, a newest prefix may exceed this target when
the process memory budget permits it. `--checkpoint-slots N` bounds ordinary entries (default
`max(8, 3 * batch-streams)`). Text chunks follow upstream's adaptive message
boundaries; reuse requires the same boundary in the new prompt. Image requests
retain fixed chunks and bypass this cache. Text snapshots persist under
`~/.cache/tensorfold/native-prefix-snapshots`; conversation snapshots use the sibling
`native-session-snapshots` directory. Override with `--snapshot-dir PATH` or
`TENSORFOLD_SNAPSHOT_DIR`; `--snapshot-dir none` disables disk persistence.
`--max-snapshots N` bounds startup loading (default `3`; `0` keeps on-demand loading).
`--spill-gib N` enables eviction spill with a disk budget (default `0`, off).
Without spill, clean shutdown retains up to two conversations within 10 GiB.
Reuse requires matching checkpoint, executable, runtime options and dependency pins.
Cross-kernel warming remains pending.
`--request-timeout-seconds N` sets an absolute deadline from connection acceptance
through queuing and generation (default `0`, disabled). Cooperative cancellation
returns HTTP 408 or an SSE error; stalled sockets close after a 250 ms allowance.
SIGINT/SIGTERM stop admission, cancel queued/active work and release the engine.
`--shutdown-grace-seconds N` bounds stalled clients during shutdown (default `5`).
In-progress GPU operations finish before their resources are released.
Sampling defaults follow the model's `generation_config.json`; `--temperature`,
`--top-k` and `--top-p` override them, then non-null request fields take precedence.
Serving defaults to 4096 output tokens, thinking enabled and the chat template's
reasoning effort. Use `--max-tokens`, `--no-thinking`, `--reasoning-effort low|medium|xhigh`
and `--thinking-budget N` to change these defaults. The request's
`thinking_budget` overrides a nonzero default when nonzero; a negative value
disables it. Budgets force `\n</think>\n\n` through decoding for tokenizers with
a whole `</think>` token, matching upstream; Gemma's thought channel is excluded.
Chat uses each model's Jinja template. Qwen accepts user `image_url` content
parts containing data URLs, up to four images and 20 MiB of image-file bytes total;
`detail: "low"` caps each image at 256 visual tokens. `--vision-urls` also accepts
public HTTPS images on port 443, using macOS libcurl and ICU. Every DNS answer
and redirect is checked; connections use a pinned address and verified TLS.
Downloads allow three redirects, 10 MiB per image, 10 seconds per image and
30 seconds across the request. Tool calls support the
Qwen, Gemma, GLM and DeepSeek formats, named/required choice, typed arguments
and `parallel_tool_calls: false`. Qwen XML arguments stream incrementally;
typed values wait for their closing tag. Other call formats and single-call
mode use the final parser, matching upstream. Prose streams as its interpretation
becomes unambiguous. Serving verifies schema-derived tool structure and copy-span
drafts against the target model. Use request `"draft": false` or server
`--no-drafts` for serial decoding. `/health.inference` includes proposed/accepted
token totals, structural-token totals and neural-token totals. Serving enables
checkpoint MTP heads for Nemotron, Flash and GLM. `--drafter DIR` loads Qwen
DFlash2, Gemma DFlash or a converted DeepSeek MTP/DSpark folder; Gemma supports
`--drafter-bits 8|4|0`. `--max-draft N` (alias `--mtp-drafts`, default `3`, range
`0`–`15`) limits neural proposals; zero leaves only tool/copy drafting enabled.
Qwen accepts `--draft-calibration FILE`. Draft caches are request-local and
included in prefix snapshots and memory admission. No models are downloaded by
these options.
One inference worker owns the model; its queue holds eight requests.
Against a running Qwen server, compare JSON/SSE text, reasoning and images with
`.zig-toolchain/zig run tools/native_http_checks.zig -- http://127.0.0.1:8080/v1/chat/completions /path/to/image.png`.
Replace the image path with `--tools-only` to check a named tool call.
Use `--tool-stream-only` to check incremental Qwen arguments against JSON output.
Use `--controls-only` with Qwen's unmodified sampling defaults to check model
defaults, thinking-budget transitions and required calls in JSON/SSE.
With `--vision-urls` enabled, pass `HTTPS_IMAGE_URL LOCAL_COPY` after the chat
endpoint to compare HTTPS/data-URL outputs, usage and streaming. Run
`.zig-toolchain/zig build test-image-http -Doptimize=safe -j1` for offline
upstream URL/address policy checks.
`.zig-toolchain/zig build test-server-lifecycle -Doptimize=safe -j1` tests deadlines,
partial requests, cancellation recovery and signal shutdown using the existing
Qwen checkpoint, with one model loaded at a time.
`test-server-rounds` checks concurrent text/image requests against isolated
outputs, JSON/SSE parity, tool streaming and cancellation on the same checkpoint.
Use `--report build/native-checks/run.json` and
`--dump-logits build/native-checks/logits.npy` for correctness evidence; create
the output directory first.

To use an existing DFlash2 model, omit `--no-drafts` and add
`--drafter "$HOME/.models/z-lab/Qwen3.8-27B-DFlash2"`.
DFlash2 embeds upstream's greedy/sampled calibration tables; `--draft-calibration FILE`
loads an override. `tensorfold fit-draft-calibration SAMPLES_JSON OUTPUT_JSON` fits
tables from `{"source":{},"samples":{"sampled":[{"depth":0,"score":-1,"landed":true}]}}`;
optional `depth_edges`/`score_edges` override upstream bins. Verify with
`.zig-toolchain/zig build test-draft-calibration -Doptimize=safe -j1`.
`--draft-capture DIR` (or `TF_DRAFT_CAPTURE`) writes upstream-compatible `.bin`
context features, `.logits` target candidates and `.json` sampling metadata through
a bounded background queue. `test-draft-capture` checks the format and writer;
`test-draft-capture-model` checks real Qwen captures against upstream replay.
Nemotron, Flash Next and GLM use checkpoint MTP heads through `--mtp-drafts N`;
`--no-drafts` selects serial decoding.
Gemma accepts a standard DFlash checkpoint through `--drafter DIR`, with
`--drafter-bits 8` (default), `4`, or `0` to retain floating-point weights.
DeepSeek accepts `--drafter DIR` containing `model.safetensors` and `config.json`
with `model_type` set to `deepseek_v4_mtp` or `deepseek_v4_dspark`.
The folder is validated before loading the backbone; `--mtp-drafts 0` or
`--no-drafts` ignores it. Without a drafter, DeepSeek uses target-only decoding.
Convert official local shards without Python using
`zig-out/bin/tensorfold convert-drafter mtp SHARD... OUTPUT_DIR`
(optional `--layer N`, default `0`), or
`zig-out/bin/tensorfold convert-drafter dspark SHARD... OUTPUT_DIR`.
DSpark requires `config.json` beside the first shard and copies its `dspark_*`
fields. Both converters write the head's `model_type` and `model.safetensors`.
Conversion uses pinned MLX arithmetic, supports blocks split across shards,
and preserves packed FP4 expert bytes. It does not download weights.
For DSpark, the requested draft budget is capped at the checkpoint's block size.
Use `--metal-simd` for the Qwen/Nemotron/Flash SIMD path and
`--metal-sampling` for keyed Metal sampling. Match seed, sampler, temperature
and backend when comparing Python and Zig.
Bonsai's SIMD path selects packed, fully widened or `widened:N` layers from
upstream's memory policy, including `TENSORFOLD_MEMORY_LIMIT_GB`. For diagnosis,
`--bonsai-form packed|widened|widened:N` overrides that choice with `--metal-simd`;
`--bonsai-form lanes` requires the tensor backend.

Qwen image input accepts up to four local PNG, JPEG or WebP files:

```sh
zig-out/bin/tensorfold run "$HOME/.models/Vontra/Qwen3.8-27B-MLX-4bit" --image /absolute/path/photo.jpg --prompt "Describe this image." --max-tokens 32 --temperature 0 --no-drafts
```

Explicit placement uses `<|vision_start|><|image_pad|><|vision_end|>` in the
prompt. The CLI does not fetch image URLs or implement an OpenAI image API.
Native image support is Qwen-specific.

Checkpoint sizes below are approximate **weight disk space**, not peak RAM:

| Checkpoint under `~/.models/` | Size | Native verification scope |
| --- | --- | --- |
| `Vontra/Qwen3.8-27B-MLX-4bit` | 15 GiB | Text, images, DFlash2 and full-model correctness |
| `z-lab/Qwen3.8-27B-DFlash2` | 3.6 GiB | Qwen draft model |
| `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | 17 GiB | Text/MTP full-model correctness |
| `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` | 105 GiB | Text/MTP, resident/bounded PLE checks |
| `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` | 8 GiB | Tensor/SIMD prefill through 4,225 tokens and greedy/seeded draft parity |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 14 GiB | Text/cache correctness; no native image path |
| `mlx-community/DeepSeek-V4-Flash-4bit` | 144 GiB | Synthetic backbone/MTP/DSpark/cache/generation; full checkpoint execution unverified; optimized prefill pending |
| `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` | 173 GiB | Synthetic backbone/MTP/generation; full checkpoint execution unverified |

Do not attempt full-model GLM/DeepSeek inference on a 128 GiB machine based on
synthetic results. Component coverage is not full-model verification.

If you deliberately choose to download a model, check its size and license first.
This **optional** command downloads about 15 GiB:

```sh
.venv/bin/hf download Vontra/Qwen3.8-27B-MLX-4bit --local-dir "$HOME/.models/Vontra/Qwen3.8-27B-MLX-4bit"
```

Agents must ask before starting any new large model download. Setup never runs
this command. Gated models may require Hugging Face authentication.

Build tests expect `build/models/<model>`. Link existing weights rather than
duplicating them:

```sh
mkdir -p build/models
ln -s "$HOME/.models/Vontra/Qwen3.8-27B-MLX-4bit" build/models/Qwen3.8-27B-MLX-4bit
```

Repeat only for checkpoints required by your tests. Inspect an existing destination
instead of overwriting it. Alternatively pass `-Dmodel-root=/path/to/directory`
containing these short model names. `~/.models` has publisher subdirectories,
so it is not directly the flat test root. For a single large checkpoint use, for
example, `check-model-schema glm "$HOME/.models/Vontra/GLM-5.3-Flash-MLX-4bit-MTP"`.

## Dependency updates and troubleshooting

Native and Python pins follow upstream together. Do not fix installation failures
by upgrading only MLX or regenerating kernels against arbitrary versions. JPEG
matches Pillow's decoder for image parity. After pulling changed pins, stage the
recorded Zig and rerun setup.

The maintainer's sync mechanism is explicit and manual:

```sh
.zig-toolchain/zig build check-upstream -j1
.zig-toolchain/zig build sync-upstream -j1
```

These are **not contributor setup commands**: they require the configured fork
and upstream SSH remotes. Sync requires a clean tree, rebases the current branch,
resolves upstream dependencies, regenerates kernels, runs extensive checks needing
local models, then pushes fork `main`. Conflicts or failed checks stop it.
Nothing is scheduled. Contributors should pull/rebase through their normal Git
workflow and use setup to reproduce the resulting checked-in pins.

`check-upstream-coverage` validates source hashes and the feature-to-declaration/test
bindings in `native/features.json`; setup, CI and manual sync run it. New sources
need an explicit mapping before `record-upstream-coverage` can acknowledge them.
`audit-native-parity` additionally fails for every missing or partial feature.
These checks report declared test scope and limitations; they do not replace
running the mapped correctness tests or establish full-model/hardware verification.

| Symptom | Next step |
| --- | --- |
| Zig compiler/API errors | Compare `.zig-toolchain/zig version` with `.zig-version`; use the staged compiler. |
| Metal compiler/SDK missing | Check Xcode's active developer directory and Metal toolchain; complete Xcode setup manually. |
| No Metal device in a sandbox/CI | Run GPU checks with GPU access; host checks do not substitute for GPU verification. |
| Missing `mlx/c/mlx.h`, `libmlxc` or JPEG | Rerun setup; the Python MLX wheel alone does not provide the native C development prefix. |
| Missing receipt or changed artifact hash | Rerun setup to rebuild the pinned installation; do not edit receipts to conceal drift. |
| `dyld` failure after moving the checkout | Reconfigure dependencies and rebuild in the new location; preserve the local library prefix. |
| Dirty dependency source or unexpected remote | Inspect the checkout in `build/deps`; preserve edits and verify its SSH remote before retrying. |
| Python pin conflict | Check interpreter compatibility and `check-dependencies`; do not loosen MLX pins independently. |
| Missing config/checkpoint/shard | Choose synthetic tests or link the complete existing model; tests never auto-download it. |
| Memory pressure | Reduce setup `--jobs`, keep Zig `-j1`, and use component tests for models that cannot fit. |
| Stale Metal exports | Verify pins, run `.venv/bin/python tools/export_native_kernels.py`, review the diff and rerun affected parity tests. |

For a useful failure report include the commit, macOS/Xcode/SDK, chip and RAM,
exact command, first failure, Zig version, dependency check and checkpoint identity.
Separate passes from unrun checks. Benchmark only after correctness succeeds and
other load is controlled.

See [`AGENTS.md`](AGENTS.md) for native agent instructions.
