# Native Zig on macOS

This fork implements TensorFold's inference orchestration in Zig and runs the
upstream Metal kernels through MLX-C. MLX supplies tensors, graph execution,
memory management and GPU operations. The completion executable does not run
Python; Python supplies development dependencies and correctness oracles.
The OpenAI-compatible HTTP server remains in the upstream Python code.

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
`mlx-serve). It refuses to silently replace an incompatible `.zig-toolchain`.
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
| Setup, sync and dependency guards | `test-setup test-sync-upstream test-dependencies test-upstream-coverage -j1` | Python for dependency tests; no models |
| Runtime loading and hardware capabilities | `test-runtime -Doptimize=safe -j1` | CPU arithmetic always checked; unavailable Metal reported explicitly |
| Hardware-selected Metal smoke suite | `test-metal-smoke -Doptimize=safe -j1` | Synthetic SIMD checks, plus tensor/GLM checks when supported; fails if Metal is unavailable |
| SIMD attention parity | `test-simd-attention -Doptimize=safe -j1` | Metal and Python; synthetic data |
| Full synthetic Metal matrix | `test-metal -Doptimize=safe -j1` | Metal and Python; includes M5-specific paths |
| Additional M5 tensor-attention fixtures | `test-metal -Dmetal-tensors=true -Doptimize=safe -j1` | M5 Metal tensor support |
| GLM backbone/MTP/cache/generation | `test-glm-model -Doptimize=safe -j1` | Metal and Python; small synthetic checkpoints |
| DeepSeek backbone/MTP/cache/generation | `test-deepseek-model test-deepseek-wide test-deepseek-packed -Doptimize=safe -j1` | Synthetic checkpoints, including production hidden/attention widths and BF16 packed hyper-connections |
| DeepSeek calibrated dense arithmetic | `test-deepseek-dense -Doptimize=safe -j1` | Synthetic scalar/MMA calibration and physical threadgroup variants |
| GLM/DeepSeek kernel components | `test-large-family-kernels -Doptimize=safe -j1` | Synthetic shapes; includes hardware-specific paths |
| Qwen, Nemotron and Flash model/cache parity | `test-models -Doptimize=safe -j1` | All three installed models; substantial unified memory |
| Gemma text/cache parity | `test-gemma-model -Doptimize=safe -j1` | Installed Gemma checkpoint |
| Image preprocessing, encoder and end-to-end | `test-images test-vision-encoder test-vision -Doptimize=safe -j1` | Installed Qwen checkpoint and image dependencies |
| Checkpoint metadata rejection | `test-schema-failures -Doptimize=safe -j1` | Installed schema checkpoints; no GPU |

Start GPU verification with:

```sh
.zig-toolchain/zig build test-metal-smoke -Doptimize=safe -j1
```

Fixtures and oracle outputs go under `build/native-checks`. Long-context/layer
traces can consume tens or hundreds of GiB. Tests do not download missing models.
Metadata checks read safetensors headers and file lengths; they do not establish
that a model fits in memory or generates correct output.

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
Use `--report build/native-checks/run.json` and
`--dump-logits build/native-checks/logits.npy` for correctness evidence; create
the output directory first.

To use an existing DFlash2 model, omit `--no-drafts` and add
`--drafter "$HOME/.models/z-lab/Qwen3.8-27B-DFlash2"`.
Nemotron, Flash Next and GLM use checkpoint MTP heads through `--mtp-drafts N`;
`--no-drafts` selects serial decoding.
DeepSeek accepts a converted `mtp.safetensors` beside its weights or in
`--drafter DIR`, with the same `--mtp-drafts N` budget.
Use `--metal-simd` for the Qwen/Nemotron/Flash SIMD path and
`--metal-sampling` for keyed Metal sampling. Match seed, sampler, temperature
and backend when comparing Python and Zig.

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
| `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` | 8 GiB | Text path and component checks; draft coverage incomplete |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 14 GiB | Text/cache correctness; no native image path |
| `mlx-community/DeepSeek-V4-Flash-4bit` | 144 GiB | Synthetic backbone/MTP/cache/generation; full checkpoint execution unverified; DSpark and optimized prefill pending |
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
