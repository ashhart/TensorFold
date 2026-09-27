# Native Zig Metal inference on Mac

This is a native port of TensorFold's Qwen3.8-27B, Nemotron Lightning, and Flash Next Metal inference paths, following the
Zig → MLX-C → MLX/Metal architecture in the neighboring `mlx-serve` project. Zig owns
loading, tokenization, the forward pass, recurrent/KV caches, DFlash2, tree verification,
and deterministic sampling. The executable does not import or launch Python.

It **uses MLX** for arrays, lazy evaluation, GPU scheduling, safetensors, and ordinary
array operations. TensorFold's specialized Metal kernels perform quantized projections,
normalization, gated DeltaNet, tree attention, and recurrent-state replay. This does not
reimplement Apple's GPU runtime.

## Build and run

Requirements: Zig **0.17.0-dev.2248+3f6a02acd** (the same nightly as `../mlx-serve`), Apple Silicon, macOS **26.2+**, and a compatible
MLX/MLX-C installation built with Metal tensor support. The tested machine is an M5 Max
with 128 GiB unified memory. M5 uses tensor kernels; older GPUs select SIMD kernels.
The SIMD path can also be exercised on M5 with `--metal-simd`; physical M1–M4 machines
have not been tested for this native port.

From the repository root, using the already-built libraries in `../mlx-serve`:

```sh
mkdir -p build
cp -R ../mlx-serve/lib/mlx build/mlx
bash scripts/fetch-zig.sh
.zig-toolchain/zig build
.zig-toolchain/zig build test
zig-out/bin/tensorfold run build/models/Qwen3.8-27B-MLX-4bit \
  --drafter build/models/Qwen3.8-27B-DFlash2 \
  --prompt 'Write a short Python function that computes the Fibonacci sequence.' \
  --max-tokens 128 --seed 1234 --warmup
```

Copy the libraries only once, when `build/mlx` does not yet exist. They have already been
staged in this checkout. The target and drafter have also been downloaded to the paths
above. These are ignored build artifacts, not committed dependencies or weights.

Use `-Dmlx-prefix=/absolute/install/prefix` to link another compatible installation.
The runtime needs `libmlx.dylib`, `libmlxc.dylib`, and `mlx.metallib` together in that
prefix's `lib` directory. Rebuild if the prefix moves; its path is embedded as an rpath.
`.zig-toolchain/zig build -Doptimize=safe` enables runtime safety checks; the default is `fast`.
The version pin is in `.zig-version`. Zig 0.17 translates the public MLX C header in the
build graph and imports the resulting module; no `@cImport` language builtin is needed.

The target checkpoint is `Vontra/Qwen3.8-27B-MLX-4bit`: sanitized BF16/affine 4-bit,
group size 64. The optional draft checkpoint is `z-lab/Qwen3.8-27B-DFlash2`; its linear
weights are quantized to 4-bit at load. The native loader validates the model recipe.

| CLI option | Behavior |
| --- | --- |
| `--prompt TEXT` | Raw text completion; no chat template is added |
| `--tokens ID,ID,...` | Exact prompt token IDs, overriding text |
| `--max-tokens N` | Maximum output tokens, default 32; stops at model EOS |
| `--drafter DIR` | Enable DFlash2 and context-copy proposals; omit for serial decoding |
| `--mtp-drafts N` | Nemotron/Flash Next: chained MTP budget, default 3, maximum 15 |
| `--no-drafts` | Nemotron/Flash Next: disable MTP and decode serially |
| `--metal-simd` | Force the non-tensor Metal path for coverage on M5 |
| `--metal-sampling` | Use the original fp32 Metal sampler instead of CPU f64 sampling |
| `--temperature T` | Default 1; zero selects greedy decoding |
| `--top-k N`, `--top-p P` | Defaults 20 and 0.95; top-k zero considers the full vocabulary |
| `--seed N` | Override the default SHA-256-derived prompt seed |
| `--warmup` | Compile common Metal variants before timing prompt/decode |
| `--report FILE` | Write prompt/output IDs, decoded text, timing, and token hash as JSON |
| `--dump-logits FILE.npy` | Save float32 logits from the final prompt block |
| `--check-exact` | Run the family's GPU verification/partial-commit parity check and exit |
| `--check-cache-stress` | Check every accepted prefix, cache snapshots, rejection, reset and memory cycles |

Text goes to stdout when decoding finishes; diagnostics go to stderr. Report parent
directories must already exist. Dense Qwen prefills in 128-token chunks; Nemotron and
Flash Next use 16-token chunks. Prompt plus
requested output is bounded to 262,144 tokens; actual memory requirements grow with context.

## How the port works

```mermaid
flowchart LR
    P[Prompt and native tokenizer] --> T[Zig target forward]
    T --> C[Committed KV and recurrent state]
    C --> D[DFlash2 or context-copy proposal]
    D --> V[Target verifies tree with Metal lane kernels]
    V --> S[Position-keyed sampling]
    S --> A[Accept matching path and commit state]
    A --> C
```

The 64 target layers alternate three gated DeltaNet layers with one full-attention
layer. All packed projections use the original Metal lane arithmetic. A verified row
has the same arithmetic regardless of the other rows in its batch. Sampling uses the
same seed/absolute-position/token hash as Python, so a draft token is accepted only when
it matches what serial decoding would select. Rejected branches never enter the cache.

DFlash2 reads five target hidden-state taps and proposes a tree of up to 15 tokens.
Eight-token suffix matches can instead propose copied continuations of up to 31 tokens.
The target verifies both through the same path. Recurrent layers replay only the accepted
path; attention layers gather only its K/V rows.

| File | Responsibility |
| --- | --- |
| `mlx.zig` | Public MLX-C bindings, array ownership, streams, Metal kernel dispatch |
| `weights.zig`, `config.zig` | Checkpoint validation/loading and packed weight preparation |
| `lanes.zig`, `metal/` | Exact quantized projections, normalization, tree metadata and attention |
| `model.zig` | Complete target forward, caches, accepted-path commit |
| `checkpoint.zig` | Shared safetensors reader and affine quantized projections |
| `nemotron.zig` | Mamba, NoPE attention, routed/shared experts, MTP and rollback |
| `flash.zig`, `ngram.zig` | Hyper-connections, GDN, sparse attention, MoE, PLE and MTP |
| `family_runtime.zig` | Chained verification and completion for Nemotron/Flash Next |
| `drafter.zig`, `copy.zig` | DFlash2 and context-copy proposals |
| `sampling.zig` | Deterministic greedy/top-k/top-p selection |
| `main.zig` | Native completion CLI and decoding loop |
| `verification.zig` | Branch/partial-commit/continuation/cache parity check |
| `acceptance.zig`, `cache_checks.zig` | Shared tree/chain acceptance and full-checkpoint cache property tests |
| `vendor/` | MIT tokenizer and I/O helpers from mlx-serve; preserved license |

The HTTP service, chat-template rendering, vision, and disk prefix caches are not part
of this native port. Attention cache
commit currently concatenates the prefix, so long-context performance needs separate
measurement. This is the native inference backend and completion CLI, not a replacement
for every `tensorfold serve` feature. Sampling defaults to CPU f64 position-keyed
sampling. `--metal-sampling` selects the original fp32 GPU algorithm (24-bit hash
uniforms and a 1,024-candidate cap); its output need not equal the f64 algorithm.
Draft verification uses the selected sampler consistently. DFlash2 candidate ranking
uses the original BF16 Metal radix top-k kernel. Nemotron uses tensor attention from
10,000 visible keys on M5 and MLX SDPA otherwise.

## Additional model families

All four required checkpoints are downloaded under ignored `build/models/` in this
checkout, including Nemotron's `mtp-4bit.safetensors` and all 22 Flash Next shards:

| Checkpoint directory | Native inference components |
| --- | --- |
| `Qwen3.8-27B-MLX-4bit` + `Qwen3.8-27B-DFlash2` | GDN, full attention, dense MLP, DFlash2 trees, context copies, recurrent replay |
| `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | Mamba convolution/SSM, NoPE attention, top-6 routed and shared experts, MTP |
| `Qwen3.8-Flash-Next-MLX-4bit-MTP` | Four residual streams, hyper-connections, GDN, indexed sparse attention, top-10 MoE, PLE hash/lookup/convolution, MTP |

```sh
zig-out/bin/tensorfold run build/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit \
  --prompt 'Write a short Python function that computes the Fibonacci sequence.' --seed 1234
zig-out/bin/tensorfold run build/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --tokens 1,2,3,4 --max-tokens 16 --seed 1234
```

Flash Next contains approximately 113 GB of tensor data. Short native runs fit on the
tested 128 GiB Mac using lazy PLE shard access, but this does not establish sufficient
memory or useful throughput for long contexts. The upstream recommended capacity is
192 GB or more.

## Validation

The checks below use the actual downloaded 27B model, not a toy substitute:

```sh
.zig-toolchain/zig build test
zig-out/bin/tensorfold run build/models/Qwen3.8-27B-MLX-4bit --check-exact
```

`--check-exact` builds a 513-token prefix (crossing the 512-token attention partition),
verifies a branched tree, partially accepts it, and compares continuation logits and
both cached arrays in all 64 layers against serial replay, bit for bit.

For parity with the Python implementation, install this repository's Python development
dependencies into `.venv`, then run:

```sh
mkdir -p build/native-checks
.venv/bin/python tools/export_native_kernels.py --check
.venv/bin/python tools/native_reference.py
zig-out/bin/tensorfold run build/models/Qwen3.8-27B-MLX-4bit \
  --tokens 1,2,3,4 --max-tokens 0 --dump-logits build/native-checks/native.npy
.venv/bin/python tools/native_reference.py \
  --compare build/native-checks/reference.npy build/native-checks/native.npy
```

All **993,280** logits matched exactly. A 128-token sampled code completion also matched
Python serial, Zig serial, and Zig DFlash2 token for token, SHA-256
`e1019b7e85e4bc6858a688dd34c854838412b1b232df1db468dcdcb821750a63`.
The measured warmed M5 Max run took 1.035 s with DFlash2 versus 4.336 s serial
(123.7 vs 29.5 output tokens/s, approximately 4.2×). These are single short-context CLI
runs, not server throughput figures; loading, warm-up, and prefill are excluded.
Separate 64-token checks also matched Python exactly for a Danish prompt with greedy
sampling and a story prompt with seed 5678, temperature 0.7, top-k 12, and top-p 0.8.
The GPU cache check and greedy DFlash2 run also passed in a ReleaseSafe build.

Nemotron's four-token pass matched all **524,288 logits** from Python exactly; its
32-token sampled MTP completion matched Python serial. Both tensor and forced SIMD
window/partial-commit checks passed, including all 52 layer caches. Flash Next's
one-token full pass matched all **248,320 logits** exactly. Its 16-token MTP output
matched Python serial, and verified rows, rollback continuation and all 48 layer
caches matched native serial. Dense Qwen's forced SIMD tree/cache check passed on M5.
Embedded kernels alone are not counted as execution coverage.
The forced SIMD DFlash2 path also produced the same 128 sampled tokens as SIMD serial
with Metal sampling enabled. Nemotron's GPU-sampled MTP output matched the original
Python GPU sampler and serial engine for the 32-token check.

The build exposes reproducible coverage targets:

```sh
.zig-toolchain/zig build test -Doptimize=safe
.zig-toolchain/zig build test-metal -Doptimize=safe -Dmetal-tensors=true
.zig-toolchain/zig build test-models -Doptimize=safe
.zig-toolchain/zig build test-cache-stress -Doptimize=safe
.zig-toolchain/zig build test-drafts -Doptimize=safe
```

`test-metal` generates independent oracles through the original Python implementation
and runs native comparisons: 44 sampling/top-k cases, three sparse-attention threshold
cases with every row and rollback continuation checked, and six tensor-attention
cases including 128-wide heads and 10K-token strided caches. Omit `-Dmetal-tensors=true`
on M1–M4. `test-models` loads the downloaded models sequentially and checks every
family's caches; it also forces the dense Qwen and Nemotron SIMD paths. It requires
the large checkpoints and enough unified memory. See [COVERAGE.md](COVERAGE.md).

`tools/native_reference.py --generate 128 --output FILE.json` creates the Python
completion report for the default prompt and seed 1234. Use native `--seed 1234 --report`
and compare with `tools/native_reference.py --compare-reports PYTHON SERIAL DRAFT`.
Use the same prompt, seed, sampling settings, and token budget in every run.

The Metal sources are checked in, generated verbatim from Python's lane modules.
After changes there, run `tools/export_native_kernels.py` and repeat the parity checks.
Python is only needed for these development checks and regeneration.

## Build MLX independently

The tested library pairing matches mlx-serve revision
`4e00f2af7a64fd846d31cfaa90247586cf853ca2`:

- MLX: `1f8e74e3f12f31365464a6867c6579f0e9b29d85`
- MLX-C: `56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6`

With CMake 3.25+, Xcode's macOS 26.2+ SDK and Metal toolchain installed, the following
builds a local prefix without requiring the neighboring repository or administrator access.
Run from the repository root. The clones require GitHub SSH access; CMake also downloads
Apple's Metal C++ headers and the JSON source archive.

```sh
mkdir -p build/deps
git clone git@github.com:ml-explore/mlx.git build/deps/mlx
git -C build/deps/mlx checkout --detach 1f8e74e3f12f31365464a6867c6579f0e9b29d85
git clone git@github.com:ml-explore/mlx-c.git build/deps/mlx-c
git -C build/deps/mlx-c checkout --detach 56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6
git clone --branch 12.1.0 --depth 1 git@github.com:fmtlib/fmt.git build/deps/fmt
cmake -S build/deps/mlx -B build/mlx-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 \
  -DBUILD_SHARED_LIBS=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
  -DFETCHCONTENT_SOURCE_DIR_FMT="$PWD/build/deps/fmt" \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/mlx"
cmake --build build/mlx-build --parallel 8
cmake --install build/mlx-build
cmake -S build/deps/mlx-c -B build/mlxc-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=26.2 \
  -DBUILD_SHARED_LIBS=ON -DMLX_C_USE_SYSTEM_MLX=ON -DMLX_C_BUILD_EXAMPLES=OFF \
  -DCMAKE_PREFIX_PATH="$PWD/build/mlx" -DCMAKE_INSTALL_PREFIX="$PWD/build/mlx"
cmake --build build/mlxc-build --parallel 8
cmake --install build/mlxc-build
.zig-toolchain/zig build
```
