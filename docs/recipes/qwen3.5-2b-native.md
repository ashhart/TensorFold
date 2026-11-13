# Qwen3.5-2B, native Metal preview

This recipe covers text generation with
[`mlx-community/Qwen3.5-2B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-4bit/tree/93760be4f1f69842a46bc13dbdc0f19e291392a3),
revision `93760be4f1f69842a46bc13dbdc0f19e291392a3`.
Build from `main` with Zig **0.17.0** and Xcode's Metal toolchain, as described in
[the native preview](../../ZIG-PREVIEW.md#build). Keep weights outside the source checkout.

```bash
hf download mlx-community/Qwen3.5-2B-MLX-4bit \
  --revision 93760be4f1f69842a46bc13dbdc0f19e291392a3 \
  --local-dir "$HOME/models/qwen35-2b"
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/qwen35-2b" \
  --name bench --port 8090 --parallel 8 --context 70000 --no-thinking
```

Structured output and required tool calls need the release archive's `lib/` or `zig build grammar -Dxgrammar=DIR`; prepare DIR with `git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar` (use the native build's same `--prefix` for grammar).

An existing download at that revision can be used directly. The native binary serves
`http://127.0.0.1:8090/v1`; decoding and tokenization run in Zig. This preview recipe invokes the native
binary directly. It does not add a release gate for Python's automatic engine selection.
The Python packed Qwen decoder's tied-head limitation is separate from this native implementation.
Normal generation stops on both the model's end IDs and the tokenizer's chat-ending token; `ignore_eos: true` disables those stops for fixed-length throughput checks.

The loader requires the 2B geometry and MLX affine 4-bit weights in groups of 64 with bf16 scales and
biases. In `capabilities --json`, `families.qwen3_5` is a union across the 2B and 27B engines,
not a promise that every listed format loads at every geometry. For this 2B geometry, only `mlx-q4g64`
is admitted; the other listed widths belong to the family's other engines. The head is the packed input embedding when the checkpoint ties it, or the checkpoint's own
`language_model.lm_head` projection when it ships one; `config.json`'s `tie_word_embeddings` must
agree with the tensors either way, and the head matmul reuses the embedding-shape pipeline either way.
The vision tower is not loaded. Images, video, other model sizes and other quantization formats are outside
this recipe. Although the config describes an MTP layer, this checkpoint has no MTP weights; context-copy
proposals use the shared lane engine with drafts enabled.

Activations remain bf16, and accumulation and DeltaNet state remain fp32. The native projection FMA chains,
row reductions and causal attention partitions differ from stock MLX's shape-selected kernels.
Another backend can therefore produce different logits or tokens. The contract is exact output within this
native engine: drafted against plain, and concurrent against solo, at the same settings.

At loading, a real-weight check compares two 16-row streams against one-row execution, including committed
logits, attention caches, recurrent state and partial keeps. A failed numerical check disables wider lanes;
one-row rounds still go through the shared lane engine. GPU execution failures stop loading.
The qualified path has up to 16 rows per stream and 32 rows per shared forward. Prompt chunks have up to
128 rows and follow the server's planned boundaries. `--no-drafts` supplies the plain reference.

Cross-request prompt reuse is opt-in with `--prompt-cache-gib N`; omitting it or passing zero keeps fresh execution.
The core prompt-cache store keeps committed recurrence, convolution tails, attention K/V and logits in host memory.
It reuses only compatible prompt cuts on the 128-token grid or the request's explicit chunk boundaries, within the
memory budget. The optional paged K/V path restores the same image into newly allocated pages. This is RAM retention;
restart persistence and learned-state files are not enabled for the 2B. With the 4-bit checkpoint on an M5 Ultra and an
M3 Ultra, every resumed chat reply equalled a fresh one. On the M3 Ultra, two prompts resuming from one 896-token kept
state, at 1, 2 and 4 streams and temperatures 0 and 1.0, gave each request the same tokens as alone (28 Oct 2026).
The checkpoint's config allows 262,144 positions, but that is not a memory or throughput qualification for
that length. `--context` controls request admission; KV buffers are sized for each request's prompt and reply.

| Checkpoint / platform | Qualification |
| --- | --- |
| Pinned affine 4-bit/group-64 checkpoint, M5 Max, macOS 27 | Native operation, forward, cache and server checks exercised; see the contribution's receipt for results and outstanding checks. |
| M1, M2, M3, M4 | Untested for this model. Uses simdgroup matrix kernels without M5 tensor units; the local load check does not replace full hardware qualification. |
| M5 on other macOS / runtime versions | Untested. Rerun the operation, fidelity and exactness checks. |
| CUDA, non-Apple platforms | No native Qwen3.5-2B GPU backend in this contribution; the CPU engine below serves it on any host. |
| Other checkpoints, model sizes, quantizations, vision / video | Outside the qualified scope. |

The preview's unresolved M1–M4 exactness qualification for existing families remains in force.
An M5 result does not establish correctness on those GPUs.

## The CPU engine

`zig build native-cpu -Doptimize=ReleaseFast` builds `zig-out/native-cpu/bin/tensorfold-native` with the same checkpoint
served on the host's cores, no GPU and no kernel library (`capabilities --json` says `"backends": ["cpu"]`); the default
builds are unchanged. It takes the same `serve` flags; `TF_QWEN35_CPU_THREADS=N` sets the worker threads beside the engine
thread, 1 to 255 (default: the logical CPUs less one), and `TF_QWEN35_CPU_WINDOWS=1` admits verify windows of up to 16
rows (copy drafts) once the engine's local exactness check against one-row steps passes at load; only this binary checks
the `TF_QWEN35_CPU_*` settings. The engine's arithmetic is the portable shape: the registered projection and RMS norm
(`zig build cpu-contract`), fp32 sums in the attention and the delta rule's recurrence, and every other row computed in
float64 and rounded to bf16 through fp32. The same bytes come back at every thread count. Across hosts, the float64
functions come from each host's libm, so equal bytes are checked, not built in: `zig build test-cpu` pins a synthetic
forward's committed bytes by digest, which CI runs on Linux, Windows and macOS, and the fidelity packet's list case gave
the same bytes on x86_64 as on arm64. Its fidelity is judged against a float64 reference, not against Metal's bits:
`tools/zig/cpu_fidelity.py` freezes a prompt set and records a serve's greedy replies; `tools/zig/qwen35_cpu_long_ref.py
--run-input` computes the reference's logits for each case (the whole forward in float64 over the exactly dequantized
weights, numpy only, the oracle #1544 judged the Vulkan engine against); and `cpu_fidelity.py compare` sets
`tf-qwen35-cpu-forward`'s teacher-forced logits (`zig build qwen35-cpu-forward`) against them row by row, listing every
top-token difference with both margins. `zig/tests/qwen35/serve_check.py` runs the drafted, concurrent and resumed
checks against its serve as against any other.

## Verification

Install the checks' Python dependency in an isolated environment. The trusted-forward check also requires
the checkpoint's tokenizer and `mlx-lm`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install mlx-lm
zig build test test-native -Dcpu=apple_m1
zig build tf-qwen35-check tf-qwen35-forward tf-qwen35-exact -Dcpu=apple_m1
zig-out/bin/tf-qwen35-check "$HOME/models/qwen35-2b"
zig-out/bin/tf-qwen35-exact "$HOME/models/qwen35-2b"
python tools/zig/qwen35_fidelity.py "$HOME/models/qwen35-2b" /tmp/qwen35-fidelity
```

The forward/cache sweep covers
attention chunk boundaries, partial acceptance followed by continuation, and unequal stream lengths.
Model fidelity is separate: the tool reports teacher-forced NLL and top-token agreement on the public
benchmark prompts, this repository's contribution guide and Python's standard library. Stock `mlx-lm`
provides the trusted forward at one row, 32-row chunks and full-prompt width to expose its own numerical
variation. These numerical comparisons do not demand bitwise equality across backends.

For the official receipt, run the [prescribed measurement tools](README.md#measurements) against the native
server. Use `tools/bench_concurrent.py --alone --serial`, `tools/bench_openai.py --tokens 64 --reps 5
--temperatures 1.0,0`, and `tools/bench_cache.py --require-cold` on a long prompt for prompt speed.
Run performance measurements without competing validation work. The previous native engine cannot run this
checkpoint, so its before baseline is unsupported; do not report an invented speed-up.
