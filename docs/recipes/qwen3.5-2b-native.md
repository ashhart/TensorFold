# Qwen3.5-2B, native Metal preview

This recipe covers text generation with
[`mlx-community/Qwen3.5-2B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-4bit/tree/93760be4f1f69842a46bc13dbdc0f19e291392a3),
revision `93760be4f1f69842a46bc13dbdc0f19e291392a3`.
Use the `zig-preview` source branch with Zig **0.17.0** and Xcode's Metal toolchain, as described in
[the native preview](../../ZIG-PREVIEW.md#build). Keep weights outside the source checkout.

```bash
hf download mlx-community/Qwen3.5-2B-MLX-4bit \
  --revision 93760be4f1f69842a46bc13dbdc0f19e291392a3 \
  --local-dir "$HOME/models/qwen35-2b"
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/qwen35-2b" \
  --name bench --port 8090 --parallel 8 --context 70000 --no-thinking
```

An existing download at that revision can be used directly. The native binary serves
`http://127.0.0.1:8090/v1`; decoding and tokenization run in Zig. This preview recipe invokes the native
binary directly. It does not add a release gate for Python's automatic engine selection.
The Python packed Qwen decoder's tied-head limitation is separate from this native implementation.
Normal generation stops on both the model's end IDs and the tokenizer's chat-ending token; `ignore_eos: true` disables those stops for fixed-length throughput checks.

The loader requires the 2B geometry, MLX affine 4-bit weights in groups of 64, bf16 scales and biases,
and tied embeddings. The packed input embedding and output projection share the same checkpoint buffer.
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

There is no cross-request prompt reuse in the native preview. Follow-up turns process the rendered
conversation afresh. A repeated reply is a fresh-execution check, not validation of a restored prefix cache.
The checkpoint's config allows 262,144 positions, but that is not a memory or throughput qualification for
that length. `--context` controls request admission; KV buffers are sized for each request's prompt and reply.

| Checkpoint / platform | Qualification |
| --- | --- |
| Pinned affine 4-bit/group-64 checkpoint, M5 Max, macOS 27 | Native operation, forward, cache and server checks exercised; see the contribution's receipt for results and outstanding checks. |
| M1, M2, M3, M4 | Untested for this model. Uses simdgroup matrix kernels without M5 tensor units; the local load check does not replace full hardware qualification. |
| M5 on other macOS / runtime versions | Untested. Rerun the operation, fidelity and exactness checks. |
| CUDA, non-Apple platforms | No native Qwen3.5-2B backend in this contribution. |
| Other checkpoints, model sizes, quantizations, vision / video | Outside the qualified scope. |

The preview's unresolved M1–M4 exactness qualification for existing families remains in force.
An M5 result does not establish correctness on those GPUs.

## Verification

Install the repository's test dependencies in an isolated environment. The capture and trusted-forward
checks also require the checkpoint's tokenizer and `mlx-lm`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,tui]'
zig build test test-native check-generated -Dcpu=apple_m1
zig build tf-qwen35-check tf-qwen35-forward tf-qwen35-exact -Dcpu=apple_m1
python tools/zig/capture_qwen35.py "$HOME/models/qwen35-2b" /tmp/qwen35-ops
zig-out/bin/tf-qwen35-check "$HOME/models/qwen35-2b" /tmp/qwen35-ops
zig-out/bin/tf-qwen35-exact "$HOME/models/qwen35-2b"
python tools/zig/qwen35_fidelity.py "$HOME/models/qwen35-2b" /tmp/qwen35-fidelity
TENSORFOLD_QWEN35_MODEL="$HOME/models/qwen35-2b" python -m pytest -q tests
```

Operation captures check binding and weight hashes with exact equality. The forward/cache sweep covers
attention chunk boundaries, partial acceptance followed by continuation, and unequal stream lengths.
Model fidelity is separate: the tool reports teacher-forced NLL and top-token agreement on the public
benchmark prompts, this repository's contribution guide and Python's standard library. Stock `mlx-lm`
provides the trusted forward at one row, 32-row chunks and full-prompt width to expose its own numerical
variation. These numerical comparisons do not demand bitwise equality across backends.

For the official receipt, run the [prescribed measurement tools](README.md#measurements) against the native
server. Use `tools/bench_concurrent.py --alone --serial`, `tools/bench_openai.py --tokens 64 --reps 5
--temperatures 1.0,0`, and `tools/prefill_cold.py` at its public default lengths and repetitions.
Run performance measurements without competing validation work. The previous native engine cannot run this
checkpoint, so its before baseline is unsupported; do not report an invented speed-up.
