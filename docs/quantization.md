# Quantized checkpoints

TensorFold reads the precision stored in a checkpoint, including its per-layer settings.
Serve a local or Hugging Face checkpoint through the usual CLI:

```bash
tensorfold info ./model-8bit
tensorfold serve ./model-8bit
```

This build extends the Qwen3.5/3.8 dense family to MLX affine weights at 2, 3, 4, 5, 6 and 8 bits, with group sizes 32, 64 and 128.
It keeps packed words, scales and biases at their stored precision, including mixed layer formats.
It does not convert a checkpoint during loading.
Choose an 8-bit checkpoint made from the original model when you want its 8-bit quality.

## Backend paths

The table, `--lane-kernels` and the CUDA path below describe the Python engine, frozen at 0.6.6 and available at the
[`v0.6.6` tag](https://github.com/ashhart/TensorFold/tree/v0.6.6). The native binary refuses `--lane-kernels`. Its
Qwen3.8-27B family reads the same affine formats, per module (`zig/src/families/qwen27/affine.zig`), and admits each
against the projection kernels the chip runs (`zig/src/families/qwen27/widths.zig`). Its Nemotron family reads them per
module on Metal (`zig/src/families/nemotron/config.zig`). Matrices at 4-bit in groups of 64 keep Nemotron's own
kernels. Every other width runs on the shared row kernels in decode, and in prompt chunks past 128 rows on
simd_qmm_bits' tiles (dense matrices in groups of 64) and Flash Next's width-general expert gathers (routed experts).
For Nemotron's MLX affine checkpoints, its CUDA and Intel GPU engines read 4-bit in groups of 64 only.

| Path | Formats |
| --- | --- |
| Qwen dense, M5 native lane kernels | Affine 2/3/4/5/6/8-bit, group 64; affine 4-bit also group 32 |
| Qwen dense, packed row kernels on Apple Silicon | Affine 2/3/4/5/6/8-bit, groups 32/64/128 |
| Qwen dense, CUDA | Affine 2/3/4/5/6/8-bit, groups 32/64/128 |
| Nemotron | Existing affine 4-bit recipe; broader expert formats are not enabled here |
| Flash Next | Existing uniform affine 4-bit/group-32 recipe |
| GLM and EXL3 | Existing family recipes and separate EXL3 integration |

With `--lane-kernels auto`, M5 uses its native kernels when they support the checkpoint's formats and otherwise uses the packed row decoder.
M1 through M4 use the packed row decoder.
`--lane-kernels on` requires an M5-generation GPU and formats supported by its native kernels; an incompatible format is refused with guidance.
The language path retains BF16 activation requirements.
The new Metal affine projection reader accepts BF16, FP16 or FP32 affine scales and biases with matching metadata dtypes.

All paths still verify draft rows through the lane engine.
The existing specialized 4-bit routes remain available; the general readers make different formats executable without expanding the entire model into dense weights.
Their performance can differ substantially from the specialized routes.

## Mixed precision

The loader reads `quantization` before the legacy `quantization_config`, matching the current MLX loader.
A nonempty per-module mapping uses that mapping's own defaults, so `{"bits": 8}` means affine 8-bit with group 64 even when the global group is 128.
Boolean `true` inherits global settings; `false` and an empty mapping disable quantization for that module where the family supports its resulting unquantized projection.
Conflicting aliases and packed shapes that disagree with the declared bit width or group size are refused.

Fused stacks combine projections only when bit width, group size, input width and metadata dtypes agree.
Otherwise the model computes those projections separately with their declared formats.
In the Python 0.6.6 engine, CUDA tensor-parallel splits retain complete groups and packed-word boundaries, and memory
admission counts the actual packed precision.
Changing a checkpoint, precision or kernel implementation also changes the conditions under which cached states remain valid.

## Quantizing like MLX

`tensorfold` quantizes a dense tensor to MLX affine 4-bit (groups of 32, 64 or 128) with MLX's own scheme: float32 arithmetic, the bias at the larger-magnitude edge of a group, a signed scale that puts zero on the grid, round half away from zero, codes from the unrounded scale and bias with bf16 only for storage.
When the dense source and the conversion scheme match how a checkpoint was made, this reproduces its words, scales and biases exactly.
`zig build tf-merge-check` builds a check of that: `tf-merge-check --quantizer DENSE SHIPPED MODULE...` quantizes each named dense tensor and compares it with the shipped one, reporting the share of equal scales, biases, codes and whole groups.
A dense tensor of another shape, size or dtype than the shipped projection, and any non-finite value or stored scale, is refused naming the tensor.

## Merging a LoRA

`tensorfold merge MODEL ADAPTER OUT` folds a PEFT LoRA into an MLX affine 4-bit checkpoint without a Python toolchain or a GPU.
`MODEL` is the quantized checkpoint and `ADAPTER` a PEFT folder; `OUT` must not exist or must be empty.
For each tensor the adapter names, `(alpha / r) * (B * A)` is added to the weights as `MODEL` decodes them, and each changed value takes the nearest code on its group's own grid: the shipped scales and biases stay, only codes move, and a value the adapter leaves unchanged keeps its code.
`tensorfold merge DENSE ADAPTER OUT --quantize SHIPPED` is the explicit opt-in to quantize fresh: the delta is added to the checkpoint's bf16 source `DENSE` and the result quantized at the shipped tensor's bits and group as above, scales and biases included.
It is kept because it carries far more of the adapter: on a rank-64 Qwen3-4B adapter at group 64, #710 measured a median 99.6% of the adapter landing with fresh quantization against 0.1% with the fold (pooled error 0.981 against 1.000), since a change smaller than half a step leaves a folded code where it was.
The product is accumulated in FP64 in order and scaled once, so a zero sum gives an exactly zero delta; a sum that is not exact in FP64 is not made exact.
Every other tensor, shard, index and `config.json` is copied byte for byte from the shipped checkpoint; with a zero adapter that is the whole checkpoint.
The merge is staged in a sibling folder and renamed to `OUT` only when everything has succeeded, so a failure before publication leaves no `OUT`; the rename replaces an empty `OUT` atomically and never a non-empty one.
A module is merged when the shipped checkpoint holds its `.weight`, `.scales` and `.biases` as rank-2 tensors in the 4-bit affine layout (and, with `--quantize`, `DENSE` holds a rank-2 `.weight` of the same shape), so attention, MLP and linear-attention projections all qualify; the checkpoint decides, not a list of names.
A module may be named in either checkpoint's namespace: the exact name is tried first, then one table of prefix rewrites (`model.language_model.` and `language_model.model.`) in both directions; a module that resolves to no name (or, with `--quantize`, no pair of names present in both folders), or to more than one, is refused naming each one tried.
It refuses, naming the tensor and the reason, anything else: modules not quantized in the shipped checkpoint (norms, embeddings), stacked or rank-3 expert tensors, fused projections, other bit widths, groups other than 32, 64 and 128, DoRA and rsLoRA, shard names that leave the folder, subdirectories, dangling symlinks and special files among its inputs, shards whose tensor ranges overlap, non-finite weights, factors or stored scales, duplicate PEFT keys, and shape mismatches; each before anything is written.
`tf-merge-check MODEL ADAPTER OUT` holds a fold to these gates:
- the exact file set and lengths of `MODEL`, and every byte outside the touched weights, identical (scales and biases included);
- every written group equal, byte for byte, to the fold recomputed from `MODEL` and `ADAPTER`;
- every code within half a step of its target on its group's grid, or at the end of the grid its target lies past: a check by distance, apart from the fold's rounding;
- for an adapter with no effect, byte identity.

`tf-merge-check DENSE ADAPTER OUT SHIPPED` holds a fresh quantization to these gates:
- the exact file set and lengths of `SHIPPED`, and every byte outside the touched tensors, identical;
- every written group equal, byte for byte, to a recomputation from `DENSE` and `ADAPTER`, with the number of groups compared reported;
- aggregate RMS error against an FP64 reference at most twice the shipped quantization's: a sanity net for a gross fault such as a wrong scale or a sign error, not a quality gate, since a delta that widens the groups' ranges legitimately coarsens the grid by a few percent (the ratios, including the worst-column one, are printed);
- for an adapter with no effect (alpha 0, or an all-zero factor, negative zero included), byte identity and ratios of exactly 1.

Each recomputation shares its merge's arithmetic, so it checks that what was written is what the code computed; the independent evidence is the fold's distance check, the pinned vectors and the byte identity of the quantizer with shipped tensors.

## Scope and qualification

In the Python 0.6.6 engine, these additions cover MLX affine safetensors checkpoints for the dense Qwen family. The
native engine's per-module readers are named under Backend paths above; for Nemotron, TensorFold's oQ4, oQ6 and oQ8 are
the measured checkpoints.
GGUF, AWQ, GPTQ, NVFP, MXFP and arbitrary 8-bit MoE expert formats require their own readers and kernels.
Unsupported formats remain explicit errors.

The new paths are experimental until their hardware qualification is complete.
Each backend must pass packed-value checks, one-row versus batched output equality, full-model drafted versus serial equality, prefix resume, concurrent-stream equality and memory admission before a release claim.
In the Python 0.6.6 engine, the Metal kernel has a dedicated hardware test in
[`tests/test_affine_rows_metal.py`](https://github.com/ashhart/TensorFold/blob/v0.6.6/tests/test_affine_rows_metal.py);
model and GPU tests must run under the machine's existing resource controls.
No speed or quality improvement is implied by a higher bit width.

## NVIDIA's NVFP4 checkpoints on CUDA

NVIDIA's ModelOpt NVFP4 checkpoints are made for NVIDIA's own math: NVFP4 activations into the NVFP4 layers and FP8
activations into the FP8 layers, quantized with the input scales stored beside the weights. On CUDA, `--precision`
chooses the math the native engine runs:

- `checkpoint`: that math. Each FP8 layer's input rows become e4m3 at the layer's stored input scale. Each NVFP4
  expert's input rows become NVFP4: an e4m3 scale for every 16 values, from their largest magnitude under the stored
  input scale, and e2m1 values rounded to nearest. Blackwell's FP8 and block-scaled FP4 tensor cores multiply them with
  fp32 sums, and the outputs are bf16. It needs compute capability 12.x (RTX 50, RTX PRO 6000 Blackwell, GB10).
- `full`: bf16 activations into every layer, with the weights as stored (W4A16 and W8A16).

Nemotron 3.5 Super VL runs `checkpoint` by default on Blackwell. Nemotron 3.5 Lightning's NVFP4 checkpoint runs `full`
and refuses `checkpoint` for now. Both modes keep the KV cache in bf16, and the layers the checkpoint stores in bf16
stay bf16.

Each mode is exact on its own terms: drafted replies equal `"draft": false`, resumed requests equal fresh ones,
concurrent streams equal their solo runs, and two GPUs give one GPU's tokens. The two modes give different tokens from
each other. On Super VL on a GB10, `checkpoint` teacher-forced on `full`'s greedy replies to three prompts (768 tokens)
gave a mean KL(full || checkpoint) of 0.0077 nats a token (0.0036, 0.0137 and 0.0057 by prompt), the same top token
at 756 of the 768 positions (98.4%), and a mean negative log-likelihood of the fed tokens 0.0058 higher (0.0692
against 0.0749). `tensorfold kl MODEL --tokens IDS --dump FILE --precision full` writes one precision's logits rows
and `--against FILE --precision checkpoint` runs the other on the same tokens. On the same GB10, `checkpoint` read a
cold 8,192-token prompt in 4.41 s against `full`'s 7.96 s (1.81x; 1.82x at 2,048 tokens), and decoded a token in
52.3 ms against 50.1 ms without drafts: decode reads the same weight bytes either way, and `checkpoint` adds the
rows' quantization.

## Native safetensors metadata

The shared Zig safetensors index retains up to five dimensions, including temporal patch convolutions
such as BF16 `[1024, 3, 2, 14, 14]`. It still rejects ranks above five, invalid dimensions,
overflowing byte counts, and offsets that disagree with tensor size or exceed the data region.
Namespace selection remains available to text loaders that do not interpret a vision tower's layouts.
Reading this metadata does not qualify a vision encoder, checkpoint or device.
