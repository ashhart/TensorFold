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
against the projection kernels the chip runs (`zig/src/families/qwen27/widths.zig`).

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

## Scope and qualification

These additions cover MLX affine safetensors checkpoints for the dense Qwen family.
GGUF, AWQ, GPTQ, NVFP, MXFP and arbitrary 8-bit MoE expert formats require their own readers and kernels.
Unsupported formats remain explicit errors.

The new paths are experimental until their hardware qualification is complete.
Each backend must pass packed-value checks, one-row versus batched output equality, full-model drafted versus serial equality, prefix resume, concurrent-stream equality and memory admission before a release claim.
In the Python 0.6.6 engine, the Metal kernel has a dedicated hardware test in
[`tests/test_affine_rows_metal.py`](https://github.com/ashhart/TensorFold/blob/v0.6.6/tests/test_affine_rows_metal.py);
model and GPU tests must run under the machine's existing resource controls.
No speed or quality improvement is implied by a higher bit width.

## Native safetensors metadata

The shared Zig safetensors index retains up to five dimensions, including temporal patch convolutions
such as BF16 `[1024, 3, 2, 14, 14]`. It still rejects ranks above five, invalid dimensions,
overflowing byte counts, and offsets that disagree with tensor size or exceed the data region.
Namespace selection remains available to text loaders that do not interpret a vision tower's layouts.
Reading this metadata does not qualify a vision encoder, checkpoint or device.
