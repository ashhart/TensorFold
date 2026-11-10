# dots.mocr on native CUDA

The native CUDA engine serves dots.mocr, its image tower and its Qwen2
decoder, through the shared dense engine (`zig/src/cuda_dense/`), from the
BF16 `dots-studio/dots.mocr` checkpoint at revision
`e539fbb52280393adc081b289ec597430a0f9031`. It is qualified on a GB10
(sm_121) only. The HIP engine is in
[dots.mocr on native HIP](dots-mocr-hip.md).

```sh
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/native/bin/tensorfold-native serve MODEL_DIRECTORY \
  --vision --context 16384 --parallel 8 --name dots --host 127.0.0.1 --port 8090
```

Without `--vision` the server reads text prompts only and refuses images by
name. With it, the default total image budget is 14,400 merged tokens per
request (`--vision-image-tokens` sets another), one tower call an image,
each image with its own attention; the largest image is 57,600 patches.
Each lane holds its request's image rows, so `--vision` lowers the number
of lanes a memory budget admits. JPEG inputs need libjpeg-turbo 3.1.4.1;
PNG needs no library. Video is refused.

Replies carry no reasoning block: all text is answer content. The engine
keeps no prompt states, so `--prompt-cache-gib` is refused; so are
`--drafter`, `--slide`, `--kv-cache fp8` and `--prompt-precision int8`.
The engine's own copy drafts and `--no-drafts` give the same tokens.

| Responsibility | Files |
|---|---|
| Lanes, pages, attention, draws, the host's declarations | `zig/src/cuda_dense/` |
| Geometry, weights, one layer's arithmetic | `zig/src/families/dots_mocr/cuda_recipe.zig`, `cuda_weights.zig` |
| q, k and v biases with the rotary in Torch's rounding | `zig/kernels/cuda/dots_rope.cuh` (`tf_dots_qkv_rope`) |
| Image rows spliced into prompt chunks, per-lane feature rows | `zig/src/cuda_dense/images.zig`, `pass.zig` |
| The tower: patches, 42 blocks, the 2×2 merger | `zig/src/families/dots_mocr/cuda_tower.zig`, `cuda_tower_weights.zig` |
| The tower's 2-D rotary and the twice-rounded patch bias | `zig/kernels/cuda/dots_vision.cuh` (`tf_dots_vision_rope`, `tf_dots_patch_bias`) |

The text reference runs the pinned oracle's `text` mode on any HIP or CUDA
GPU, and `score` classifies every native id by HF's teacher-forced logits
(exact, one BF16 step, or a failure, which exits nonzero):

```sh
python3 tools/zig/dots_mocr_reference.py text MODEL_DIRECTORY REFERENCE \
  --prompt-file PROMPT.txt --pci PCI_ADDRESS
python3 tools/zig/dots_mocr_reference.py score MODEL_DIRECTORY REFERENCE \
  NATIVE_IDS.json DIFFERENCES.json
```

An image reference on CUDA runs the tower with SDPA (the checkpoint's
config names flash attention, which has no aarch64 wheel) and records the
attention backends it used; `score` reads the reference's pixels and grid
for image prompts, and `stages` compares a native tower dump with the
reference's taps, stage by stage:

```sh
python3 tools/zig/dots_mocr_reference.py oracle MODEL_DIRECTORY IMAGE \
  REFERENCE --attention sdpa --pci PCI_ADDRESS [--vision-only]
tf-cuda-test dots-stages MODEL_DIRECTORY IMAGE NATIVE_STAGES
python3 tools/zig/dots_mocr_reference.py stages REFERENCE NATIVE_STAGES
```

`tf-cuda-test dots-image MODEL_DIRECTORY [IMAGE]` encodes a generated page,
and IMAGE when given, twice each, requires equal rows, and checks the page's
merged rows against a hash recorded on the GB10, so a tower change fails
without the oracle. `tf-cuda-test dots-text MODEL_DIRECTORY` does the same
for the decoder: three prompts of our own (`fixtures/own-prompts.json` beside
the family's code) each generate up to 64 greedy ids, whose hash must equal the
one recorded on the GB10. `tf-cuda-test dots-help` lists the dots commands.

The reference pins Pillow 12.2.0 and qwen-vl-utils 0.0.10: Pillow 13
resizes the same image to other pixels.
