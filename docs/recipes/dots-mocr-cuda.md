# dots.mocr on native CUDA

The native CUDA engine serves dots.mocr's Qwen2 decoder for text prompts
through the shared dense engine (`zig/src/cuda_dense/`), from the BF16
`dots-studio/dots.mocr` checkpoint at revision
`e539fbb52280393adc081b289ec597430a0f9031`. It is qualified on a GB10
(sm_121) only. Images are not served on CUDA yet: `--vision` is refused
by name. The HIP engine, with the image tower, is in
[dots.mocr on native HIP](dots-mocr-hip.md).

```sh
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/native/bin/tensorfold-native serve MODEL_DIRECTORY \
  --context 8192 --parallel 8 --name dots --host 127.0.0.1 --port 8090
```

Replies carry no reasoning block: all text is answer content. The engine
keeps no prompt states, so `--prompt-cache-gib` is refused; so are
`--drafter`, `--slide`, `--kv-cache fp8` and `--prompt-precision int8`.
The engine's own copy drafts and `--no-drafts` give the same tokens.

| Responsibility | Files |
|---|---|
| Lanes, pages, attention, draws, the host's declarations | `zig/src/cuda_dense/` |
| Geometry, weights, one layer's arithmetic | `zig/src/families/dots_mocr/cuda_recipe.zig`, `cuda_weights.zig` |
| q, k and v biases with the rotary in Torch's rounding | `zig/kernels/cuda/dots_rope.cuh` (`tf_dots_qkv_rope`) |

The text reference runs the pinned oracle's `text` mode on any HIP or CUDA
GPU, and `score` classifies every native id by HF's teacher-forced logits
(exact, one BF16 step, or a failure, which exits nonzero):

```sh
python3 tools/zig/dots_mocr_reference.py text MODEL_DIRECTORY REFERENCE \
  --prompt-file PROMPT.txt --pci PCI_ADDRESS
python3 tools/zig/dots_mocr_reference.py score MODEL_DIRECTORY REFERENCE \
  NATIVE_IDS.json DIFFERENCES.json
```
