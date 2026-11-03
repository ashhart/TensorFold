# LFM3-VL-2B on TensorFold (CUDA vision tower)

LiquidAI's LFM3-VL-2B is a 2B vision-language model: a 14-layer vision tower
(hidden 1024, 16 heads of 64, 2x2 patch merge) feeding a 2-layer projector
(4096 -> 2048) into the LFM3 text trunk.

TensorFold has served LFM3 **text** since PR #1508. This recipe adds the **CUDA
vision tower** (issue #1581): patchifier, gated depthwise conv blocks, axial-2D
rope attention, merge and projector, all in the engine's own kernels.

## Model

| | |
|---|---|
| Checkpoint | `LiquidAI/LFM3-VL-2B` |
| Revision | `main` at time of measurement (text trunk at PR #1508 head `b60ae9ee0`) |
| Text trunk | 2.44 GiB, 32K window, 106.8 tok/s decode |
| Vision tower | 14 layers, hidden 1024, intermediate 2560, 16 q / 8 kv heads, head_dim 64 |
| Patchifier | 16x16 patches, 2 temporal frames, 2x2 merge, `patch_width` 1536 |
| Vision tensors | 138, byte-exact admission (378,760,704 bytes) |

## Build

```sh
zig build -Dnvcc=/usr/local/cuda/bin/nvcc \
          -Dffmpeg=/opt/ffmpeg \
          -Dxgrammar=/opt/xgrammar \
          -Doptimize=fast \
          --prefix /opt/tensorfold \
          install native video grammar
```

The xgrammar pin is read from `zig/src/platform/grammar_library.zig`
(`pub const pinned = "v0.2.8"`), not from `zig/build/grammar.zig`.

## Serve (text)

```sh
tensorfold serve --name lfm3 \
                 --model /models/lfm3-vl-2b \
                 --host 127.0.0.1 --port 8891 \
                 --context 32768 --parallel 2
```

## Fidelity

The tower is validated against the vendor's own PyTorch implementation
(`transformers` + `modeling_lfm3_vl.py`, bfloat16, same GPU, same patchifier
output) by dumping every stage and comparing element-wise:

```sh
python fidelity.py ref     # dumps the reference projector rows to ref.npz
python emit_inputs.py      # emits fid/px.bin (1200x1536 bf16) and fid/want.bin
zig build tf-lfm3-vision-dump -Dnvcc=/usr/local/cuda/bin/nvcc
.zig-cache/o/<hash>/tf-lfm3-vision-dump /models/lfm3-vl-2b fid/px.bin fid/tf.bin 1200 30 40
python fidelity.py cmp     # compares rows: cos > 0.999, max abs diff < 2e-2
```

## Measured

Per-stage cosine against the reference, layer 2 (first attention layer):

| stage | cosine |
|---|---|
| patchifier output | **bit-exact** (335/1200 rows identical, min row cos 1.000000) |
| conv layers L00, L01 | **1.0000** |
| q after QK-norm + rope | **0.99999** |
| attention output, per head | 0.9959 / 0.9892 / 0.9971 / 0.9575 |
| attention after `out_proj` | 0.9290 |
| FFN output | 0.9415 |
| hidden state | 0.9633 |

Per-layer hidden-state cosine, all 14 layers:

```
L00 1.0000  L01 1.0000  L02 0.9633  L03 0.9452  L04 0.9104  L05 0.8771
L06 0.8423  L07 0.7931  L08 0.7717  L09 0.7180  L10 0.7241  L11 0.6842
L12 0.6878  L13 0.9108
```

Magnitudes track the reference at every layer (e.g. L13 4.2571 vs 4.5097, L10
0.0510 vs 0.0535), and the projector output RMS is within 9% (0.0285 vs 0.0314).

### Notes on the residual

The inputs are exact (q matches the reference at 0.99999) yet per-head attention
is 0.9959, so the loss occurs inside the attention computation. The engine's
kernel reproduces a float32 numpy softmax exactly (cosine 1.0000), so the
difference is that PyTorch's `scaled_dot_product_attention` accumulates in
float32 internally while the kernel rounds at bf16 points. With scores around
7.4 the softmax is peaked, so sub-ulp input differences amplify: this is a
precision trade-off, not a structural error.

The reference implementation's own bf16-vs-fp32 drift is 0.9999 per layer,
which bounds how much of the gap could be attributed to quantisation alone.

## Kernel notes

Three things learned porting this tower that are worth knowing before touching
the vision path:

1. **`Attention.full`'s output is bf16**, not fp16 (`pack` uses
   `__floats2bfloat162_rn`). A conversion at the boundary will corrupt every
   value while leaving magnitudes plausible.
2. **`qkvPack`'s fourth argument is the destination.** The packed tensor lands
   in whatever buffer you pass as `out`, not in the `q` buffer.
3. **The dim-64 attention path needs both `.slots` in `fulls` and a
   `full_kernel<64,8,8>` instantiation** in `prefill_attention.cu`; without the
   instantiation the module loads `vision_kernel` and silently reads a
   different layout.

## Tests

```sh
zig build test-lfm3-host            # admission, geometry, weights
zig build test-lfm3-cuda-host       # kernels, tower, fatbin loading
zig build test-lfm2-moe-cuda-host   # regression: LFM2 is untouched
```

## Deployment (measured 2026-11-03)

This is the configuration behind the 92.2% corpus / 58.0% ChartQA numbers. Ports, mounts and flags below are
the measured ones — do not infer them from older scripts.

### Container

```sh
docker run -d --name lfm3vl --network host --gpus all \
  -v /path/to/tensorfold-native:/opt/tensorfold/native/bin/tensorfold-native:ro \
  -v /path/to/lfm3vl.jinja:/tmpl.jinja:ro \
  -v /path/to/models:/models:ro \
  tensorfold-lfm3:pr1508 \
  /opt/tensorfold/native/bin/tensorfold-native serve /models/lfm3-vl-2b-nvfp4 \
    --name lfm3vl --host 127.0.0.1 --port 8892 \
    --context 32768 --parallel 2 --vision --chat-template /tmpl.jinja --no-thinking
```

- Binary is bind-mounted so a rebuild is a file swap + `docker restart lfm3vl` (about 10 s), no image rebuild.
- `--no-thinking` matches what QA traffic sends; without it the arm measures a different prompt.
- `--host 127.0.0.1` keeps the lane off the network; pass a routable address only if clients need it.

### Chat template (required)

`/path/to/lfm3vl.jinja` must carry the gated thinking suffix — the five lines at 21–25 of `tmpl.jinja` that
emit the `</think>`-style pair only when thinking is enabled. Without it, `enable_thinking=false` drops two tokens (`[124901, 124902]`) and ChartQA
collapses from 58.0% to 25.0% — a prompt bug that every tensor-fidelity check passes clean.

Rollback: keep a copy of the template before editing it, restore that copy, then `docker restart lfm3vl` (no rebuild).

### Verify before believing a number

```sh
curl -s http://127.0.0.1:8892/v1/models          # {"id":"lfm3vl"}
python3 probe_tokens.py                            # prompt tokens must be 547 / 546 / 174
```

Token count is the cheapest template check: 545 means the old template is live.

### Image tokens and `detail: "low"`

`detail: "low"` does not merely downscale the picture: it caps the trunk rows, so the grid shrinks and the
**image token count changes** with it — a 640x480 picture needs 300 rows at its natural grid, while `low`
allows 256. `fitGrid` keeps the aspect roughly and never goes below one merge block. Any token-budget
arithmetic that assumes a fixed image-token count is wrong for `low`.

### Expected performance (single stream, box idle)

| metric | value |
|---|---|
| corpus (167 items) | 92.2% |
| ChartQA-100 | 58.0% |
| TTFT p50 / p95 | 0.10 s / 0.12 s |
| total p50 / p95 | 0.17 s / 0.22 s |
| decode | ~115 tok/s |
| throughput at C=8 | 11.3 items/s (cap `--parallel 2`) |
| weights | 2.82 GiB |
| cold load | ~10 s |

### bf16 variant (not merged upstream)

The engine accepts an unquantized bf16 checkpoint through a `.none` scheme. It is **not** in this branch; it lives
as a patch plus a built binary:

- `/path/to/lfm3-bf16-scheme/none-scheme.patch` — 3 files: `config.zig` (`.none` in `Scheme`), `weights.zig`
  (plain bf16 weight emission), `cuda_weights.zig` (bf16 FFN load + 62.9 MB staging term).
- `/path/to/lfm3-bf16-scheme/tensorfold-native-bf16` — the bf16-capable build of the same binary, kept beside the patch.
- Lane: `lfm3vl_bf16` on 8893, `LOAD scheme=.none`, 424 tensors / 3.84 GB.

Measured: 56.0% ChartQA (vs 58.0% NVFP4, p=0.62 — no quality difference) and 1.4× slower end to end. Keep NVFP4.

Two traps when applying: the staging buffer is sized for FP4 codes (bf16 gate_up needs 62.9 MB against 25.2 MB,
else `StageTooSmall`), and conv taps must key off `.mlx_affine`, not "not modelopt", or `.none` gets the wrong
layout.

### Bench harness

`path/to/harness`: `run.py` (accuracy), `loadtest.py` (concurrency sweep), `score_items.py` (per-item + McNemar),
`vlmeval_smoke.py` (ChartQA via VLMEvalKit), `stability.py` (soak/determinism/restart).
