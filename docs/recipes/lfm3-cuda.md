# LFM3-1.7B NVFP4, native CUDA

`lfm3` serves LiquidAI's ModelOpt NVFP4 text export,
[`LiquidAI/LFM3-1.7B-NVFP4`](https://huggingface.co/LiquidAI/LFM3-1.7B-NVFP4) at revision
`b30025bbbbcbf62f6c110b40d51d36af6579f6a2`, on one NVIDIA GPU through the lane host. Checked on a GB10 (sm_121).
The MLX affine and FP8 exports and the VL model are not loaded on CUDA yet, and Metal refuses this export by name
(`Lfm3ModeloptNvfp4NotOnMetal`).

```bash
hf download LiquidAI/LFM3-1.7B-NVFP4 --revision b30025bbbbcbf62f6c110b40d51d36af6579f6a2 \
  --local-dir "$HOME/models/lfm3-1.7b-nvfp4"
zig build native -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/native/bin/tensorfold-native serve "$HOME/models/lfm3-1.7b-nvfp4" --port 8080 --parallel 8
```

## Weight-only

The mode is weight-only. The FFN's FP4 weights (w1, w3, w2, E2M1 in groups of 16 with e4m3 block scales and one fp32
tensor scale) are used as stored, and every activation stays bf16. The export also carries static FP4 input scales;
they are read, because the admission requires them, but not applied. The mixers (conv, q/k/v/gate, o), the embedding
and the tied head are bf16 in the export and run as bf16.

## What runs

- Conv layers (13): Liquid's 7-tap short conv with its rolling state (`liquid_conv.cuh`, shared with LFM2.5).
- Full-attention layers (6): keys and values in pages of the core KV pool, partial RoPE through `qk_rows`' frequency
  table, `pattn_kernel<128, 8, 4, 8>` for prompt rows and the shared paged chunk and merge kernels for decode rows.
- Sliding-attention layers (5, window 256): a ring a stream through the shared window-attention kernels.
- The lane backend, the kept states of the prompt cache and the native glue are shared with LFM2.5-24B-A2B
  (`zig/src/families/liquid/`). Kept states live in pages allocated once inside the prompt-cache budget.
- The default window is 32,768 tokens (`--context`); the model allows 262,144.

## Receipt

GB10, greedy, `--parallel 8`:

- drafted == plain and concurrent == solo on 8 of 8 prompts, prompts of 300 to 2,960 tokens across the window,
  prompt rows == one-row decode at 32 cut points, resumed == fresh, and the token hashes after a restart;
- a float64 forward of the same weights on the CPU (`tools/lfm3/ref64.py`) picks the engine's greedy token on 825 of
  829 teacher-forced prompt rows and 127 of 128 reply rows; `tools/lfm3/REF64.md` lists every miss with both tokens
  and margins;
- decode, total tok/s: 112.8 at 1 stream, 420.0 at 4, 767.4 at 8; cold TTFT 0.150 s at 2k tokens, 0.615 s at 8k,
  2.84 s at 32k.

## Licence

The weights are under Liquid AI's LFM Open License v1.0 (`license: other`, `lfm1.0` on the model card; the text is the
`LICENSE` file in the checkpoint), as the MLX exports are. Use, changes and redistribution are allowed; commercial
use is licensed only to an organisation under US$10 million in annual revenue, and a non-profit may use it for
research or other non-commercial purposes regardless. TensorFold's code stays Apache-2.0.
