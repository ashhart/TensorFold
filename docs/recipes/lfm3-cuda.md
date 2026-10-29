# LFM3-1.7B, native CUDA

`lfm3` serves three of LiquidAI's LFM3-1.7B text exports on one NVIDIA GPU through the lane host. Checked on a GB10
(sm_121).

| export | revision | weights on CUDA |
|---|---|---|
| [`LiquidAI/LFM3-1.7B-NVFP4`](https://huggingface.co/LiquidAI/LFM3-1.7B-NVFP4) | `b30025bbbbcbf62f6c110b40d51d36af6579f6a2` | FP4 FFN, bf16 rest, weight-only |
| [`LiquidAI/LFM3-1.7B-FP8`](https://huggingface.co/LiquidAI/LFM3-1.7B-FP8) | `243955da95b1cd68026f293b29de4b4c13f3638f` | block FP8 FFN and q/k/v/o, bf16 rest, weight-only |
| [`LiquidAI/LFM3-1.7B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit) | `d0ca73dcfa70ce18733754a492cdc1fb0123262d` | MLX affine 4 bits in groups of 64, as stored |

The 6- and 8-bit MLX exports and the VL model are not loaded on CUDA yet. Metal refuses the NVFP4 and FP8 exports by
name (`Lfm3ModeloptNvfp4NotOnMetal`, `Lfm3Fp8BlockNotOnMetal`).

```bash
hf download LiquidAI/LFM3-1.7B-NVFP4 --revision b30025bbbbcbf62f6c110b40d51d36af6579f6a2 \
  --local-dir "$HOME/models/lfm3-1.7b-nvfp4"
zig build native -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/native/bin/tensorfold-native serve "$HOME/models/lfm3-1.7b-nvfp4" --port 8080 --parallel 8
```

## Weight-only (NVFP4 and FP8)

Both exports run weight-only. NVFP4: the FFN's FP4 weights (w1, w3, w2, E2M1 in groups of 16 with e4m3 block scales and one fp32
tensor scale) are used as stored, and every activation stays bf16. The export also carries static FP4 input scales;
they are read, because the admission requires them, but not applied. The mixers (conv, q/k/v/gate, o), the embedding
and the tied head are bf16 in the export and run as bf16.

FP8: the FFN and q/k/v/o weights (e4m3 with a bf16 scale per 128 x 128 block) run on the block-FP8 lane, and the
activations stay bf16; the export's dynamic FP8 input quantization (groups of 128) is not applied. The conv
projections, the attention gate, the embedding and the tied head are bf16. The gate is a projection of its own here,
as it is stored bf16 beside the FP8 q, k and v.

## MLX 4-bit

Every projection, the embedding and the tied head are affine 4-bit in groups of 64, used as stored: prompt chunks on
the affine prompt GEMM, decode rows on the affine verify tiles. The embedding's rows are read from the same codes.

## What runs

- Conv layers (13): Liquid's 7-tap short conv with its rolling state (`liquid_conv.cuh`, shared with LFM2.5).
- Full-attention layers (6): keys and values in pages of the core KV pool, partial RoPE through `qk_rows`' frequency
  table, `pattn_kernel<128, 8, 4, 8>` for prompt rows and the shared paged chunk and merge kernels for decode rows.
- Sliding-attention layers (5, window 256): a ring a stream through the shared window-attention kernels.
- The lane backend, the kept states of the prompt cache and the native glue are shared with LFM2.5-24B-A2B
  (`zig/src/families/liquid/`). Kept states live in pages allocated once inside the prompt-cache budget.
- The default window is 32,768 tokens (`--context`); the model allows 262,144.

## Receipt

NVFP4, GB10, greedy, `--parallel 8`:

- drafted == plain and concurrent == solo on 8 of 8 prompts, prompts of 300 to 2,960 tokens across the window,
  prompt rows == one-row decode at 32 cut points, resumed == fresh, and the token hashes after a restart;
- a float64 forward of the same weights on the CPU (`tools/lfm3/ref64.py`) picks the engine's greedy token on 825 of
  829 teacher-forced prompt rows and 127 of 128 reply rows; `tools/lfm3/REF64.md` lists every miss with both tokens
  and margins;
- decode, total tok/s: 112.8 at 1 stream, 420.0 at 4, 767.4 at 8; cold TTFT 0.150 s at 2k tokens, 0.615 s at 8k,
  2.84 s at 32k.

FP8 and MLX-4bit, GB10, greedy, `--parallel 8`, drafts off:

- concurrent == solo: 4 prompts of 64 tokens give the same token hashes alone, 4 at once and 8 at once;
- the float64 forward picks the engine's token on 824 of 829 prompt rows and 128 of 128 reply rows (FP8) and on 825 of
  829 and 128 of 128 (MLX-4bit); every miss is listed in `tools/lfm3/REF64.md`, with float64 margins of 0.004 to 0.085.

## Licence

The weights are under Liquid AI's LFM Open License v1.0 (`license: other`, `lfm1.0` on the model card; the text is the
`LICENSE` file in the checkpoint), as the MLX exports are. Use, changes and redistribution are allowed; commercial
use is licensed only to an organisation under US$10 million in annual revenue, and a non-profit may use it for
research or other non-commercial purposes regardless. TensorFold's code stays Apache-2.0.
