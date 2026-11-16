# LFM2.5-24B-A2B, native Metal and CUDA

This recipe covers [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit)
at revision `667408d494bde191c2cb9dd4f4549827816b9275`. The numbers below are measured on this 4-bit export only.

```bash
hf download LiquidAI/LFM2.5-24B-A2B-MLX-4bit \
  --revision 667408d494bde191c2cb9dd4f4549827816b9275 --local-dir "$HOME/models/lfm2.5-24b-a2b-4bit"
zig build native -Dcpu=apple_m1
zig-out/native/bin/tensorfold-native serve "$HOME/models/lfm2.5-24b-a2b-4bit" --port 8080 --parallel 16
```

A Mac with less memory may not hold 16 streams at the checkpoint's 131,072-token window; the server then names how
many fit. `--context 40960` holds 16 on a 48 GB M4 Pro.

## Shared rounds

Every stream's tokens go through the lane rounds. A round of several streams is one forward over all their rows.
In it each layer's short conv (30 layers) and each attention layer's qk, attention and merge (10 layers) run once
over every row (`lfm2_segments.metal`), from a table of each stream's cache addresses, row offsets and key counts
(`segments.zig`). Each row's arithmetic is the per-stream kernel's, so a reply equals its solo run. Prompt calls and
one-stream windows keep the per-stream launches. A greedy row takes the GPU's pick of its logits (`lfm2_argmax`);
sampled and masked rows draw on the host. The dense twin takes its MMA window only for matrices of 4,096 outputs or
more (the short conv's input and the head); narrower ones keep the scalar passes.

`zig build test-lfm2` runs the family's tests. `tf-kernel-contract --only lfm2` checks the shared-round kernels and
the pick, and `tf-lfm2-exact` compares windows and shared rounds with one-row decode on the real weights.

## Measured

Greedy, 256 tokens, 16 public prompts, a different one a stream, 3 reps in each of 3 alternated rounds on an M5 Ultra;
aggregate tok/s, the range of the rounds' medians. Before is main at 0d24ecb3 (each stream's own launches), after adds
the shared-round kernels on it. The M4 Pro is not yet measured on a quiet machine.

| Streams | before | after |
| --- | --- | --- |
| 1 | 258.5-259.2 | 266.5-267.7 |
| 4 | 483.5-486.5 | 565.2-566.1 |
| 8 | 551.5-551.9 | 732.2-735.6 |
| 16 | 660.7-661.9 | 922.5-923.5 |

Every reply was token for token equal before and after, and equal to its solo run and to `"draft": false`.
Cold prefill at 2k, 8k and 32k tokens and peak memory did not change.

## CUDA

`lfm2_moe` also serves the 4-bit export above (same revision, `667408d494bde191c2cb9dd4f4549827816b9275`) on one
NVIDIA GPU, through the lane host. Checked on a GB10 (sm_121).

```bash
zig build native -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/native/bin/tensorfold-native serve "$HOME/models/lfm2.5-24b-a2b-4bit" --port 8080 --parallel 8
```

The default window is 32,768 tokens (`--context`), and `--parallel` holds up to 128 streams if the memory budget fits
them. Streams lease 512-position pages of keys and values from one pool as they need them
(`TENSORFOLD_LFM2_KV_PAGES=0` gives each stream fixed pages instead). The prompt cache keeps a chat's state at its
history marks, the conv states and the K/V rows before the mark, in pages allocated once inside its budget
(`--prompt-cache-gib`). The family reads MLX affine 4-bit weights in groups of
64, with the embedding at 4, 6 or 8 bits. The 6- and 8-bit exports are refused by name and stay on Metal. Drafts
are the lane core's copy drafts from the prompt's own text; there is no draft head yet.

Most launches are shared CUDA operators: the affine linears and experts, the norms, `qk_rows`, the MoE combine, the
pick and the draws. The family adds three kernels: the router (fp32 weights, sigmoid plus bias, top-4) in
`lfm2_moe.cu`, and the 3-tap short conv and its state commit in `liquid_conv.cuh`. It also adds two instances of shared
templates at head dim 64: `pattn_kernel<64, 8, 4, 8>` for prompt rows, and `tf_paged_attn_chunk_segs_64` with
`tf_paged_attn_merge_segs_64` (`paged_attn_segs.cuh`) for decode rows. All of them are under the kernel contract
(see [the contract](../kernel-contract.md#coverage)).

Receipt (GB10, nvcc 13.0, `--parallel 8`, greedy):

- drafted == plain and concurrent == solo on 8 of 8 prompts, and the token hashes are the same after a restart;
- a float64 forward of the same weights on the CPU (`tools/lfm2_moe/ref64.py`) picks the engine's greedy token on
  421 of 435 teacher-forced prompt rows and 123 of 128 reply rows; `tools/lfm2_moe/REF64.md` lists every miss with
  both tokens and margins;
- decode, total tok/s: 87 at 1 stream, 194 at 4, 303 at 8;
- cold TTFT: 0.32 s at 2.8k prompt tokens, 1.32 s at 11.2k.

## Licence

The weights are under Liquid AI's LFM Open License v1.0 (`license: other`, `lfm1.0` on the model card; the text is the
`LICENSE` file in the checkpoint). Use, changes and redistribution are allowed; commercial use is licensed only to
an organisation under US$10 million in annual revenue, and a non-profit may use it for research or other
non-commercial purposes regardless. Read the licence before serving it commercially.
