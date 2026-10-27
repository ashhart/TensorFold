# Muse Spark Open

The `muse_spark` family serves an EXL3 pack of Muse Spark Open's text model (exllamav3's `muse_spark` module, mul1
codebook) on the native CUDA server, on one GPU or two DGX Sparks as tensor-parallel ranks, with RadixArk's DFlash2
drafter as an option. Package: `zig/src/families/muse_spark/`, kernels in `zig/kernels/cuda/muse_spark/`. There is
no Python family for this model, so the kernels are hand-written CUDA and the parity reference is exllamav3's forward.

## The model

- 62 layers of hidden size 8,192: 16 full-attention layers (no RoPE) and 46 sliding-window layers (2,047 keys, RoPE).
- Attention: head dim 64, 128 query heads and 16 KV heads, a sigmoid output gate per attention, gated residuals.
- MoE: a latent MoE (8,192 to 4,096 before the experts and back after), 8 routed experts a token from a sigmoid
  router with a score bias, a norm on every expert's output.
- Head: 202,048 tokens (padded to 202,112 rows), softcapped.
- `config.zig` refuses configs the kernels do not handle (dense MLPs, shared experts, attention bias, tied
  embeddings, head dim other than 64, GQA other than 8:1, non-default RoPE, codebooks other than mul1).

## Checkpoints

| | Repository | Revision |
|---|---|---|
| Pack (EXL3 mul1, 1.8 bpw) | `ajensenwaud/Muse-Spark-Open-816B-A42B-EXL3-1.8bpw` | `38ad2925945840c3a677bd3f213867d30368636c` (weights as of `efcdf2ec4222c4597e291f1b07edd19c3c02c652`; later commits change the card only) |
| Pack source | `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` | `4885eb2a901c2638854ef35cf6833e6447622afb` |
| Drafter source (DFlash2, bf16) | `RadixArk/Muse-Spark-Open-DFlash2` | `46c90f412549c472941a68382d91a99e5e6372a0` |
| Drafter pack (EXL3 4.0 bpw) | built from the drafter source by Muspark's `scripts/quantise_drafter.sh` | exllamav3 `058b8630fd081f3ff46fce6246955b157c3a2e3b` plus Muspark's `patches-exl3/` |

Licences: Muse Spark Open (`meta-models/Muse-Spark-Open-816B-A42B`), AMD's MXFP4 export and the EXL3 pack are under
Meta's Muse Model License and its Usage Policy (the pack ships it as `LICENSE.md`; it asks for "Built with Muse Spark
Open" and a separate agreement for large model-as-a-service use). RadixArk's DFlash2 drafter states no licence on its
card or in its repository, so no drafter pack is published: it is built locally from the original. exllamav3, the
quantiser and the parity reference, is MIT (Copyright (c) 2025 Turboderp).

## Files

| File | What it does |
|---|---|
| `config.zig`, `load.zig`, `pack.zig` | Config, a rank's TP cuts, the pack's load (EXL3 strips, norms, gates, router) |
| `exl3.zig` | EXL3 layers on the shared `cuda.exl3` linear and `cuda.exl3_experts` operators |
| `forward.zig`, `ops.zig` | The decode forward for windows of rows over any slots and positions, eager or graphed |
| `prompt.zig` | The prompt path: 1,024-row chunks, experts in 512-row windows, the head on the last row |
| `engine.zig`, `native.zig` | The model and TP session, the plan link's words, the lanes backend behind `serve` |
| `kept.zig` | The prompt cache's kept states (`snap_*`): full-layer rows pinned in the slot, rings copied |
| `dflash2*.zig` | The DFlash2 drafter, split over the ranks like the target |
| `m1_main.zig`, `draft_main.zig`, `prompt_main.zig`, `pair.zig` | `tf-muse`: checks and `follow` (rank 1) |

Kernels: the EXL3 projections and routed experts run on the shared operators (`exl3_linear.cu`, `exl3_experts.cu`).
The family's own kernels: `attention.cu` (head dim 64 decode attention), `glue.cu` (norms, gated residuals, router,
top-k, expert norms and sums, softcap, argmax, RoPE tables), `linear_multi.cu` (q/k/v/gate in one launch and 32-row
passes over `exl3_decode.cuh`, the shared linear's bits), `experts2.cu` (32-member tiles of the shared grouped GEMM,
its bits; prompt windows), `dflash.cu` and `dflash2.cu` (the drafter). Every launch is registered for the kernel
contract's gate 5 (`core/contract/muse_cuda_records.zig`, sm_121); `muse_rms`, the linear forms and the 32-member
tiles also run gates 1-4 (`tests/cuda/contract_muse*.zig`), the last two as bit-equal replacements of the shared kernels.

A row's bits do not depend on its window, its neighbours, the graph bucket or how a prompt is cut into chunks, so
graphed decode equals eager decode, the prompt path equals decode rows, and a resumed prompt equals a fresh one.

## Serving

- One GPU: `tensorfold-native serve --model PACK`.
- Two Sparks: rank 1 runs `tf-muse follow PACK [DRAFTER]` with `TF_TP_WORLD=2 TF_TP_RANK=1 TF_TP_MASTER=<rank 0>
  TF_TP_PORT=<port>`; then rank 0 runs `serve` with the same `TF_TP_*` and `TF_TP_RANK=0`. `TF_COMM_BACKEND`
  picks `roce` (the default) or `nccl` for the gathers; both give the same bytes.
- `--drafter DIR`: the DFlash2 pack (drafts stay on unless the server turns them off). Rank 1 takes the same path as
  its second argument; without one it looks in `PACK/dflash2`.
- `--prompt-cache-gib N`: the prompt cache's budget a rank (`0` turns it off; 1 when not given). A kept state costs
  about 0.2 GiB a rank whatever its length.
- Memory: at the 1.8 bpw pack a rank holds about 108 GiB serving (measured on two GB10s with a 24K-token prompt
  running: 108.8 GiB a process, 5.5-5.9 GiB MemAvailable left a node). The server's default reserve, a tenth of
  unified memory (12.2 GiB), leaves no room for the caches beside it, so give the reserve explicitly, e.g.
  `TENSORFOLD_MEMORY_RESERVE_GIB=3` on both ranks (the Muspark recipe's setting).
- The KV cache is fp16.
- The embedding stays in host memory (rows are copied from the pack's mapping each window), the load uses 64 MiB
  arena blocks, and each layer's pages leave the page cache once uploaded.

## Checks (`tf-muse`)

- `tf-muse m1 MODEL IN_DIR OUT_DIR`: teacher-forced decode rows over each text's ids (`IN_DIR/names.txt`,
  `<name>.ids` int32), then greedy rows; writes `<name>.logits.f32` and `<name>.greedy.i32`. Knobs: `MUSE_ROWS`,
  `MUSE_SLOTS`, `MUSE_LAYERS`, `MUSE_CONTEXT`, `MUSE_GREEDY`, `MUSE_CAPTURE`, `MUSE_TEACHER`, `MUSE_GRAPHS`,
  `MUSE_PAIR=1` (both ranks on one GPU). At TP=2 set `TF_TP_*` on both nodes; rank 0 writes the outputs.
- `tf-muse prompt MODEL IN_DIR OUT_DIR`: each chunk plan of `MUSE_PROMPT_PLANS` against decode rows, byte for byte.
- `tf-muse dflash|drafted MODEL IN_DIR OUT_DIR`: the drafter's fixed protocol and drafted == plain greedy
  (`TF_MUSE_DRAFTER` names the drafter pack, `TF_MUSE_DRAFT_TAPS` overrides its taps on truncated packs).
- Graphs against eager: `m1` with `MUSE_GRAPHS=1` against an eager run; the logits files must be byte-equal.
