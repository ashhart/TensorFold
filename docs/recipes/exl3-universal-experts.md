# Universal EXL3 routed experts on CUDA (`tensorfold/cuda/exl3/experts.*`)

One grouped trellis GEMV launch per projection runs a whole MoE layer's routed experts, where every expert
matrix may have its own bit width (1 to 8 bits, half-bit widths included, e.g. 3.5) and the layer uses any of
the three EXL3 codebooks (`3inst`, `mcg`, `mul1`) — mixed freely inside one layer, as MiMo-V2.6-Flash's
2.50bpw pack and SAGE packs are, or uniform. It is the structure of GLM-5.3's own expert kernel
(`families/glm5_next/cuda/exl3.cu`, the [EXL3 recipe](exl3.md)) generalized off the 4-bit-`mcg` case.
The accompanying [EXL3 weights](exl3.md) recipe has the format and the dense linear layer.

Measured on MiMo-V2.6-Flash-RL-EXL3 `2.50bpw` (`mul1`, experts at 2/3/4/5 bits), one DGX Spark (GB10),
exclusive GPU, `tools/mimo_exl3_bench.py`, medians of 5 reps x 30 iters x 2 sets, SM clock 2405-2489 MHz
during the run. GB/s is over the distinct experts' trellis bytes actually read (gate + up + down) per call.

| Layer (widths) | Rows | this kernel | CUDA graph | ExLlamaV3 `exl3_moe_mixedk` | ExLlamaV3 `exl3_moe_coop` |
| --- | ---: | ---: | ---: | ---: | ---: |
| L3 (2-bit everywhere) | 1 | 179.6 GB/s | 195.6 | 49.9 GB/s (3.6x slower) | 177.8 GB/s |
| | 8 | 219.6 | 222.0 | 77.8 (2.8x) | 220.2 |
| L30 (2-3 bit) | 1 | 209.7 | 217.2 | 70.5 (3.0x) | not dispatched |
| | 8 | 224.7 | 226.3 | 115.9 (1.9x) | |
| L47 (2-4 bit) | 1 | 204.2 | 209.2 | 86.7 (2.4x) | not dispatched |
| | 8 | 210.3 | 213.5 | 140.0 (1.5x) | |

MiMo's layers are mixed-K, so the path ExLlamaV3 actually runs on them is its per-expert
`exl3_moe_mixedk` dispatch, and this kernel is 1.5x to 3.6x faster per layer. Where a layer is uniform the
two are within 2%: this work does not beat `exl3_moe_coop` on uniform layers, it removes the mixed-K penalty.

## What it is

* Grid `(uids, n_tiles, mats * SK * member_tiles)`, 128 threads (4 warps) per program, `NT` 16x16 trellis
  tiles per program per step, K split `SK` ways for the reduction, `mats` handles gate/up/down of the layer.
* `__exl3x_lane_map<K2>` computes, once per program, the word and shift each lane needs for the `j`-th of
  its 8 values. The 16-bit state of value `p` of a 16x16 tile ends at bit
  `end(p) = (p >> 1) * K2 + (K2 or K2 >> 1)` when `p` is odd/even, so window `w` of the 8-value group
  is a fixed shift of a 64-bit word merge, and a group of `GV` values never spans more than two
  32-bit words (proved by enumeration over every `K2` in 1..16 and asserted at compile time by
  `exl3x_check_gv`, which is why `GV` degrades gracefully: 8 for `K2<=8`, 4 for 9..12, 2 for 13..14).
* Pair decode `cb_pair` is one `half2` op sequence per two values: `3inst` is a multiply-add then a
  mask/xor plus `__hadd2`, `mcg` the same with the other multiplier, `mul1` an `__dp4a` byte sum then
  `__hfma2`. Both halves come out of one 32-bit result, so two values cost ~5 instructions and the
  per-value decode cost does not grow with the bit width. `K2 == 8` keeps the GLM kernel's exact
  shift-based lane decode (`__funnelshift_r` on words `L-1`/`L`), which is why GLM parity is
  bit-exact by construction rather than by argument; every other width goes through the generic path.
* The number of codebook/width specializations is bounded: the kernel is templated on the codebook
  (3 files, `experts_cb{0,1,2}.cu`, so they compile in parallel) and on the tile config, and the bit
  width is a *warp-uniform* branch inside the instance, with only the widths present in the layer's
  `k2` range compiled (`k2=8` only for GLM; `4..8` for a 2..4-bit pack, and so on).
* Weights are read in place: an expert matrix is the checkpoint's trellis tensor
  (`int16 [K/16, N/16, 16 * K]`) reached through a pointer table; only `suh`/`svh` are stacked.
* Nothing accumulates with atomics: the K split partials and the warps' sums go through shared memory in
  a fixed order, so a row's bits do not depend on the window it arrived in, only on its own row.

The Python side is `experts.py`: `prepare` / `prepare_stacked` (per-expert trellis pointers, per-expert
K2, stacked scales), `Scratch` (`P = rows * slots` fp32 `y`/`z`, the pick table), `routed` (device
grouping, the two input rotations, the grouped gate/up, the SwiGLU epilogue, the grouped down, the down
epilogue, the combine, and with weights the epilogue and the combine fused into one launch), and `dequant`
(a debug binding that writes the kernel's decoded weights as fp16, for the ExLlamaV3 comparison).

## Bit-exactness

* `dequant` equals ExLlamaV3's `ext.reconstruct` bit for bit over every codebook and every K2 in
  {1..8, 1.5..7.5} (`tests/cuda/test_exl3_experts.py`).
* On GLM-5.3-shaped synthetic 4-bit `mcg` data (D=1024, I=1024, E=288, R in 1,2,3,4,8,16) the full
  routed pipeline (group + rotations + both GEMVs + both epilogues + combine) is `torch.equal` to GLM's
  `families/glm5_next` `exl3_mm.routed` — same tile settings, same accumulation order. This is a
  synthetic layer: no GLM-5.3-Flash EXL3 checkpoint was available where the kernel was developed.
* Row invariance per [the CUDA recipe book](cuda.md): verified synthetically for mixed 2/3/4-bit `mul1`
  layers over windows of 1, 2, 3, 16, 17, 64 and 128 rows, and on MiMo-V2.6-Flash's real expert tensors
  (a row alone equals the same row in an 8-row window).
* A CUDA graph replaying the layer equals the eager call at 1, 2, 4 and 8 rows on real MiMo layers L3,
  L30 and L47, after the allocator has been churned: `routed` allocates nothing and reads its inputs,
  tables and scratch where the caller keeps them, which is what an engine's graph capture needs.
* The routed output is within 2e-4 (relative, max abs) of a float64 `reconstruct`-based reference.

## Porting it to another runtime

The kernel is self-contained (`experts_grouped.cuh`: `<cuda_fp16.h>`, `mma.h`, a `Q*`/`bits` descriptor
per matrix, `long*` pointer tables and the four group arrays `uids`, `ucount`, `members`, `ids`; no
cuBLAS, no Triton), so a runtime that already holds EXL3 modules could use it with:

1. a small kernel that expands a per-expert histogram into `uids/ucount/members` on the device (its
   per-expert entry points already build such a histogram for their mixed dispatch);
2. scratch: `P = rows * slots` fp32 `y`/`z`, a `rows * slots` int32 pick table, and `[mats, SK, P, N]`
   fp32 accumulators — a few MiB for a 256-expert top-8 layer at 8 rows;
3. no weight preparation: the trellis stays where it is, with a per-expert K2 and `suh`/`svh`;
4. epilogue and combine launches of its own — or the fused `down_combine` here, which is one launch;
5. the same verification the tests above use: `dequant` against `reconstruct` over all codebooks and
   widths, row invariance, and a float64 reference.
