# glm_moe_dsa prefill toward 1500-1900 tok/s — port plan from MiaAI GLM-5.3-Flash patches (2026-10-01)
Baseline ~700 tok/s @4-32K (TF), 228 @1M (DCP4). Per token @8K ~1.3 ms: MoE 0.39, MLA 0.19, dense 0.175, all-reduce
0.34 (mostly rank-skew wait), other ~0.2. None of Mia's prompt patches are in upstream v0.6.1.
Realistic: ~1300-1500 @8-32K (attention 1248 head-layers/rank vs Flash 352; 78 ARs on 4 ranks). 1900 unlikely.

Ranked:
1. Prompt-expert kernel for any K (Mia 0004 mainloop/epilogue + 0020 order + 0001 ideas) with universal
   load_words<K2>/decode_tile<CB,K2> (mul1 codebook, mixed K2/3/4 per item, per-expert ptr tables, gu_stride), FWHT/suh
   fused into the stage-load prologue (removes exl3_moe_had_in). MoE 3.2 -> ~1.3 s/8K (+25-30%). 1-2 weeks.
   Not reusable as-is: glm5_next exl3.cu is 4-bit + mcg2 + uniform stack; rotate-once (0009a) impossible (suh differs).
2. All-reduce skew + deeper overlap: measure busy per rank (cap state, compaction), N-piece overlap + 0033-style
   front work during exchanges (fused.py:881-943). +15-25%. Do first.
3. One-pass sparse MLA prompt kernel (0009b): no partials/merge, gather prefetch; frees memory (0028). +7-9%.
4. Dense EXL3 prompt GEMM without fp16 materialization (prefill.py:87-89, per half). +5-8%. After 1.
5. Sequence-parallel glue (0010/0033 pattern): RS -> residual+rmsnorm+q_a/kv_a/indexer wq_b on own rows -> AG qa+kva.
   +8-12%. After 2. Then try PROMPT_ROWS 8192.
1M: indexer + DCP 1024-row cap; levers = lift cap (needs 3's memory) + FP8 indexer scores.
N/A to us: 0009a/c, 0024, 0017/0022, 0006 (decode only), HC/KDA parts.

## 10-01 results
- Radix top-k (commit ac40307): exact vs torch.topk+sort, 3.6-7x per 128-row block (1.5 vs 5.4 ms @128K keys);
  4-byte scores on one rank. Expected 128K prefill: top-k ~65 s -> ~15-20 s of 286 s.
- Multi-row indexer scoring (RB rows/program) REJECTED: not faster (0.8-1.2x) and not bit-exact. The scorer is
  tensor-core compute-bound (128 rows x 128K keys x 32 heads x 128 = 137 GFLOP in ~3 ms ~ 45 TFLOP/s). Lever = FP8
  indexer (q/k e4m3, ~2x MMA rate) behind a switch + quality check (DSA/vLLM run the indexer in FP8).
- Dense prompt GEMM: FP8 path REJECTED (Triton fp8 tl.dot on GB10 peaks ~80-84 TFLOP/s = well-tiled fp16; +3.8e-2 err).
  Per-shape fp16 tiles instead (bit-identical): 2048x4096 3.0x, 6144x3072 2.54x, 3072x6144 1.45x, 6144x512 1.39x
  (commit 545e3e3, branch fp8-dense, which also has the UnpackCache). One-pass attention: 32K slower, 128K 1.17x (opt-in).
- 1M/256K profile (10-01 21:00) lost: ppbench client timed out under the profiler (prefill 14 min); rerun with longer
  timeout later (not the current bottleneck).
