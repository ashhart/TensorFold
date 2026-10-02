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
- 10-01 SHA isolation: 32K sampled replies are NOT reproducible run-to-run with the default prefill reduce (NCCL bf16
  ring in the 2-micro-batch overlap; RADIX=0 run matched neither the radix run nor pre-rebase). Bit-identity tests must
  use TF_GLM53_PREFILL_REDUCE=rs + TF_GLM53_PROMPT_OVERLAP=0 (the exact rank-order reduce; upstream's planned default).
  Radix top-k / UnpackCache / tuned tiles are bit-exact at kernel level; the any-K MoE prompt kernel (e36a1ea) is not
  (rel 8e-4 vs routed) - prompt-path bits per configuration.
- 10-01 22:20 prefill-all (radix + UnpackCache + tuned tiles + any-K MoE prompt kernel ON) on 4 Sparks, DCP1, ft7:
  prefill 4K 840 / 8K 870 / 32K 892 / 128K 675 tok/s (TTFT 4.9 / 9.4 / 36.5 / 193 s) vs baseline 726 / ~773 / 717 / 497.
  Needles 32K + 128K PASS (needle.py --max-tokens 512; the 32-token default truncates the model's in-content reasoning).
  Decode @32K unchanged (prose 25.9-27.3, code 30.7-34.4). Down-projection tuning: register epilogue rejected (slower),
  fp16 accumulation + item ordering added as opt-in (~0.3-0.8 ms / ~1%).

## 10-02 sequence-parallel prompt glue (item 5; branch seqpar-glue, opt-in TF_GLM53_PROMPT_SP=1)
- fused.compute_prompt_sp: each half's rows split in `world` contiguous blocks (padded, zero rows). Every all-reduce ->
  reduce-scatter over rows (PREFILL_REDUCE=ring: NCCL bf16 ncclReduceScatter; rs: all-to-all of fp32 row blocks +
  rank-order sum = the exact reduce, now half the bytes of the non-SP exact path). On own rows only: residual add,
  both RMSNorms, q_a, kv_a, q_a norm, indexer wk / weights_proj / wq_b + rope, the indexer top-k
  (TF_GLM53_SP_SELECT=1, default; 0 = gather iw and select every row), router + expert top-k. All-gathers (in place,
  one NCCL group): qn|kva|ik(|iw) -> every rank writes every row's latent / index key; tok (full-indexer layers);
  post-attention normed x|pick|wts for the experts / dense MLP. Final hidden (and DFlash taps) gathered once.
- Streams: main runs only every-row work (q_b..o_proj, experts / MLP); a half's RS -> own-row glue -> AG chain runs on
  the comm stream under the other half's every-row work (chains in issue order: half A's keys land before B selects).
- Accuracy (4-layer subset, threadcomm, 2048-row chunks, prompts 1000/3000/4500, vs fused.compute exact): hidden rel
  1.3e-3..5.6e-3 (ar-ring today 5.2e-3..5.6e-3), first-token logits rel 4.2e-3..9.7e-3 (today 4.7e-3..9.6e-3), greedy
  first tokens identical; SP exact bit-identical run to run and on all 4 ranks.
- One rank's GPU time (solo rank 0 of 4, collectives = local copies), 32K prompt: MoE layers (6,3,4,5 = 1 full
  indexer in 4, like the body) 2048-row chunks 3450 -> 3028 ms ring / 2944 exact (-12 / -15%); 4096-row chunks 3347 ->
  2981 / 2902 (-11 / -13%); last chunk (30K keys) 434 -> 379 / 372 ms. Layers 0-3 (3 full indexers): -23%. Replicated
  selection (SP_SELECT=0) gives back most of the long-context win.
- Bytes on the wire a row and layer (per rank): AR ring bf16 36.9 KB; SP ring 33.6 KB (RS 2 x 9.2, AG qn/kva/ik 4.3,
  tok 6.1 on 1 layer in 4, normed+picks 9.3); SP exact 52 KB (fp32 RS) vs non-SP exact 73.7 KB. NCCL launches a layer
  and half: 2 -> 4-5 (grouped AGs).
