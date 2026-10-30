# Drama 3 CUDA prompt chunks

The CUDA prompt prefill can run in bounded row chunks instead of one row at a time. It is off by default: the stock path prefills one row a chunk, and its output is byte-identical to the path that shipped before this feature. Batching is opt-in, through two settings.

- The serving host reads `TF_DRAMA_PREFILL_ROWS`, which accepts integers from 1 through 1024 and defaults to 1 (the single-row path). `TF_DRAMA_PREFILL_ROWS=N` prefills N rows a chunk.
- The generation tool takes `--max-rows N`, which selects both the engine's row capacity and the rows a chunk, and defaults to 1. `--max-rows N` prefills N rows a chunk.

On a GB10 with the Drama 3 checkpoint, 128-row chunks brought the first audio from 18.2 s to 0.76 s and the whole utterance from 47.5 s to 30.2 s. That is the gain the opt-in buys; the default pays none of it and keeps the stock bytes.

Metal's native host is untouched: it prefills one row a chunk, and its engine's capacity is unchanged. The shared options keep Metal's own values; only the CUDA path asks for a row count of its own.

## The plan

Each prompt starts at offset zero and takes full chunks followed by its remaining rows. Chunk boundaries do not depend on the number of active lanes, request timing or queue order. The last backbone row is retained for decode; the tower consumes only prompt rows with a successor. A shared round still returns to the scheduler after its chunk. Cancellation granularity is therefore a chunk, not a row.

An empty or oversized capacity is refused before any checkpoint or GPU allocation: more than 8192 rows a chunk is `BadPrefillRows`, and an attention query chunk outside 1 through 1024 is `BadAttentionChunk`. The serving host refuses `TF_DRAMA_PREFILL_ROWS` outside 1 through 1024 by name.

## Memory and synchronization

The bounds below are per lane, at context 8192. CUDA's row-dependent buffers in `Engine.allocState` contribute 290832 bytes per row, excluding routing-plan storage. With 128 attention query rows, score storage is 67174400 bytes, and row buffers plus scores total 104400896 bytes per lane. At capacity 1024 the same subtotal is 835207168 bytes. These are allocation formulas, not device measurements; weights, KV, recurrent state, routing plans, codec storage, alignment and library workspaces are additional.

`setPos` still synchronizes its asynchronous upload before its stack-backed scalar expires. It is called for each prompt chunk, not each row within that chunk. Attention may also change position between its bounded query subchunks. The host's completion wait remains at the round boundary. No asynchronous upload lifetime or completion guarantee is removed.

## Arithmetic contract

Chunk size is an arithmetic setting, not a promise of identical hidden bits across sizes. This matters only where a caller opts in: the default's one-row chunks are the stock path.

- Dense projections in `cuda/kernels.zig:linear` use `d3_gemv` for one row and cuBLAS for multiple rows. The former accumulates `dot8` groups per lane and performs a warp reduction; cuBLAS chooses a tiled reduction over the same input dimension. Both consume BF16 inputs and produce BF16 outputs with FP32 accumulation, but their grouping and rounding can differ. This affects backbone attention QKV/output, GDN input/output, router/shared-expert projections, and tower embedding/hidden mixing, QKV/output and feed-forward projections. The sampled-token head remains a one-row GEMV.
- Full attention in `cuda/forward.zig:attention` uses split-key online softmax for one row. Its QK dot products, running softmax denominator and weighted-value sums are merged across warps and key splits. Multiple rows use BF16 QK GEMM into FP32 scores, a block-reduced causal softmax, and FP32 PV GEMM. These regroup the QK, exponential-denominator and weighted-value sums. The retained causal keys are unchanged, but GEMM dimensions and reductions can change with query chunk size and the final partial chunk. Both paths round the attention result and sigmoid gate to BF16 before their product. `cuda/blas.zig` requests FP32 compute, not TF32.
- Routed FP8 experts switch from decode tiles to prompt tiles above two rows. `drama3_fp8.cu` preserves the per-output MMA order inside each input scale group and the ordered FP32 `fmaf` accumulation of scaled groups. The tile changes routing and staging, not that output's sum. Per-row quantization and router reductions remain unchanged for identical inputs.
- The causal convolution and GDN recurrence retain their sequential row order and FP32 state arithmetic. Their inputs can nevertheless differ after a changed dense projection. Row-local normalization, rotary preparation and embedding operations do not introduce a cross-row sum.

Consequently, code selection near a logit tie can differ across capacities. Do not use exact cross-capacity frames or PCM as the fidelity gate. Keep the chosen capacity fixed for a request and record it in receipts. No GPU kernel implementation is changed by this patch.

## Qualification

`--check` reads upstream PyTorch oracle `vectors.safetensors`, including prompt layers, tower layers, decoded hidden states and teacher-forced picks. Prompt probes honor `--max-rows` and use each chunk's actual reference-row offset instead of the reference tail. The existing cosine and greedy-reference checks remain in force. The extra `PREFILL` records carry worst-row cosine and maximum absolute error for each tensor chunk.

Given an existing checkpoint and oracle dump, run both modes against exactly the same vectors:

```sh
"$BIN" --model "$CKPT" --check "$VECTORS" --max-rows 1 2>"$OUT/check-1.log"
"$BIN" --model "$CKPT" --check "$VECTORS" --max-rows 128 2>"$OUT/check-128.log"
python3 tools/zig/drama_prefill_compare.py "$OUT/check-1.log" "$OUT/check-128.log"
```

The comparator requires complete, contiguous coverage of every backbone and tower prompt tensor, both upstream checks passing, no lower worst-row cosine, and no larger maximum absolute error for any tensor. It reports both measurements, without inventing a tolerance for cross-capacity differences. The full oracle-only tower probe still requires scratch for the vector's entire prompt; oracle vectors exceeding the checked allocation capacity are refused.

Run `--replay 1` at each selected capacity to retain the existing exact eager/graph, poison/reset, codec chunking and interleaved-lane checks. Repeat a fixed-seed request with the same capacity and require identical frame and PCM bytes. Compare capacities 64, 128 and the default explicitly; record whether their outputs differ. The default and explicit 1 must have the same partition.

For free-running audio, compare stop reason and length, then word error rate and speaker similarity on the same text and reference. Keep checkpoint and evaluation-model revisions, text normalization and thresholds fixed before the run. Report cross-capacity frame/PCM equality as information, not a rejection criterion. Performance qualification includes first-audio latency, decode rate, peak physical memory and device allocation high-water marks. A host-only test pass does not establish these GPU outcomes.

Host coverage:

```sh
CUDA_VISIBLE_DEVICES= zig build test test-kernel-contract test-golden test-cuda-host check-drama-cuda-host
zig build test-drama-prefill-host
python3 -m unittest discover -s tools/zig -p test_drama_prefill_compare.py
```

`test-drama-prefill-host` checks the plan itself: that the serving host's default and the generation tool's default both take the stock single-row plan, and that an opt-in capacity plans N-row chunks with a partial tail.
