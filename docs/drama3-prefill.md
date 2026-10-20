# Drama 3 CUDA prompt chunks

The native speech host advances each prompt by at most `TF_DRAMA_PREFILL_ROWS` rows per shared round. The setting accepts integers from 1 through 1024 and defaults to 128. `drama3-generate --max-rows` selects its engine capacity, from 1 through 8192, also defaulting to 128. The larger engine bound accommodates whole-context oracle scratch; it is not the recommended serving batch. Invalid capacities fail before checkpoint or GPU allocation. Attention query chunks must be from 1 through 1024; their effective size is bounded by the allocated row capacity.

Each prompt starts at offset zero and takes full chunks followed by its remaining rows. Chunk boundaries do not depend on the number of active lanes, request timing or queue order. The last backbone row is retained for decode; the tower consumes only prompt rows with a successor. A shared round still returns to the scheduler after its chunk. Cancellation granularity is therefore a chunk, not a row.

The setting `TF_DRAMA_PREFILL_ROWS=1` restores single-row CUDA prompt arithmetic and scheduling. Metal's native host remains at single-row prefill, and its engine's non-probe prefill is unchanged. Shared default scratch capacities are smaller on both backends.

## Memory and synchronization

The default bounds the per-lane scratch expansion rather than allocating an entire context's prompt workspace. At context 8192, CUDA's row-dependent buffers in `Engine.allocState` contribute 290832 bytes per row, excluding routing-plan storage. With 128 attention query rows, score storage is 67174400 bytes, and row buffers plus scores total 104400896 bytes per lane. At capacity 1024 the same subtotal is 835207168 bytes. These are allocation formulas, not device measurements; weights, KV, recurrent state, routing plans, codec storage, alignment and library workspaces are additional.

`setPos` still synchronizes its asynchronous upload before its stack-backed scalar expires. It is called for each prompt chunk, not each row within that chunk. Attention may also change position between its bounded query subchunks. The host's completion wait remains at the round boundary. No asynchronous upload lifetime or completion guarantee is removed.

## Arithmetic contract

Chunk size is an arithmetic setting, not a promise of identical hidden bits across sizes.

- Dense projections in `cuda/kernels.zig:linear` use `d3_gemv` for one row and cuBLAS for multiple rows. The former accumulates `dot8` groups per lane and performs a warp reduction; cuBLAS chooses a tiled reduction over the same input dimension. Both consume BF16 inputs and produce BF16 outputs with FP32 accumulation, but their grouping and rounding can differ. This affects backbone attention QKV/output, GDN input/output, router/shared-expert projections, and tower embedding/hidden mixing, QKV/output and feed-forward projections. The sampled-token head remains a one-row GEMV.
- Full attention in `cuda/forward.zig:attention` uses split-key online softmax for one row. Its QK dot products, running softmax denominator and weighted-value sums are merged across warps and key splits. Multiple rows use BF16 QK GEMM into FP32 scores, a block-reduced causal softmax, and FP32 PV GEMM. These regroup the QK, exponential-denominator and weighted-value sums. The retained causal keys are unchanged, but GEMM dimensions and reductions can change with query chunk size and the final partial chunk. Both paths round the attention result and sigmoid gate to BF16 before their product. `cuda/blas.zig` requests FP32 compute, not TF32.
- Routed FP8 experts switch from decode tiles to prompt tiles above two rows. `drama3_fp8.cu` preserves the per-output MMA order inside each input scale group and the ordered FP32 `fmaf` accumulation of scaled groups. The tile changes routing and staging, not that output's sum. Per-row quantization and router reductions remain unchanged for identical inputs.
- The causal convolution and GDN recurrence retain their sequential row order and FP32 state arithmetic. Their inputs can nevertheless differ after a changed dense projection. Row-local normalization, rotary preparation and embedding operations do not introduce a cross-row sum.

Consequently, code selection near a logit tie can differ across capacities. Do not use exact cross-capacity frames or PCM as the fidelity gate. Keep the chosen capacity fixed for a request and record it in receipts. No GPU kernel implementation is changed by this patch.

## Qualification

`--check` reads upstream PyTorch oracle `vectors.safetensors`, including prompt layers, tower layers, decoded hidden states and teacher-forced picks. Prompt probes now honor `--max-rows` and use each chunk's actual reference-row offset instead of the reference tail. The existing cosine and greedy-reference checks remain in force. The extra `PREFILL` records carry worst-row cosine and maximum absolute error for each tensor chunk.

Given an existing checkpoint and oracle dump, run both modes against exactly the same vectors:

```sh
"$BIN" --model "$CKPT" --check "$VECTORS" --max-rows 1 2>"$OUT/check-1.log"
"$BIN" --model "$CKPT" --check "$VECTORS" --max-rows 128 2>"$OUT/check-128.log"
python3 tools/zig/drama_prefill_compare.py "$OUT/check-1.log" "$OUT/check-128.log"
```

The comparator requires complete, contiguous coverage of every backbone and tower prompt tensor, both upstream checks passing, no lower worst-row cosine, and no larger maximum absolute error for any tensor. It reports both measurements, without inventing a tolerance for cross-capacity differences. The full oracle-only tower probe still requires scratch for the vector's entire prompt; oracle vectors exceeding the checked allocation capacity are refused.

Run `--replay 1` at each selected capacity to retain the existing exact eager/graph, poison/reset, codec chunking and interleaved-lane checks. Repeat a fixed-seed request with the same capacity and require identical frame and PCM bytes. Compare capacities 64, 128 and the default explicitly; record whether their outputs differ. The default and explicit 128 must have the same partition.

For free-running audio, compare stop reason and length, then word error rate and speaker similarity on the same text and reference. Keep checkpoint and evaluation-model revisions, text normalization and thresholds fixed before the run. Report cross-capacity frame/PCM equality as information, not a rejection criterion. Performance qualification includes first-audio latency, decode rate, peak physical memory and device allocation high-water marks. A host-only test pass does not establish these GPU outcomes.

Host coverage:

```sh
CUDA_VISIBLE_DEVICES= zig build test test-kernel-contract test-golden test-cuda-host check-drama-cuda-host
python3 -m unittest discover -s tools/zig -p test_drama_prefill_compare.py
```
