# GLM prompt K policy experiment

Fresh prompt batching credits @BobClawblaw's #1014. Its default remains one K slice at every prompt row count.
This separate experiment selects K slices from the weight's N and K dimensions, independent of the number of
prompts, their row count or their position in a batch. It uses the existing tiled kernels in windows of at most
64 rows and the existing ordered fp32 slice reduction. Verify and draft windows keep their previous policy.

Select `TF_GLM_PROMPT_K=shape` on every rank to try it. An absent value or `one` keeps the default.
The experiment requires `TF_GLM_MM=1`; the untiled combination is refused. The selected policy is included in rank
agreement, so different choices refuse before loading weights. Applying this patch does not enable the experiment.

No speed or device qualification is claimed. The different reduction grouping can change near-ties even though
precision is unchanged. Keep the current numerical and quality bars. The default must remain unchanged until
matched device evidence supports another decision.

## Device checks

Run the current kernel contract and measured-resource gate first. This experiment changes no kernel source, ABI,
thread count or shared-memory specialization. Keep the existing GLM launch records; pending resources still need
measurement on the device being qualified.

Run `tf-cuda-test glm-mm-fixed` on a GB10. It compares the default and experimental policies for bf16 weights and
4-, 6- and 8-bit weights, with fp32 and bf16 outputs. Every input row runs alone, then in batches at offsets 0, 1 and
7, across row buckets and partial tails through 129 rows. Input stride includes padding; output prefix and tail
bytes are guarded. Two matrix shapes exercise shape-selected K splits. Results are checked bitwise against each
row alone and an independent fp64 product on exact dyadic fixtures. Packed fixtures use independently assembled
codes and fp16 scales. These synthetic checks do not establish full-checkpoint numerical fidelity.

The shader-free `test-glm-mm-policy-host` checks actual dispatch with recording CUDA calls. The separate
`compile-glm-prompt-gate-host` code-generates the actual device gate with inert images, without executing it.
Neither replaces the device run. The ordinary `compile-cuda-tests` target also includes the gate in the runner.

Then run the existing GLM layer/operator references at all selected checkpoint widths, and the served greedy and
seeded checks: drafted/plain, fresh/resumed and concurrent/solo. Keep prompt caching enabled for resumes. Include
MTP and DFlash2, unequal prompt lengths, several 2048-row passes, an already-live stream, cancelled admissions and
image/resumed prompts alongside fresh text. Repeat both-rank gather/split/ring cases.

## Timing

Use one pinned binary, checkpoint, precision, context, prompt set and cache state. Compare `one` and `shape` on both
ranks in matched A/B/B/A runs with one, eight and sixteen streams. Record prompt time, first-token latency, decode
throughput and whole-run throughput separately. Keep drafting and cache settings equal. Repeat structured-output
and quality checks, including the original IFEval/JSON comparisons. A faster isolated matmul is insufficient to
resolve the reported lone-stream regression or justify a default change.
