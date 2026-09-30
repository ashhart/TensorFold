# GLM visible-pool selection measurements

Candidate retains the power-of-two allocation buckets and RB=1. Pool-score
programs beyond the last visible pool store -inf without dot products. Radix
selection scans only max(512, complete visible pools) per row. Keeping at least
512 entries preserves the previous lower-index -inf ties at the dense boundary.
No new allocation sizes, host position reads or graph keys are introduced.

## GB10 kernels

Generated tensors at GLM per-rank shapes: 32 index heads, width 128; latent
attention 32 heads, width 512. A full model remained resident during these
microbenchmarks. CUDA graphs replayed with changing device positions.

At position 205000, complete select_tokens graph replay (ms):

| Rows | Original | Candidate | Reduction | Sparse latent attention |
|---:|---:|---:|---:|---:|
| 1 | 0.4493 | 0.3636 | 19.1% | 0.0613 |
| 6 | 0.4611 | 0.3669 | 20.4% | 0.1380 |
| 16 | 0.5771 | 0.4577 | 20.7% | 0.2544 |

Selection still costs more than sparse attention; this does not solve all DSA
cost. Skipping dot products alone reduced complete selection much less; skipping
the radix padding is necessary. Larger RB did not consistently improve timings,
so production uses RB=1.

Baseline/candidate tokens, counts and latent attention outputs were bit-equal
at positions 2049, 16383, 199999, 205000, for 1/6/16 rows, eager and graph replay,
including negative device-position offsets. 60 earlier score comparisons also
matched bitwise. The harness requires an original sparse.py via POOL_REFERENCE.

CUDA tests: 37 existing GLM kernel/latent tests passed, plus 10 visible radix
selection cases covering random scores, all ties, zero pools and graph capture.
Host suite before the radix change: 3160 passed, 738 skipped. CPU cannot execute
these CUDA-only changes; the final candidate was verified by the CUDA tests.

## Full GLM EXL3, two actual GB10 ranks

Same checkpoint and local production performance patches as the decisions
validation deployment; only sparse.py was replaced on both ranks for this test.
Context 200000, drafts loaded, greedy requests, thinking off. Three repeated
requests for each of two prompts, baseline then candidate with a restart:

- 3616 prompt tokens: all three choices identical before/after. Baseline times
  7.787/9.280/7.569 s; candidate 12.434/7.526/7.540 s. First candidate request
  includes cold/JIT work: no reliable speed claim from this small sample.
- 18016 prompt tokens: all three choices identical before/after. Baseline
  33.593/34.083/33.342 s; candidate 32.874/32.914/32.907 s. Median reduction
  about 2.0%, not a controlled throughput result.

These prompts exercise sparse selection but not the 205k endpoint. A full-model
long-context decode/profile and speculative-policy matrix remain unmeasured.
The kernel timings at 205k are not full-model timings at that context.

## Qwen3.8-27B / M4 Max

Qwen does not use GLM's CUDA DSA pool selector. MLX paths are unchanged as well.
A full Vontra/Qwen3.8-27B-MLX-4bit server on the candidate branch, context 1024,
drafts off, returned HELLO, 4 for 2+2, and Dzień dobry for good morning; each
request repeated twice with identical choices. This is a smoke test, not an
acceleration claim or baseline/candidate numerical parity test. Server stopped.
Full GLM on the 48 GB Mac was not loaded in this run.
