# qmm_q4_vec4 pre-registration (written 2026-11-03 before any GPU run of this change)


## Question

Does reading the q4 projection's weights and activations with 16-byte loads (`qmm_q4_vec4`) make
Qwen3.5-2B decode faster on the 13-inch M1, with byte-identical outputs?

## Arms (two ELFs of `tf-qwen35-model`, ReleaseSafe, aarch64)

- old: TensorFold PR #1544 tip `b3ab9ba2` (scalar `qmm_bf16` only).
- new: the same plus commit `e742e023` (`qmm_q4_vec4` chosen when x and w windows are 16-byte aligned).
- Nothing else differs. Same checkpoint, same input (unicode-24, 24 prompt tokens), context 512, 64 free-running
  greedy steps, two runs per process.

## Gates (every process, both arms; any failure stops the unit)

1. Exit 0 and a complete receipt (house receipt check: status ok, oracle ids and hashes, byte-equal repeats).
2. Generated ids equal the pinned ids (hash `6bfbb1a5...`).
3. Full logits bytes of every process equal the first valid old-arm process's logits (one digest for all six).
4. Disk, ICD, checkpoint, input and ELF identity pins hold before the first process starts.

Identity across arms is a gate. A speed number from a run that fails a gate is not reported as a result.

## Metric

Per process, from the probe's own stderr milestones (1 ms resolution):

- primary: run 2 duration = `run2-done` minus `run1-done` (prefill of 24 tokens plus 63 decode steps,
  warm, with the per-step logits readback the probe always does);
- secondary: run 1 duration = `run1-done` minus `engine-ready`.

Order old,new,new,old,old,new (three pairs, both orders). Per pair: gain = (old - new) / old on run 2.
Report the median of the three pairs, the range, and every raw value.

## Decision rule

- Speedup: median pair gain of at least 3 percent AND all three pair gains positive AND all gates hold.
- No speedup: median under 3 percent or any pair not positive. Reported as such; the change stays unmerged.
- Any gate failure: the candidate dies, no speed claim.

## Forecast (a prediction, written before the run; I do not know the answer)

I expect run 2 to shrink by 10 to 35 percent: the profile of the earlier build puts quantized matmul near 80 percent of
decode and each thread issues about six times fewer loads. I give a regression (new slower) a 10 percent chance,
because a wider load can raise register pressure on this compiler.

## Not claimed

No claim about other chips, other quantizations (3, 5, 6, 8 bit and fp modes keep the scalar image), other models,
or the Metal/MLX path. The software-device (llvmpipe) identity pass does not count as GPU evidence.
Two runs of one build on one host are not a significance test.

## Addendum 2026-11-03 (written after the first pair, before the follow-up rows)

First pair result (unicode-24, context 512): all gates held, six logits identical, median run-2 gain 40.09 percent
(pairs 41.0, 40.0, 40.1), rule MET. That was above my 10 to 35 percent forecast.

Follow-up rows use the same two ELFs, the same gates and the same decision rule on three more cases:
a fresh replicate of unicode-24 (host-state variance), document-256 at context 512, and document-1024 at context 1536.
The documents prefill 8 and 32 chunks of 32 rows, so the projections run with m up to 32, a different regime from m = 1 decode.

Forecast, written before these rows: every gate holds (the identity probe already passed on this GPU for m up to 32).
Gains on the document inputs are positive but smaller than 40 percent, because prefill and attention take a larger share of
run time. I give 15 percent to a document-1024 gain below 3 percent. The replicate lands within 5 points of 40 percent.
If any gate fails on any case, the candidate is not mergeable on that evidence, whatever the other cases show.
Each case is judged by the rule on its own; the cases are not pooled.
