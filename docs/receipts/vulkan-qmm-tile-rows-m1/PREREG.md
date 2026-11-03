# Four rows a thread for prompt chunks: pre-registration (written 2026-11-03 before any GPU run of this change)

## Question

After batched submission, decode runs near the weight-read floor (about 35 ms a step against about 25 ms), but the long-document runs are mostly prefill:
1,024 prompt tokens take about 13.5 of the 15.7 s. Prefill chunks have up to 32 rows, and `qmm_q4_tile` gives each (row, column) output to its own thread, so each
weight is dequantized again for every row. Does one thread computing four rows of the same column (`qmm_q4_tile_rows`, taken when a chunk has at least four rows)
shorten a run, with byte-identical logits? Each output keeps its own accumulator and its own sum order; the dequantized value is computed once and used four times.

## Arms (two ELFs of `tf-qwen35-model`, ReleaseSafe, aarch64)

- old: the batched-submit build, commit `b7770d64`.
- new: the same plus `qmm_q4_tile_rows`.

## Gates and metric

As before: exit 0, pinned ids, one logits digest per case (unicode-24 `e5447d9a...`, document-256 `147a2d2e...`, document-1024 `7f9d8e65...`), run 2 duration from the
probe's milestones, three pairs in the order old,new,new,old,old,new. The stage 1 probe must show the tiled path equal to the default image on every case, including
row counts of 4, 5, 7 and 33 that leave a remainder.

## Decision rule

Median gain of at least 3 percent and all pairs positive and all gates hold. Otherwise not proposed.

## Forecast (a prediction)

document-1024 gains 20 to 45 percent and document-256 10 to 30 percent, since their time is mostly prefill. unicode-24 (a 24-row prompt, mostly decode) gains under 10 percent.
I give 25 percent to a gain under 3 percent on document-1024 (four row-accumulators plus four activation loads may cost occupancy, as the load-batching variant did) and 10 percent to a
byte difference that fails the identity gate (the compiler may fuse the shared dequantized value differently from the per-row expression).

## Not claimed

Other chips and models. Chunks of fewer than four rows keep the one-row image.
