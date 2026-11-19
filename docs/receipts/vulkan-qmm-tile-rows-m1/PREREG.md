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


# Eight rows a thread: pre-registration (written 2026-11-03 before any GPU run of this change)

## Question

The profile of document-1024 (a build that submits every dispatch, so each dispatch carries about 90 us of fixed cost) gave: `qmm_q4_tile_rows` 41.3 percent of the
summed dispatch time (5,952 calls, 1,209 us each), `qmm_q4_tile` 24.2 percent (decode and chunks under four rows), `gdn.prefill_bf16` 8.6, `attention.sdpa_bf16` 7.2, the rest
under 5 each. My forecast for the rows image (at least 70 percent) missed: the fixed cost inflates the small kernels, and the qmm images together are 65.5 percent.
Still, the rows image is five times the next kernel. Does computing eight rows of a column a thread (the dequantized value used for eight accumulators) shorten a run
further than four, with byte-identical logits? Chunks of 8 or more rows take the new image; 4 to 7 rows would keep the four-row image only if I wrote both, so this
experiment replaces four by eight and chunks of fewer than eight rows keep `qmm_q4_tile`.

## Arms (two ELFs of `tf-qwen35-model`, ReleaseSafe, aarch64)

- old: the four-rows build, commit `17d90c5b` (the PR head).
- new: the same with `tile_rows` = 8 and the shader computing eight rows a thread.

## Gates and metric

As before: exit 0, pinned ids, one logits digest per case, run 2 duration from the probe's milestones, three pairs in the order old,new,new,old,old,new.
Cases: document-1024 and document-256 (both are 32-row chunks, the same row count the experiment targets), then unicode-24 only if the first two pass.

## Decision rule

Median gain of at least 3 percent on document-1024 and every pair positive and all gates hold. Otherwise not proposed.

## Forecast (a prediction)

document-1024 gains 5 to 15 percent over four rows: dequantization is shared by twice as many rows, but eight accumulators and eight activation vectors raise register pressure,
as the load-batching variant did. I give 35 percent to a gain under 3 percent and 15 percent to a regression.

## Not claimed

Other chips and models.


# Eight rows a thread, written out without arrays: pre-registration (2026-11-03, before any GPU run of this change)

## Question

The first eight-rows build (arrays `xs[8]`, `acc[8]`, loops over r) was 7.1 percent slower on document-1024 and 4.8 percent slower on document-256 than the four-row
build, with equal logits digests. Two causes fit: eight accumulators and eight activation vectors raise register pressure, or the compiler put the arrays in scratch
memory. The four-row shader has no arrays. This build writes eight rows out the same way (x0..x7, acc0..acc7, same order of sums). If it is still slower, the cause is
register pressure and eight rows is closed. If it is faster than four rows, the arrays were the cause.

## Arms and gates

old: the four-rows build (`17d90c5b`). new: the unrolled eight-rows build. Same pairs, order, gates and digests as `PREREG-rows8.md`. Cases: document-1024, document-256.

## Decision rule

Median gain of at least 3 percent on document-1024, every pair positive, all gates hold. Otherwise not proposed.

## Forecast (a prediction)

60 percent: still slower than four rows (register pressure, same sign as the array build, within 3 points of it). 25 percent: within 3 percent of four rows. 15 percent: 3 percent or more faster.

## Not claimed

Other chips and models.


# Four rows and two columns a thread: pre-registration (2026-11-03, before any GPU run of this change)

## Question

Eight rows a thread lost twice (arrays: 7.1 percent slower on document-1024; written out: 7.6 percent slower), so register pressure, not codegen, limits more rows.
At four rows a weight costs about one dequantization, four multiply-adds and four bf16 unpacks of x: the unpacks are about 40 percent of the arithmetic, and more rows
do not shrink that share. Two columns a thread (columns 16 apart in a tile) unpack each x once for two weights, which cuts the arithmetic per product by about 22 percent
with 8 accumulators but only 4 activation vectors live (eight rows holds 8). The dispatch has ceil(m/4) * ceil(n/32) * 16 threads. Each output keeps the same
sum in the same order, so the logits must stay byte-identical.

## Arms and gates

old: the four-rows build (`17d90c5b` kernel; the merged PR head has the same kernel). new: four rows and two columns a thread. Same pairs, order, gates and digests as
`PREREG-rows8.md`. Cases: document-1024, document-256.

## Decision rule

Median gain of at least 3 percent on document-1024, every pair positive, all gates hold. Otherwise not proposed.

## Forecast (a prediction)

40 percent: 3 to 12 percent faster. 35 percent: within 3 percent of four rows. 25 percent: slower (the eight accumulators and two weight vectors still spill).

## Not claimed

Other chips and models; decode (this image serves chunks of 4 or more rows only).


# Rows image with two columns a thread against the batched build: pre-registration (2026-11-03, before these three runs)

## Question

#2017's kernel (`qmm_q4_tile_rows`, four rows a thread) measured 4.1, 24.8 and 32.4 percent less time than the batched build on unicode-24, document-256 and
document-1024. With two columns a thread added (commit `9eef5a52`), the same image measured 5.7 percent (document-256) and 7.6 percent (document-1024) less time
than the four-rows build in two-arm pairs, with equal digests. What is the total against the batched build, on the three inputs of the earlier PRs?

## Arms and gates

old: the batched-submission build (#2016). new: the build at `9eef5a52`. Same pairs, order, gates and digests as `PREREG-rows.md`: exit 0, pinned ids, one logits digest
per case across all six processes (`e5447d9a3ab5`, `147a2d2e0232`, `7f9d8e65e90e`), run 2 duration, pairs alternated old,new,new,old,old,new.

## Forecast (derived from the numbers above, a prediction)

unicode-24: 5 to 10 percent less time. document-256: 29 to 33 percent. document-1024: 38 to 42 percent.

## Not claimed

Other chips and models; decode (this image serves chunks of 4 or more rows only).
