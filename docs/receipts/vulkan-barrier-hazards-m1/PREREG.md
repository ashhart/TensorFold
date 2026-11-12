# Compute-to-compute dispatch barrier: pre-registration (2026-11-03, before any GPU run of this change)

## Question

The decode microbenchmark of `qmm_q4_tile` on the real decode shapes (400 dispatches, batched, 13-inch M1) gave 23.85 ms a decode step with the engine's barrier after every
dispatch and 16.99 ms with no barrier (a 16-column dispatch: 80 us against 3 us). The barrier costs about 70 us a dispatch. The engine's barrier lists the transfer stage and transfer
access among its destinations, which a dispatch followed by a dispatch does not need (copies and fills carry their own barriers in both directions). Does narrowing it to compute to compute
(source: shader write; destination: shader read and write) cut the cost? Nothing else changes, so the logits must stay byte-identical.

## Arms and gates

old: the two-columns build (`9eef5a52`). new: the same with the narrower barrier in `Pipeline.record`. Same pairs, order, gates and digests as `PREREG-rows.md`; all three cases.

## Decision rule

Median gain of at least 3 percent on unicode-24, every pair positive, all gates hold. Otherwise not proposed.

## Forecast (a prediction)

50 percent: no gain (under 3 percent; the driver already treats the barrier alike). 35 percent: 3 to 15 percent less time on unicode-24. 15 percent: more than 15 percent.

## Not claimed

Other chips and models. Barriers between dependent dispatches remain; this changes their kind, not their number.


# Hazard-tracked barriers: pre-registration (2026-11-03, before any GPU run of this change)

## Question

The decode microbenchmark on the 13-inch M1 (Honeykrisp Mesa 26.3.0-devel git-9323171c12, real Qwen3.5-2B decode shapes, 175 `qmm_q4_tile` dispatches a step, batched) gave
23.85 ms a decode step with the engine's barrier after every dispatch and 16.99 ms with none; a 16-column dispatch took 80 us against 3 us. A barrier narrowed to compute-to-compute
changed nothing (-0.14 percent on unicode-24), so the cost is the barrier, not its masks. The engine records a barrier after every dispatch, but many dispatches do not depend on the one
before (the q, k and v projections; the gate and up projections; the four GDN input projections). This build tracks the byte ranges every dispatch since the last barrier read and wrote
(each binding's buffer, offset, length and the pipeline's written-bindings mask) and records a barrier only before a dispatch that reads what an earlier one wrote, or writes what one read
or wrote. A full tracking list forces a barrier. Same dispatches in the same order: the logits must stay byte-identical.

## Arms and gates

old: the two-columns build (`9eef5a52`). new: the same plus hazard-tracked barriers. Same pairs, order, gates and digests as `PREREG-rows.md`: exit 0, pinned ids, one logits digest per
case across all six processes, run 2 duration, pairs alternated old,new,new,old,old,new. Cases: unicode-24, document-256, document-1024. A digest that differs from the pinned one fails the case
whatever the time.

## Decision rule

Median gain of at least 3 percent on unicode-24, every pair positive, all gates hold. Otherwise not proposed.

## Forecast (a prediction)

The no-barrier bound for the projections alone is 29 percent of a decode step (7 ms of 23.85), and decode is about 19 percent of the document-1024 run and about 90 percent of unicode-24's.
Dependent chains keep their barriers, so I expect a third to a half of the bound: 8 to 15 percent less time on unicode-24, 2 to 5 percent on document-1024. I give 20 percent to
a gain under 3 percent on unicode-24 (most dispatches are dependent), and 5 percent to a digest difference (a missed hazard).

## Not claimed

Other chips, models and drivers. The bound comes from the projections alone; the other kernels in a step are not in it.

## Record of the first build (added 2026-11-03, after its unicode-24 pair)

The first build of this change failed its digest gate on unicode-24: logits digest `e8535a02dd1e` against the pinned `e5447d9a3ab5` (the run was 5 percent shorter, which does not matter).
Cause: `greedy.zig` records its three dependent dispatches with `Pipeline.record` and relied on the barrier that call used to add, and its first fill needs the dispatches run before it fenced.
Fix: `record` keeps its barrier (it is the API of direct recorders); only `Pipeline.run` elides; `Commands.sync` fences open dispatches before a direct recorder starts. The corrected build is
a new pair under the same arms, decision rule and forecast.
