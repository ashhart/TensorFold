# EXL3 Qwen3.8-Flash-Next on upstream TensorFold: final report

Branch `exl3-on-upstream-wip` (staging branch `exl3-c-main`), originally based on upstream tag `v1.1.0` and now merged with current upstream `main`.
Machine: par1-cs13, two RTX 5060 Ti 16 GiB (sm_120), 251 GB RAM, pack
`Qwen3.8-Flash-Next-exl3-3.05bpw`. All measurements are test measurements on idle cards under a lock
wrapper, not benchmarks.

## Current handover (2026-10-20)

Upstream draft PR [#1113](https://github.com/ashhart/TensorFold/pull/1113) is open from the fork branch
`exl3-on-upstream-wip` into upstream `main`. The branch integrates current upstream at `d1018b318`; exact upstream
commit ancestry is retained. The PR description contains the proposed dependency-ordered split plan. This remains
one large draft and is not merge-ready.

The user's acceptance target is TP=2 with CUDA graphs on both decode and prefill, and graphs must improve
performance. BF16 is the main KV format; dynamic KV paging to RAM is also required. CPU expert execution is
deferred. Decode graphs match eager for eligible captured shapes, but input preparation, the MTP chain's absorb
step, and host readback remain outside capture. No repeatable graph speedup over optimized blocking-NCCL eager
decode has been demonstrated. Prefill graphs capture each layer's dense sub-blocks; expert scheduling and ring
copies remain eager. A new event-wait-node path preserves overlap with TP=2 sub-block reductions and passes the
bit-equality gate. Across 8192-row prompts with 1024-, 2048- and 4096-row chunks, graph throughput tracks eager
within run variation; a graph-specific prefill win is not demonstrated. An 8192-row chunk OOMed and fell back to
the non-deep path, so it is excluded from graph results. Graph performance remains an open acceptance gate.

The near-200k × 2 TP=2 BF16 needle test (195,938 and 195,957 prompt tokens) passed on this tree before the final
agent changes; it was not rerun on the exact PR head. Treat that result as demonstrated on a near-final tree, not
as a final-head gate or an exact-200k test. The follow-up implementation now combines BF16 paging, snapshots, TP=2,
graphs, MTP and two concurrent HTTP chat histories at ordinary lengths. Restored graph state/logits match exactly;
warm, branch and rewind cache phases each reuse 4,122/4,139 prompt tokens with exact response hashes versus cold
cache. The 200k × 2 gate remains explicitly deferred. Live-rank overflow/failure injection and repeatable graph
speedup remain open; no graph performance win is claimed. BF16 is required; int8 KV is out of scope.

## 2026-10-20 upstream-main integration and release checks

The merge resolves the shared Flash Next hooks against current upstream `main`. On par1-cs13, Zig 0.17
`zig build test-cuda-host` passes after fixing a test fixture that left the new paging option undefined;
`zig build native -Dnvcc=nvcc -Dsm=120 -j16` also passes. `zig fmt --check zig/src zig/tests` passes after formatting
the three reported Zig files. These checks validate compilation and host tests only; the final-head TP=2 serving,
graph-performance, BF16 paging/snapshot/cache-reuse, and 200k × 2 gates remain to be rerun.

The lean checker does not pass: current upstream `main` has 444 findings and this branch has 1,761, leaving
1,317 new findings. Treat this as an explicit CONTRIBUTING blocker; the PR stays draft. On 2026-10-20, removed
118 tool co-author/session trailer lines from the 76 feature commits using a commit-object rewrite scoped to the
feature range. Original author identities, subjects, complete tree, and exact upstream `main` commit IDs are
preserved. The branch history is now grouped into five implementation commits, one upstream integration merge,
and one focused documentation commit. The draft omits the obsolete v1.1.0 and int8-first planning memos.

## 2026-10-20 prefill graph profile addendum

The new `prefill-prof` gate ran TP=2 on both GPUs with BF16 KV paging enabled, 8,192 rows, 4,096-row chunks, two
warm-ups, and one measured forward per mode. Both eager and graph runs returned the same cross-rank logits hash
(`ffcf463b229c2aa6`); graph mode replayed 96 dense-layer graphs. One matched untraced pair measured 5,597.1 ms
eager and 5,639.5 ms graph. In the matched Nsight pair, rank 0 measured 5,693.6 ms eager and 5,696.4 ms graph.
Neither pair establishes a speedup.

Nsight shows the graph replacing eager kernel submissions (20,559 `cuLaunchKernel` calls versus 2,103 plus 96
`cuGraphLaunch` calls) while expert-copy volume and kernel/collective totals remain nearly unchanged. Each rank
transferred about 28.8 GB H2D in approximately 48k operations; rank 0 scheduled 14,426 expert copies (13,993 counted
as prefetch) and 570 passes, with about 35 ms CPU planning. The measured picks-D2H-to-synchronize interval is about
5.2 s cumulatively, but includes the preceding dense kernels and NCCL dependencies; it is not pure CPU stall time.
The next prefill performance work is therefore a proper expert-routing/transfer pipeline, not another capture-threshold
change. The design investigation should first establish staging-arena memory fit after BF16 paging, pools, scratch,
and snapshots; then prototype GPU-built stable member lists and fixed-buffer double-buffered transfers while keeping
existing expert arithmetic and reduction order.

## What this adds

The EXL3 pack now runs through upstream's own Flash Next CUDA family and server (`tensorfold-native serve`),
on one card and at tp=2. Integration touches existing upstream files and adds new files under
`zig/src/families/flashnext/cuda_exl3_*.zig`, `zig/kernels/cuda/flashnext_exl3_*.cu` and `zig/tests/cuda/`.

| Area | What exists | Main files |
|---|---|---|
| Weights | EXL3 dense, shared, routed experts, 5-bit lm_head, PLE n-gram table read from the pack | `cuda_exl3_load.zig`, `cuda_exl3_store.zig`, `cuda_exl3_ple.zig` |
| Expert offload | Per-layer device line pool fed from a pinned host bank; LRU plus use count; misses copied on a side stream | `cuda_exl3_pool.zig`, `exl3_offload.zig`, `exl3_round.zig` |
| tp=2 | Expert parallel: rank 0 routed 0-255 (+MTP head), rank 1 routed 256-511 + shared expert; one fp32 all-reduce per layer; per-rank pools and half-banks; vocabulary-split lm_head | `cuda_exl3_tp.zig`, `flashnext_exl3_tp.cu` |
| Prefill | Expert-major deep chunks (thousands of rows), one ring of lines across all layers, 64-member tile kernel for the mul1 3-bit experts, tp=2 version with overlapped per-sub-block reduction, expert-major MTP head | `cuda_exl3_prefill*.zig`, `flashnext_exl3_prefill.cu` |
| MTP | Draft rows, drafts with the prompt cache at tp=2, rank 1 keeps no head cache in its windows | `cuda_exl3_mtp.zig`, `cuda_rank_engine.zig` hooks |
| Long context | Attention head split across ranks (12 query heads + 1 KV head each), indexer keys in mapped pinned host memory, 1024-row chunk, prompt states parked in host memory | `cuda_exl3_heads.zig`, `cuda_exl3_kv.zig`, `cuda_snap.zig` |
| KV paging | Full BF16 K/V of the 12 attention layers in pinned host memory with a device window and block cache; allocation/budget geometry now unified, default off | `cuda_exl3_kvpage.zig`, `cuda_exl3_kv.zig`, `flashnext_exl3_kvpage.cu` |
| CUDA graphs | Eligible decode rounds/windows capture the 48-layer pass, NCCL sums and head gathers; prefill captures dense layer sub-blocks; greedy MTP proposals have a conditional graph; graph instantiation runs on a worker | `cuda_exl3_dplan.zig`, `cuda_exl3_graph.zig`, `cuda_exl3_chain.zig`, `cuda_exl3_recycle.zig` |

## Verification

References: a float64 oracle (`exl3_*_cpu.zig`, kept as test code) and an independent Python run of the
upstream-quality reference (`tools/exl3_ref_golden.py`).

- **Single card, served over HTTP:** drafted output equals undrafted on every prompt; lm_head matches float64 to
  2.95e-3 of peak; teacher-forced top-1 equals the reference at all 16 tested prompt positions. This is a narrow
  sample, not an end-to-end strict golden pass.
- **tp=2, served over HTTP** (`--tp 2 --rank R`): drafted equals undrafted, bench hash `2c5d814cfa80` in every
  configuration; layers 0, 3 and 17 are bit-identical to tp=1 when run all-resident. The serving script's hash
  checks are truncated and it can exit successfully after a golden mismatch; interpret its status as evidence,
  not a strict pass/fail gate.
- **Near-200k context, 2 concurrent requests, tp=2, bf16 KV:** 195,938- and 195,957-token needle requests sent together
  were both answered correctly, with drafts on and off, drafted equal to undrafted.
- **KV paging:** at 200k on one card, last-row logits, 128 greedy tokens and every K/V row are bit-equal to keeping
  K/V on the card; sequence VRAM fell from 5.484 GiB to 1.391 GiB. TP=2 paging has only been tried sequentially
  at shorter contexts, without token hashes or paging counters.
- **Graphs:** logits of every graphed step are bit-equal to eager on both ranks; the device planner equals the host
  planner on 1,600 random rounds.

### Test measurements (ms per token, tp=2 serve gate, two 128-token runs each)

| Configuration | drafts off | drafts on |
|---|---|---|
| Before the stall fixes, eager | 29.7 - 30.1 | 14.6 - 14.9 |
| Defaults (n-gram gather inline) | 24.7 - 26.2 | 11.9 - 12.6 |
| Blocking NCCL communicator | 21.2 - 23.3 | 11.5 - 12.2 |
| Graphs + blocking NCCL (final branch) | 21.1 - 23.4 | 11.5 - 13.7 |

The three lower rows were reproduced on the final branch head (`44d65597`) by a separate gate run: bench hash
`2c5d814cfa80` and drafted equal to undrafted in every configuration; two of three golden prompts match.

Prefill, real text, rows per second: tp=2 deep 1,555 at 8k, 1,486 at 32k, 1,421 at 65k, 1,225 for a real 200k
prompt (tp=1 deep: 1,531 / 1,461 / 1,395 / 818). The old batched pass ran at 100 - 210 rows/s. A prior note's
168 s “on one card” duration is near the tp=2 rate (about 163 s by the reported rate), not the tp=1 rate (about
244 s); its configuration is unverified, so it is excluded. Saved concurrent-run logs show 312.1 / 562.5 s without drafts
and 380.1 / 693.2 s with drafts; the earlier 304 / 554 s figures are not reconciled to those logs.

## What graphs do and do not give

For eligible shapes, decode graphs match eager bit for bit, but a repeatable performance win over the optimized
eager path has **not been demonstrated** on this hardware. The decode speedup came from removing host stalls: the
pool's per-layer host wait (replaced by the device planner), thread spawns in the n-gram gather, and the non-blocking
NCCL communicator's host cost. Once those are gone a step is bound by GPU kernels (profile: about 20 ms of kernels
per token; dense EXL3 projections 6 ms, bf16 hyper-connection lanes 4.5 ms, all-reduces 2 - 3.5 ms, experts 3 ms).
With drafts on, the first drafted request after start is slower in the recorded gate; graph instantiation now runs
on a worker, but graph execution still synchronizes for results and timings do not isolate first-use capture from
replay. A TP=2 projection-fork experiment preserved logits but did not produce a repeatable graph-specific gain.
Graphs stay opt-in (`TF_FLASHNEXT_EXL3_GRAPH=1`). Prefill graph captures now preserve sub-block reduction overlap
with external event-wait nodes and pass bit parity. At chunk sizes 1024, 2048 and 4096, graph throughput remained
within run variation of eager; the former 10% regression is removed, but a speedup is not proven. An 8192-row
chunk request OOMed and used fallback, so it is not counted. Prefill graphs remain opt-in
(`TF_FLASHNEXT_EXL3_GRAPH_PREFILL=1`).

## Prefill graph replay follow-up (2026-10-20)

A TP=2, two-lane HTTP run at 16,040 prompt tokens used BF16 paging, MTP, snapshots and the default
`TF_FLASHNEXT_EXL3_GRAPH_PREFILL_AFTER=2`. Both ranks reported 72 prefill graph replays, 216 eager layer executions,
96 captures and about 314 ms cumulative capture time. Warm/rewind reused 16,033 tokens and branch reused 16,023;
all response hashes matched the cold baselines. The earlier 12k run's zero replays are explained by the threshold:
the available same-shape chunks only reached capture, with no later same-shape chunk to replay.

The DPLAN run and two graph runs measured cold prefill at 27.85/16.47 s, 27.92/16.39 s and 28.00/16.51 s across
the two concurrent lanes. This is within run/lane variation and is not a demonstrated graph speedup. A follow-up
keeps the complete exact replay key, but normalizes only the position and paging-window-origin hash for capture
sightings and driver-checked executable-update candidates; sequence/scratch/event ownership remains strict. The
updated gate retained exact output/cache hashes and reported 72 replays, 192 eager layer executions, 120 captures and
24 successful `Exec.update` calls per rank. Capture time was 325 ms; DPLAN/graph cold prefill measured 16.44/27.80 s
and 16.51/27.99 s. This exercises executable reuse across page-window changes, but shows no speedup; incompatible
topology still uses the instantiate fallback. Next profile the expert/pool host boundary and dense graph replay
versus eager time before making more topology changes.

## Switches

| Variable | Effect | Default |
|---|---|---|
| `TF_FLASHNEXT_EXL3_GRAPH=1` | Decode graphs (and prefill dense-half graphs where enabled) | off |
| `TF_FLASHNEXT_EXL3_GRAPH_PREFILL=1` | Prefill layer graphs at tp=2 | off |
| `TF_FLASHNEXT_NCCL_BLOCKING=1` | Blocking NCCL communicator (init has no deadline if the peer never joins) | off |
| `TF_FLASHNEXT_ATTN_SPLIT=0` | Replicated attention at tp=2 (the split is on by default) | split on |
| `TF_FLASHNEXT_IKC_HOST=1` | Indexer keys in mapped pinned host memory (needed for 200k x 2) | off |
| `TF_FLASHNEXT_KVPAGE=1` | K/V paging to host memory; takes precedence over the head split | off |
| `TF_FLASHNEXT_EXL3_PREFILL=0` | Turn the deep prefill off | on |
| `TF_FLASHNEXT_SHARD_HEAD=0` | Whole lm_head on both ranks | split on |
| `TF_FLASHNEXT_DRAFT_HEAD=affine4` | Cheaper draft head; changes which drafts are proposed (88/111 accepted instead of 90/113), output stays bit-equal | off |

The 200k x 2 tp=2 run used resident BF16 KV (not dynamic KV paging) and was run with
`TF_FLASHNEXT_EXL3_STREAMS=2 TF_FLASHNEXT_IKC_HOST=1
TF_FLASHNEXT_EXL3_PREFILL_ROWS=2048 TENSORFOLD_MEMORY_RESERVE_GIB=0.75 TF_FLASHNEXT_NCCL_BLOCKING=1 --parallel 2`;
the pool had 11 lines per layer on rank 0 and 29 on rank 1 (floor 10).

## Known gaps and risks

1. **Golden gate fails on one prompt.** The 16-token prompt leaves the Python reference at step 3 (ours 9564,
   reference 6124, reference margin 0.125, one bf16 step) on one card and at tp=2. Two of three golden prompts
   match exactly. Treated as a bf16 near-tie; not proven. A teacher-forced margin study would settle it.
2. **The head split is not bit-identical to replicated attention** (different fp32 order in the o_proj sum).
   Greedy continuations after 8k and 32k real tokens differ after 3 - 4 tokens; not shown to be near-ties because
   the server returns no logprobs.
3. **tp=2 parity is not established.** A saved 39-position teacher-forced run matched tp=1 at 37/39 positions
   and failed at positions 18 and 33; logits also differ due to fp32 reduction order. At 65k and 200k the
   deep-prefill last row swaps a near-tie pair between tp=1 and tp=2. The single-layer 16-position top-1 check
   does not cover this case.
4. **200k x 2 remains deferred.** Rank-0 pool headroom is tight and simultaneous prefill can hurt decode. The prior
   near-200k resident-BF16 retrieval gate is not a claim for paged 200k capacity. Ordinary-length prompt-cache reuse
   now works with TP=2, BF16 paging, MTP and graph mode: two concurrent chat histories passed warm, branch and rewind
   phases with 4,122/4,139 prompt tokens reused and exact response-hash parity against cold-cache baselines.
5. **BF16 KV paging and snapshots now pass ordinary-length TP=2 correctness gates.** A two-GPU graph test restored
   two paged lanes and preserved exact state/logits; each rank replayed prefill graphs 72 times and decode graphs 5
   times. The HTTP cache gate proves actual snapshot reuse, MTP continuation and two concurrent requests. Still open:
   repeatable graph speedup/profiling, live-rank overflow and failure injection, and long-context paged capacity.
   200k × 2 is deliberately not the next gate. The user’s priority is to make graph execution and cache reuse robust
   and fast first. Int8 KV is out of scope; BF16 remains the primary and required format.
6. **200k reference parity is not established:** a saved 200k prefill comparison exited 1 with max logit
   difference 1.1406 and top-1 6009 versus 19251 (reference margin 0.25). The successful 200k × 2 needle answers
   establish retrieval/serving behavior for those prompts, not strict full-logit parity.
7. **Graph path:** graphs execute and replay for both prefill and decode, including TP2 BF16 paged snapshot restore.
   However, neither decode nor prefill graphs have a repeatable measured speedup yet. MTP verification also has a redundant GPU best-selection/readback
   boundary after full-logit readback. An opt-in verify-tail experiment preserved output/hash but showed no
   observable speed change in one TP=2 A/B pair; no graph speedup is claimed.
8. **Prefill at tp=2 is only about 2% faster than tp=1 at equal chunk sizes:** the replicated dense work
   (projections, GDN, attention, hyper-connections) is the bottleneck, not the experts.
9. **Blocking NCCL is opt-in** until its init has a timeout.
10. Existing automated gates need stricter pass/fail behavior before serving as acceptance gates: the HTTP gate can
    print golden mismatches without failing, and compares only truncated drafted/undrafted hash prefixes; the
    prefill comparison allows a close-margin top-1 and can skip missing reference dumps. The passing summaries
    above must be read with those limits.
11. **MTP draft-token parity is not directly asserted for graph replay.** The graph test checks main-model logits,
    not the draft proposals or replay count. Final drafted/undrafted output equality does not prove the proposed
    tokens matched. The draft-row test covers limited inputs and does not assert argmax equality.
12. `zig build test-cuda-host` fails on clean upstream v1.1.0 on macOS as well; `test-flashnext-exl3-host`
   passes. Upstream PR #1046, which touches our hook points, could not be read.

## Next steps

1. Meet the graph acceptance gate: reduce decode host boundaries (first target is MTP verify's redundant best-
   selection/readback), make decode graphs measurably faster than optimized eager decode, and extend prefill capture
   beyond dense sub-blocks without losing reduction overlap. Recheck with repeated TP=2 gates before changing defaults.
2. Exercise fail-closed paging overflow and collective cleanup on live TP2 ranks; snapshots/MTP rollback and
   ordinary-context cache reuse are implemented and pass their current correctness gates. Defer the 200k × 2
   acceptance run until graph execution and cache reuse are optimized.
3. Add an init timeout to the blocking NCCL communicator before making it the tp=2 default.
4. Make the HTTP and prefill gates fail on strict mismatches, compare full hashes, and require reference artifacts
   for long-context parity runs; investigate the 200k reference-logit failure.
5. Assert actual graph replay and compare full MTP draft-token sequences against eager; strengthen the draft-row
   test to assert the decision-critical top-1 output.
6. Margin study for the golden near-tie and for the head split (needs logprobs from the server).
7. Spread the MTP head across both ranks (rank 1 idles about 28% with drafts on); split the replicated dense
   work to speed up prefill at tp=2.
8. Optional later: exact same-layer CPU execution of low-weight pool misses (AVX2 on the EPYC 7452); design notes
   and the KTransformers review are in the session record, nothing is built.
9. Split this branch into reviewable PRs after the TP=2 graph gates pass.

## Branch map

`exl3-c-main` merges: core adapter, pool, PLE, draft rows, expert-parallel TP=2, serving, split `lm_head`, deep
prefill, BF16 KV paging, snapshots/cache reuse, and CUDA graphs. Current design and validation notes:
`docs/plans/exl3-kv-paging.md` and `docs/plans/exl3-graphs.md`. Superseded v1.1.0 and int8-first planning memos
were removed from the draft because they describe obsolete decisions and conflict with the BF16 paging scope.

Not done on purpose: no change to `docs/recipes`, no family or model registration (held until the strict golden
gate passes).
