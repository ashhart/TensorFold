# V6 decode experiment — implementation contract and evidence

Updated: 2026-10-22. Target: the existing TP2 GB10 experimental deployment on
port 8890 only. No weight, dtype, split-K, sampling, DFlash confidence, context
or concurrency changes. These are data-movement/launch optimizations, not a
new quantization recipe.

Status: implementation, operator checks, focused CUDA contracts, dual-node
sampling, whole-model regressions and repeated off/on measurements completed.
**No reproducible end-to-end decode benefit was established, so v6 is retained
as an opt-in experiment, not selected for normal use. The normal controller
continues to select v5.** The microbenchmarks below are not model speedups.

## Frozen artifacts and controls

| Artifact | Identity |
|---|---|
| Baseline | Frozen `exp-v5-20261022`; retained image/controller |
| Baseline source archive | `source-v5.tar.gz`, SHA-256 `1cf4f6d3d4afeb3a374f0e967790638ef06619146165da6a4f8746663ad2f1a3` (archive hash, not a Git commit) |
| Candidate image tag | `local-reference/tensorfold-fp4local:exp-v6-20261022` |
| Candidate image ID | `sha256:c1af38edb6abcce029cb78193b542fb22116e1fd9ee288d829f9470aebb90a28` |
| Candidate binary SHA-256 | `5d30d06f53a066ba954150dfb93b6e0a21420e97805c3adbebfa7a97885bc43b` |

All three candidates are independent, default-off switches. `tfctl-v6` exposes
`TUNING=off|kda|moe|gather|all`, default `off`, so the same candidate build can
be tested with individual changes and their combination. Both ranks must
agree on strict `0`/`1` settings before loading the model; this is especially
important for the changed collective sequence. The controller checks image
identity on both nodes. Keep the immutable v5 image/controller as a separate
rollback path; turning v6 switches off is not a substitute for that baseline.

## Implemented candidates and arithmetic boundaries

### KDA direct projection output — `TF_GLM_KDA_DIRECT_OUT`

For the FP4-Local split input projection, MXFP8 QKV and BF16 auxiliary outputs
write directly into the final strided row. The TP2 shape is input width 4096,
QKV width 12288, auxiliary width 288, final row stride 12576 elements.

- Dispatch is restricted to non-prompt rows 2–32 with a supported selected
  kernel, a projection lane, MXFP8 split-K equal to 1, and supported BF16
  strided output. Single-row, prefill and unsupported paths retain the old
  implementation.
- MXFP8 MMA order and its BF16 output boundary are unchanged. The auxiliary
  BF16 projection retains split-K 8, the same partial results and the same
  ordered reduction; only the reducer's destination stride changes.
- Two intermediate copy launches are removed. No activation conversion,
  weight conversion, split-K retuning or persistent buffer is added.
- Code: `cuda_kda_block.zig`, the strided `qmmf` kernels/glue, and
  `glm_mm_reduce_stride` in the existing family kernel modules.

### Shared-expert tail fusion — `TF_GLM_FUSED_MOE_TAIL`

`tf_glm_combine_shared` replaces the FP32 shared-expert `put_slot` copy plus
the existing ordered combine. It reads the shared result directly for the
last slot. The tested TP2 shape has width 4096, eight routed slots and one
shared slot.

- Enabled only when `!prompt && l.exl3 == null`. Prefill retains its existing
  BF16 shared-slot rounding boundary; EXL3 is unchanged.
- The accumulator starts at positive zero. Slots are visited in the same
  order, using `__fmaf_rn(value, weight, accumulator)` for every slot. The
  shared-slot multiplication is retained even when its weight is 1; no
  reassociation, separate multiply/add, or altered output cast is introduced.
- Inputs and FP32 output use the same storage representations. Shared and
  routed scratch are disjoint, execution remains on the same stream, and
  there is no later consumer that requires the removed shared-slot copy.
- Launch: block 128, grid `(rows, ceil(width/1024), 1)`, eight contiguous
  columns per thread, no dynamic shared memory. No new persistent buffer.
- Code: `glm_glue_rows.cuh`, `cuda_glue.zig`, `cuda_moe.zig`.

### Sample-window gather — `TF_GLM_SAMPLE_GATHER`

Equal-width TP2 sampled logits use one window-wide all-gather, followed by a
byte-exact rank-major to row-major copy into the existing full-logit buffer.
The vocabulary is 154880, with 77440 values per rank.

- Logit dtype, sampling rule, RNG inputs, seed handling and draw kernel are
  unchanged. Token IDs and sampled probability bits are compared, not just
  final decoded text.
- The production sampler capacity is eight rows (`fw.window_rows`). Enabled
  scratch is 2,478,080 bytes (about 2.36 MiB) per node/engine, allocated through
  `DeviceBuffer` and included in actual post-load memory accounting; four
  concurrent streams do not multiply this allocation by four. No separate
  static admission-estimator adjustment is claimed.
  Disabled mode allocates none of this extra scratch. The test harness also
  constructs a 32-row sampler with 9,912,320 bytes (about 9.45 MiB).
- The existing RoCE fast path has a 4 MiB per-shard limit. Rows 1–8 fit;
  a 32-row shard is 4,956,160 bytes and uses the existing NCCL fallback.
  The 32-row NCCL case is a harness stress test, not the production window.
  Unsupported layouts and the public gather path used by grammar/prompt-logprob
  callers retain the existing behavior. The public gather test also covers
  larger caller-owned output buffers; it does not establish that `draw` accepts
  requests beyond its allocated capacity.
- Code: `cuda_sampler.zig`, `cuda_ranks.zig`; dual-node test:
  `zig/tests/cuda/glm_sample_gather.zig`.

## Detailed profiling and failure handling

`TF_GLM_PROF=1` plus `TF_GLM_DETAIL_PROF=1` enables optional nested CUDA-event
timing. `kda_project` includes its nested `kda_copy`; detailed intervals can
overlap and must not be summed as disjoint GPU time. The legacy MoE copy is
`moe_slot`, while the fused tail is one `moe_combine` interval. Profiling is
OFF for speed tests.

Profiler lifetimes use an active-call epoch: a stale/double-ended scope cannot
close a slot in a later call. Calls open only after a successful event record.
Record failures, overflow, unclosed scopes, elapsed-time errors, negative or
non-finite timings invalidate the **whole call**; partial totals are not
committed. Aborts and coarse synchronization failures discard the detailed
call. No synchronization is added per scope. Host tests cover nesting/reuse,
allocation unwind, stale/double scopes, invalid-call modes and aborts.

## Completed validation

GPU operator receipts identify NVIDIA GB10, PyTorch `2.13.0+cu130`, CUDA 13.0
and SHA-256 hashes of the frozen and candidate fatbins. Frozen v5 fatbins are
independent of candidate flag-off code, avoiding a comparison only against
newly compiled implementations of both sides.

| Check | Result and scope |
|---|---|
| Frozen-v5 MoE Python comparison | 136 cases passed, five byte-identical repeats per case |
| Frozen-v5 KDA Python comparison | All 52 cases passed, five repeats; frozen v5, candidate off and candidate direct outputs matched |
| Focused CUDA contracts | MoE: 1 passed, 0 failed; KDA: 2 passed, 0 failed |
| Dual-node sampled gather | Each rank passed 108 exact sampled draws; RoCE and NCCL fallback exercised |
| Final host build/tests | 254/254 build steps succeeded; 1067 passed, 4 skipped |

MoE Python coverage includes every row count 1–32 at width 4096, random,
cancellation/stress, non-finite, signed-zero/subnormal fixtures, and boundary
widths 8/1016/1024/1032 at rows 1 and 32. Checks include five repeats, guard
bytes, immutable inputs, an intentionally poisoned obsolete shared slot,
nonzero row offsets, untouched outside rows, solo-row equivalence and an
independent FP64 reference/classification check. The focused contract adds
36 shape comparisons and exact integer-sum stress controls against accidental
low-precision accumulation.

KDA Python coverage uses rows 1–8, 15, 16, 17, 31 and 32 with random, zero,
extreme and non-finite inputs; real packed MXFP8/BF16 projections, row padding,
guard bytes and input/weight immutability are checked. Row 1 is a kernel test,
not an enabled runtime optimization. Focused contracts cover the two strided
outputs separately, five-repeat replacement checks, rejected unsupported
edges, non-finite behavior and accumulation stress.

Dual-node sampling coverage uses rows 1–8 and 32, four fixtures and three
sampling rules, plus grammar, source-guard, raw NCCL and over-capacity checks.
It includes RNG positions beyond 32K; **this is not a whole-model >32K context
test**.

The standalone focused contract runner does **not** exercise the unified
runner's planted failure controls. These receipts qualify this local
experiment only, not a complete upstream kernel qualification or all-device
correctness claim. The four host-test skips likewise are not passes.

Receipts and reproduction sources, relative to this experiment directory:

- `results/moe-tail-v6.jsonl`, `tests/test_moe_tail.py`;
  `upstream/zig/tests/cuda/contract_glm_moe_tail.zig`.
- `results/kda-direct-v6.jsonl`, `tests/bench_kda_direct.py`;
  `upstream/zig/tests/cuda/contract_kda_direct.zig`.
- `results/contract-decode-v6.log` (focused contracts and driver resources).
- `results/sample-gather-v6-rank0.log`, `results/sample-gather-v6-rank1.log`.
- `results/build-image-v6.log` (final build/tests and image ID).

## Operator microtimings — not whole-model performance

Times below are median microseconds per tested operator chain, from the GPU
receipts. They compare different harnesses and must not be added together or
converted directly to tokens/s.

### MoE copy + combine versus fused tail

This harness replays 20 chains per CUDA graph, with six alternating timing
samples. It measures only the FP32 tail, not expert matmuls or communication.

| Rows | Frozen v5 copy + combine (µs) | Fused tail (µs) |
|---:|---:|---:|
| 1 | 2.970 | 1.944 |
| 2 | 2.970 | 1.946 |
| 4 | 2.972 | 1.945 |
| 8 | 3.140 | 2.072 |
| 16 | 3.831 | 2.486 |
| 32 | 5.490 | 3.674 |

The isolated chain is about 1.49–1.54× faster, saving about 1.0–1.8 µs. This
does **not** mean the model is 1.5× faster.

### KDA projections and destination copies

This harness uses Python/ctypes-fed dependent launches and CUDA events, no CUDA
graph and **PDL off**. Times can include host-fed idle gaps and are not pure
individual-kernel execution time. There is no full-model or TP scheduler around
it. A representative subset of
the 13 tested row counts follows. Candidate off is shown to distinguish
recompilation/run variance from the enabled change.

| Rows | Frozen v5 (µs) | Candidate off (µs) | Candidate direct (µs) |
|---:|---:|---:|---:|
| 2 | 231.073 | 231.687 | 226.444 |
| 4 | 234.588 | 235.242 | 227.933 |
| 8 | 235.519 | 236.238 | 228.692 |
| 16 | 240.483 | 241.243 | 231.728 |
| 32 | 253.252 | 252.547 | 236.823 |

Savings are a few to roughly 17 µs for these measured shapes. Real stream,
PDL, graph and communication behavior can change the realized benefit.
Sample-window gather has a correctness receipt but no separately qualified
end-to-end timing result at this stage.

## Driver resource receipts

These are actual CUDA driver attributes/occupancy observations on GB10, not
just `cuobjdump` text. All listed kernels report zero local memory and zero
static user shared memory.

| Kernel | Registers/thread | Dynamic shared (bytes) | Max threads/block | Active blocks/SM |
|---|---:|---:|---:|---:|
| `tf_glm_combine_shared` | 44 | 0 | 1024 | Not recorded |
| Old `tf_glm_combine` | 40 | 0 | 1024 | Not recorded |
| Old `tf_glm_put_slot` | 22 | 0 | 1024 | Not recorded |
| `qmmf_mxfp8_stride16` | 54 | 25088 | 128 | 3 |
| `qmmf_mxfp8_stride32` | 69 | 33280 | 128 | 2 |
| `glm_mm_reduce_stride` | 40 | 0 | 1024 | 6 |

`Max threads/block` is the kernel limit, not the selected launch size; for
example, the MoE tail launches 128 threads. `cuobjdump` lists 1024 bytes while
the driver reports 0 static user shared memory; this table uses the measured
driver attributes without inferring the source of that discrepancy. Occupancy and register
counts alone are not performance predictions.

## Remaining gates and final-selection record

Do not promote a candidate solely because it starts successfully or passes
the isolated operators. Operator candidates were measured independently.
Whole-model checks compare all-off with the combination, with profiling
disabled. Individual full-model ablations have not been run: each TP2 reload
takes roughly seven minutes, so the bounded experiment first checks combined
benefit and regressions. The lack of those ablations limits attribution.

- [x] Whole-model fixed-seed output comparison against immutable v5 and
  same-build off, including DFlash and plain decoding.
- [x] Four concurrent requests; >32K cold/cache reuse and cancel/retry.
- [x] Image inputs, tools, Responses and the existing local compatibility
  fixes, under the candidate configuration.
- [x] End-to-end prefill/decode/acceptance timing, RAM/swap and peak-memory
  observations; any measurable regression attributed before promotion.
- [x] Final selection: v6 not promoted; normal controller and immutable v5
  remain unchanged. A final v5 readiness/smoke receipt is recorded at handoff.

Final selection: **v5 retained**. Whole-model correctness: **passed for the
listed local cases**. Whole-model speedup: **not demonstrated**. Broad
model-quality/capacity claims and full upstream qualification remain out of
scope.

### First combined whole-model result

The fresh v5 baseline has two repetitions, first v6 off has one, and v6 all
has three. These fixed public fixtures use 384 generated tokens (reasoning
included), seed 119, thinking enabled/low, and the unchanged DFlash2 settings.
Every output token SHA-256 and every single-request draft acceptance count
matches. No claim is made that these fixtures reproduce the user's longer
article/HTML workload.

| Decode fixture | v5 fresh median token/s | v6 all median token/s |
|---|---:|---:|
| Prose, temperature 0 | 39.59 | 39.68 |
| HTML, temperature 0 | 39.99 | 40.02 |
| Prose, temperature 1 | 35.43 | 35.53 |
| HTML, temperature 1 | 46.42 | 47.08 |

The four-cell mean decode duration improves only about 0.38%; per-cell median
changes range from +0.05% to +1.43%. Within-cell candidate variation reaches
2.4%, so this is **not evidence of a robust end-to-end acceleration**. The
single v6-off run is unusually slower in some cells and cannot establish a
4–6% improvement. Repeated off/on comparison is retained separately below.

Short concurrent tests include prefill in aggregate throughput: two requests
44.22→44.56 token/s, four requests 57.80→57.66. All request bodies and hashes
match; only one concurrent run per state was taken, so these are effectively
flat observations, not concurrency speedup claims.

Cold 2K/8K/33K/66K prefill seconds: v5 1.735/7.334/28.941/57.888;
v6 all 1.758/7.431/29.271/58.485. The candidate reports zero reused tokens
and matches all output hashes. Its approximately 1% slower times are small
non-alternating observations; these decode-only switches do not claim to
accelerate prefill. This is cache-miss execution after a separate prompt-kernel
warmup, not a cache-disabled/fresh-process sweep.

Served regressions exited 0 for `smoke`, `sampled`, `parallel`, `long`,
`cancel-long`, `vision`, `api`. Two sampled seeds match plain decoding;
all four concurrent replies match solo runs. The 41346-token cached and
plain runs match cold execution (40960 tokens restored). The 41363-token
stream was interrupted after 20 SSE lines / seven generated deltas, before
completion; fixed-seed sampled drafted/plain retries then matched. Direct and
tool-returned images identified the red square; required tools, JSON schema,
Responses include tolerance and SSE completed correctly.

Whole-model receipts: `perf-v5-before-v6.jsonl`, `perf-v6-off.jsonl`,
`perf-v6-all.jsonl`, `parallel-v5-before-v6.jsonl`, `parallel-v6-all.jsonl`,
`prefill-v6-all.jsonl`, and `served-v6-all-*.jsonl` under `results/`.

### Why operator wins do not imply a large decode gain

The MoE tail saves only about a microsecond per affected layer, not the large
expert matrix multiplications. KDA direct output still reads the same weights
and performs the same projections. Sample-window gather replaces `2R` grouped
NCCL broadcasts with one window all-gather, but does **not** reduce payload
bytes or necessarily save `2R` network round trips (the old calls were grouped).
Its local reconstruction changes from R copies to 2R copies. It affects only
ordinary sampled output, not greedy/grammar draws or the main transformer
communication; concurrent streams still gather separately. Main weight reads,
MLA/KDA compute and the sequential TP dependency chain remain.

The new detailed profiler was host-tested and is packaged for future work;
it was not enabled in these speed receipts, and no new fine-grained whole-model
timing breakdown is claimed here.

### Repeated off/on result and decision

After the initial all-on run and feature checks, the service was restarted
all-off and then all-on, each with a separate warmup and three repetitions.
This gives an all→off→all sequence on the identical v6 image. All requests,
token hashes and single-request speculative counts still match v5.

| Decode fixture | v6 off repeat median token/s | v6 all repeat median token/s | Change |
|---|---:|---:|---:|
| Prose, temperature 0 | 39.76 | 39.53 | −0.57% |
| HTML, temperature 0 | 40.05 | 39.91 | −0.34% |
| Prose, temperature 1 | 35.49 | 35.37 | −0.32% |
| HTML, temperature 1 | 46.74 | 46.68 | −0.13% |

Mean decode duration across the four equally weighted cells is 9.5425 s off
versus 9.5766 s on (+0.36%). This reverses the tiny initial advantage. Neither
a stable benefit nor a material regression is established; clocks are stock
and variable, and the differences are too small to justify default adoption.
The complete attempts are kept in `perf-v6-{off,all}-repeat.jsonl`.

The three switches remain available independently for future attribution;
the v6 image and frozen source are retained. The normal `tfctl` continues to
delegate to `tfctl-v5` and retains its successful N128 prefill optimization.
No extra Dense FP8, new activation quantization, changed split-K or changed
DFlash2 policy was introduced to obtain a favorable result.

### Memory and deployment boundary

All-on head load-end device buffers are 99,626,734,100 bytes; the recorded
owned-device peak during load is 99,873,149,908 bytes. This is the engine's
owned-buffer counter, not total system RAM or a full peak-resident-set trace.
After load and testing both nodes had approximately 17 GiB MemAvailable.
Resident swap existed (~118 MiB on A, up to ~697 MiB observed on B during
feature tests). A short live sample had no swap-out and only one B interval
with 18 KiB/s swap-in; it does not prove that the whole run had zero paging.
No swapoff, page-cache dropping or context reduction was used.

Context remains 524288, stream limit four, DFlash2 maximum seven, FP8 latent
KV, 2 GiB prompt-cache request and 6 GiB OS reserve. V6 tested the original
profile only; the Huihui view was not requalified in this round. Existing
vLLM/EXL3 deployments, the 8888 endpoint, OliveTin and checkpoint files remain
unchanged. V6 tests use only the isolated 8890 endpoint.

To try a retained candidate later (stop the current experiment first):

```bash
ssh node-a /srv/tensorfold-experiment/tfctl stop
ssh node-a env TUNING=all /srv/tensorfold-experiment/tfctl-v6 start original 8890
```

Use `TUNING=kda`, `moe` or `gather` for an individual candidate, or `off` for
the same-build control. The v6 controller defaults to off and profiling off.
To restore the selected v5:

```bash
ssh node-a /srv/tensorfold-experiment/tfctl-v6 stop
ssh node-a /srv/tensorfold-experiment/tfctl start original 8890
```

A start return is not readiness; wait until `/v1/models` succeeds. Both images
and versioned controllers are preserved. `provenance-v6.json` pins the source,
binary and image identities; `tests/summarize_v6.py` rechecks output hashes and
summarizes the retained local speed records.

### Final live handoff

V5 was actually restarted on both nodes and `/v1/models` returned the original
model with a 524288-token window and thinking default high. Both running image
IDs equal `sha256:77a6204988f5012919aca93198a267f5cc8faafb0f49ee9848d5479a13d0fb20`.
The post-rollback `smoke` command exited 0: eight receipts, both draft/plain
pairs equal, 17×19 correctly answered, required tool and Responses succeeded.
The receipt is `results/rollback-v5-after-v6.jsonl`.

Final MemAvailable is about 17.50 GiB on A and 17.46 GiB on B; resident swap
about 119/179 MiB. Five two-second final intervals on each node report zero
swap-in/out (`vmstat-v5-restored-{head,worker}.log`). These are final idle
observations, not a claim that every preceding load/test had zero paging.
The selected v5 service is left running on `192.0.2.1:8890`.
