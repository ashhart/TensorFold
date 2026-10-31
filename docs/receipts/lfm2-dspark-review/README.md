# DSpark review fixes on current main

Source `14565fa73`, rebased onto main `d0c9dc9fb`, tested on 2026-10-28.
Apple M4 Pro, 20 GPU cores, 48 GiB RAM, macOS 26.5.1, Zig 0.17.0,
Apple Metal compiler 32023.883. Builds target `apple_m1`.

The shared and lone guards from the rejected PRs are absent. The lane core retains
confidence/grant pricing, hold-cost pricing and `copy_first`. Hold ordering reuses
one bounded allocation instead of allocating on every round. The initialization
cleanup releases Metal buffers before their owning list; injected scratch and warm
allocation failures exercise that path. Config admission refuses rank above 256 and
nonfinite f32 RoPE theta. The quantizer test widens its size calculation before
multiplication, and `--drafter` refuses a target restricted to one-row verification.

## Checks

- `zig build test test-kernel-contract test-golden fuzz-cortheon native dev-tools -Dcpu=apple_m1 --summary all`:
  749/749 steps succeeded; 8,414 tests passed, 46 skipped, 0 failed.
- `zig build -Dcpu=apple_m1 --summary all`: 239/239 steps succeeded.
- `zig build test-lfm2 test-lfm2-dspark -Dcpu=apple_m1 --summary all`:
  111/111 tests passed. The final scratch/warm failure test also passed in a subsequent
  focused run: 51/51 DSpark tests.
- `zig build test-lfm2-dspark -Dcpu=apple_m1 -Doptimize=debug --summary all`:
  51/51 passed, including both injected allocation failures.
- Debug `zig test` on `dspark_config.zig`: 4/4 passed; on `dspark_quant.zig`: 2/2 passed.
  The original quantizer test from `2e5966f68` aborted with integer overflow in Debug.
- `zig fmt --check zig` and `git diff --check`: passed.
- Lean: 428 findings on main, 428 on this source, 0 new.
- `tf-kernel-contract --only lfm2`: 11 kernels passed gates 1-4, 0 failures,
  0 known findings; all 17 planted controls red, 0 missed; 0 limit problems.
- `tf-kernel-contract --only tile_qmm`: 6 kernels passed gates 1-4, 0 failures,
  0 known findings; 0 limit problems.
- `tf-lfm2-dspark MODEL DRAFTER 256`: 9/9 drafts equal the independent fp32
  reference's picks; maximum confidence difference 0.00106; 20 steady drafter
  rounds had zero host allocations and resizes; exit 0.
- Completed `tf-lfm2-exact`: 72/72 checks on both M4 Pro and M5 Ultra, zero
  failures or skipped checks. This supersedes the earlier interrupted run.

## Served equality

Target `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` at
`667408d494bde191c2cb9dd4f4549827816b9275`, with the local released DSpark weights.
The drafter safetensors SHA-256 is recorded in the accompanying JSON.

For both default bf16 and explicit `--drafter-bits 8`, serve with `--parallel 16
--context 4096 --prompt-cache-gib 0 --keep-warm 0`. Run:

```sh
python3 tools/bench_concurrent.py http://127.0.0.1:18194 review \
  --tokens 64 --reps 1 --temperatures 0 --levels 1,4,8,16 --alone --serial
python3 tools/bench_concurrent.py http://127.0.0.1:18194 review \
  --tokens 64 --reps 1 --temperatures 0 --levels 1,4,8,16 --alone --serial \
  --prompts tools/bench_prompts.json
```

Each precision mode passed all 12 cells: eight same-prompt cells (code and chat)
and four distinct-prompt cells. Concurrent replies equal solo replies and drafted
solo replies equal `"draft": false` token hashes. Zero unequal replies and zero
failed requests. [Equality evidence](equality-and-load.json) retains per-request
concurrent, solo and plain token hashes.

These are short correctness checks, with one repetition, while builds were also
running. Their timings do not establish the decode performance gate.

## Peak during load

One model process at a time under a model lock. Both servers used the same target
and bf16 drafter, `--parallel 1 --context 4096 --prompt-cache-gib 0 --keep-warm 0`.
Read `/memory` after readiness and before sending any request. Its lifetime peak
agreed with `/usr/bin/time -l` at process exit.

- Exact #1392 head `1bb6e5db2`: 16,036,097,352 bytes, 14.935 GiB.
- Corrected stack source `14565fa73`: 15,527,930,776 bytes, 14.462 GiB.

The measured reduction during load is 508,166,576 bytes (484.6 MiB). This compares
the original pre-allocation head with the corrected stack rebased onto current
main; it is not an isolated comparison of #1393's commit on the same base.

## Extended qualification on the published stack

Source `2dbbc0e8f`, main baseline `d0c9dc9fb`. Additional device: Apple M5 Ultra,
256 GiB RAM, macOS 27.0, Metal compiler 32023.921.6, Zig 0.17.0.
[Extended evidence](extended-qualification.json) retains request hashes, counters,
per-round throughput and latency. Generated text and token arrays are omitted.
[Reproduction harness](qualification.py) supports `functional`, `long`,
`performance`, `performance qwen`, and `cold`; set `TF_QUAL_MAIN`,
`TF_QUAL_CANDIDATE`, `TF_QUAL_MODEL`, `TF_QUAL_DRAFTER`, and `TF_QUAL_OUT`
for the local installations. Run one model process at a time.

Both devices passed real two-turn resumed-versus-fresh chat equality (3,517 cached
prompt tokens), 29 seeded sampled equality cases across 1/4/8/16 streams, and
cancellation/replacement stress with Metal API validation enabled. M4 completed
143 requests with 84 cancellations at six slots and 252 with 134 at sixteen;
M5 completed 453 with 247 and 852 with 459 respectively. Zero wrong replies or
request errors. Server shutdowns were graceful.

M5 plain/drafted serving hashes agree at 4,096, 16,384, 65,536 and 130,816 prompt
tokens. Independent fp32 drafter checks agree on all nine picks at histories
65,535 and 130,815; maximum confidence differences 0.00532 and 0.00351.
Twenty steady rounds had zero host allocations/resizes. M5 also passes the same
11 LFM2 and six tile_qmm contracts, with all 17 LFM2 planted controls red.

### Performance remains an open issue

Three alternating rounds per arm, 256 generated tokens, 1/4/8/16 streams,
32-token warmups and uncached prompts. Aggregate decode rate uses the span from
the earliest first token to the latest last token. The `chat` fixture label in
the JSON means prose submitted to raw completions, not a templated chat request.
All cross-arm output hashes agree. These results do **not** clear a no-loss gate.

On M5, drafted versus main-plain median decode changes for code at 1/4/8/16
streams are +19.44%, -0.96%, -2.20%, -4.23%; for prose they are +35.89%,
-0.72%, -1.25%, -1.17%. Candidate-plain closely follows main-plain.
On M4, changes are approximately -6.0%, -13.9%, -5.6%, +0.7% for code and
-6.4%, -13.4%, -7.8%, -8.0% for prose. M4 also shows variation between plain
arms, so these measurements do not isolate every source of the loss.

M5 cold prefill medians at 2k/8k/32k/64k tokens are 0.1527/0.6027/2.9244/7.3276
seconds on main-plain versus 0.1634/0.6327/3.0466/7.5652 with the drafter.
Prompt counts were checked and cached-token counts were zero.

Other-family Qwen3.8-27B plus DFlash2 was tested at one and four streams with
three alternating rounds on both devices. M4 results vary with each executable's
independently measured persistent cost record: fresh records give code changes
of -7.13%/+3.71% and prose -6.47%/+0.83%. The original records and fresh-record
runs are both retained. No timings were copied between executables. M5 changes are +0.20%/-0.47% for code and +0.05%/+0.70% for prose.
No blanket other-family performance pass is claimed.

Two isolated, uncommitted experiments (a larger absorb staging buffer and
removing forced per-stream copy pricing) did not consistently remove the loss.
Neither experiment is part of this PR. Further pricing work requires its own
validation; the rejected shared/lone guards have not been restored.

M1, M2, M3, CUDA and HIP were not rerun for this revision. Focused Metal checks
are not a full cross-family kernel-contract receipt. All nine GitHub checks passed on `2dbbc0e8f` before this receipt-only update.

### Performance follow-up (`b4deee277`, M4 and M5)

Candidate source `b4deee277` keeps context ingestion on the original bf16 row
kernel through 32 rows, and uses the existing tiled kernel for held blocks past
16 rows. It measures each admissible held depth and interpolates neighboring
measured prices. The hold-price unit test covers a nonlinear timing curve.
Qwen/DFlash2 dispatch and scheduling are unchanged; only LFM2 supplies this hold
price table.

`performance-investigation.json` retains all 64 M4 comparison cells, their output
hashes, source hashes and actual arm order. Four-round median changes against
the published PR are +1.06%, +0.55%, effectively flat and +0.57% for code at
1/4/8/16 streams; prose changes are +0.27%, +0.75%, +0.41% and +0.42%.
The 43.99 tok/s lone-code outlier is retained. The final arm order repeats the
previous order, so this is not a perfectly balanced four-round experiment.
All reply hashes agree. Independent fp32 drafter picks agree in all nine cases.
Focused DSpark, shared-lane equality and native-option checks pass: 49 build
steps and 56 tests. Formatting, whitespace and harness syntax checks pass.
These small gains do not establish a no-loss result against main.

An earlier candidate changed context projections as well as held blocks and
regressed M4 16-stream prose by 8.38%. Preserving context projections removes
that regression in this comparison, supporting context arithmetic as a cause.
A separate acceptance-cap scheduler experiment lost roughly 2–3% in some M5
cells and is excluded. Faster individual holds alone did not guarantee faster
serving. The exact retained candidate also completed M5 validation, with source hashes
matching this tree and tested executable hashes retained in the JSON receipt.


M5 completed 72 cells in three Latin rotations of main without the external
DSpark drafter, the published PR and the candidate. Four-round M4 and three-round
M5 results use the same 256-token requests and aggregate decode metric. The main
arm still permits suffix copying; it is not the literal `"draft": false` oracle.
Candidate versus published median changes at 1/4/8/16 streams are -0.26%, +0.04%,
+0.31%, +0.58% for code and -0.04%, -0.02%, +0.06%, +0.72% for prose.
Against main, code is +21.87%, -0.90%, -2.22%, -3.51%; prose is +36.18%, -0.55%,
-1.19%, -0.36%. Wider concurrency therefore still does not clear a no-loss gate.

Final M5 LFM2 contracts: 11 pass, zero failures, all 17 planted controls red,
zero limit problems. Independent FP32 drafter picks agree in all nine cases,
with zero host allocations/resizes in 20 steady rounds. Resumed/fresh equality,
29 seeded sampled cases, eight 8-bit cells across 1/4/8/16 streams, and long
contexts of 65,536 and 130,816 tokens (drafted/undrafted and four concurrent/solo)
all pass. Cancellation stress completes 423 requests with 234 cancellations at
six slots and 827 with 445 cancellations at sixteen slots, zero wrong replies
or request errors. All servers shut down gracefully. Hashes and counters are in
`performance-investigation.json`. This extends, rather than replaces, the
historical full forward/cache and qualification receipts above.

After that qualification, main `dd0a9cc5e` was merged to resolve a new CLI
conflict. The resolution retains main's drafter-name reporting and this stack's
optional precision default. Native plus the same focused regression targets
pass: 66/66 build steps, 56/56 tests. Lean compares at 425 findings on both main
and the merged tree, zero new. Performance and served-GPU receipts above are
for `b4deee277` before this merge; they are not relabeled as exact merged-head
measurements.

### Wide-batch calibration (`099455490`)

LFM2 timed single-stream verification windows through 16 rows, but the core may
verify 32 shared rows. It therefore priced the extra rows by proportional
extrapolation. This revision measures the shared path with two individually
qualified windows, and anchors its measured increments to the existing 16-row
window cost. Per-stream qualification remains 16 rows. The granted-depth rule
also reuses its first copy-proposer decision: asking again can mutate the
proposer's silence state and change that same decision. A stateful regression
reproduces the duplicate-query bug. Qwen/DFlash2's policy is unchanged.

`wide-calibration-candidate.json` retains the exact source and executable hashes,
checkpoint/tokenizer identities, rendered prompt IDs, all 128 cells, request
SSE timing spans, output hashes and counters. Four Latin rotations compare main
`dd0a9cc5e`, published head `68c7435b0`, candidate without the drafter, and
candidate with bf16 drafting. Every output hash agrees. The plain performance
arms keep suffix copying enabled; literal `"draft": false` is the separate
correctness oracle.

Against the published PR, candidate drafted code changes at 1/4/8/16 streams are
+0.48%, **-1.93%**, -0.01%, **+10.96%**; prose changes are +0.05%, -0.42%,
+0.50%, **+8.86%**. At 16 streams it also beats main: 886.46 versus 830.13 tok/s
for code, and 862.91 versus 798.61 for prose. The task owner accepted the
four-stream tradeoff for the larger sixteen-stream gain. This is not a universal
no-loss claim. The candidate without the drafter also loses 2.04% on four-stream
prose versus main; all four arms are retained, including that result.

M5 final correctness passes all 11 LFM2 contracts, with all 17 planted controls
red and zero limit problems; 9/9 independent FP32 drafter picks; 20 steady rounds
with zero host allocations/resizes; resumed/fresh equality with 3,517 reused
prompt tokens; 29 seeded sampled cases; eight 8-bit cells; and drafted/plain plus
four-concurrent/solo equality at 65,536 and 130,816 prompt tokens. Six-slot stress
completes 423 requests with 233 cancellations; sixteen-slot stress completes
845 with 455 cancellations. Zero wrong replies or request errors; graceful
shutdowns throughout.

M4 passes the same 11 operator contracts and 17 controls, 9/9 independent drafter
picks (worst confidence difference 0.00106), and 20 allocation-free steady
rounds. Its served run stops at the resumed-cache prerequisite: the OS reports
memory pressure and explicitly disables keeping prompt states, so reused tokens
remain zero. Its later stress/sampled/8-bit phases did not run. No pressure guard
was bypassed, and no new M4 serving or throughput qualification is claimed.

Local native/family/shared-lane checks pass 69/69 steps and 117/117 tests. The
default build passes 239/239 steps. The direct proposer regression, 14 granted
and 34 copy-policy tests also pass. M5 pre-request, context-4096, one-stream bf16
load peaks are 15,709,645,744 bytes for published and 15,733,451,768 for candidate
(+22.7 MiB, one run each). Hot startup increases by roughly one second. Other
chips/backends and the full cross-family operator gate were not rerun.

`wide-calibration-ablations.json` retains 765 M5 diagnostic cells and 35 interrupted,
contended M4 cells. Interrupted runs are marked; they are not qualification
passes. Larger staging, changed copy-rate policies, confidence fallbacks and
uniform late-hold budgeting were discarded rather than included in this patch.

To reproduce the four-arm comparison, reserve the GPU exclusively and build
all three checkouts with `zig build native -Dcpu=apple_m1 -Dversion=1.1.8`, then run:

```sh
TF_QUAL_MAIN=/path/to/main/zig-out/native/bin/tensorfold-native \
TF_QUAL_PUBLISHED=/path/to/published/zig-out/native/bin/tensorfold-native \
TF_QUAL_CANDIDATE=/path/to/candidate/zig-out/native/bin/tensorfold-native \
TF_QUAL_CANDIDATE_ROOT=/path/to/candidate \
TF_QUAL_DRAFTER=/path/to/LFM2.5-24B-A2B-DSpark \
TF_QUAL_OUT=/tmp/tf-lfm2-wide-matrix \
python3 docs/receipts/lfm2-dspark-review/wide_matrix.py
```

`TF_QUAL_MODEL` overrides the default target snapshot path. The reproducer uses
only Python's standard library and the sibling `qualification.py` helper.

### Allocation follow-up

Probability fallback now overwrites its existing output buffer when a provider declines. Grafted probabilities fill the draft prefix directly in the final output. Structured-output held tokens are read directly into the final row buffer before grafts. Probability multiplication order, copy rates, parent indexing and decoding policy are unchanged.

For `n` fallback probabilities this saves `8*n` arena bytes. For `d` drafted prefix nodes it saves `8*d` arena bytes plus their copy. With `h` held tokens and `g` graft tokens, grammar row assembly saves `4*(2*h+g)` arena bytes and the held/joined intermediate copies. These are per affected call, not reductions in model weights or a measured serving peak.

The bounded-workspace regression permits only the final probability buffer, denies allocator growth, and covers both declined providers and grafted prefixes. The previous implementation fails it with OutOfMemory; the refactor passes. A grammar regression covers held-token and graft order, including replacing a rejected graft with grammar-generated tokens. The shared-lane/native host checks pass 57/57 steps, 341 tests passed and three skipped. The Metal draft-source fixture now initializes the shared timing count introduced by the calibration change; otherwise its deliberately undefined backend could crash.

The M5 native build passes. An alternating before/after serving comparison at 1/4/8/16 streams is queued behind another qualification's GPU lock. No extra throughput improvement or universal no-regression result is claimed for this allocation-only follow-up yet.
