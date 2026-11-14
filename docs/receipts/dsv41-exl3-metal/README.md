# EXL3 Metal follow-up qualification

This is a follow-up to [#2373](https://github.com/ashhart/TensorFold/pull/2373), not release qualification.
The parent is `bec140d545aa50b765f26cf295b43179c6b2834f`.
The initial served comparison used `e070123319b69e73c969b91409dd7195fbbef93e` production sources and
`128cfe743a76f59b350d0c750a12669708029ff3` qualification fixtures.
Projection contracts and the first batched-head repair were recorded at `6aa834fdb4c58192a21a31524a5bf298b7b7ecec`.
Later attention repairs change numerical results: the earlier served hashes and timings are historical, not final qualification.
[fidelity-128.json](fidelity-128.json) identifies the repaired capture's production files by SHA256.

The [maintainer's parent review](https://github.com/ashhart/TensorFold/pull/2373#issuecomment-6730527087)
still disputes code provenance and lists integration gates. This dependent delta does not establish the parent's copy-rule compliance.

[environment.json](environment.json) records the checkpoint metadata fingerprints, source revision, build settings
and hardware: **M5 Ultra, 64 GPU cores, 256 GiB**, not the parent's historical 80-core machine.
The pack is pinned to `45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b`.
[model-fingerprints.json](model-fingerprints.json) binds all 39 weight shards, both streamed source shards,
the pack/source metadata and tokenizers, the chat template and the generated token map.

## Checks already exercised

- `zig build test-lane-api-host -j4 --summary all`: 545 passed.
- The initial host EXL3 and environment checks passed (environment 12/12); a temporary `xcrun` shim blocked native
  compilation. With the real Metal 27.2.47 compiler, M4 native checks subsequently passed 118 tests with 3 skips.
- On the M5, `zig build native test-exl3 test-environment-host test-native test-deepseek-v41
  -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8 --summary all` at `59535dad914819ea4d281f87c81a138c4f2c4e1b`:
  the family GPU target passed 73/73, EXL3 host checks passed, and environment checks passed 12/12.
  The combined command failed because the native cone diagnostic imported an excluded transport module.
- After removing that dependency and making cone mismatches fail closed, `zig build native test-native
  -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8 --summary all` at the serving revision passed:
  69/69 build steps, 118 tests passed, 3 skipped.
- An actual `GLM_LOAD_LIMIT_GB=1` candidate startup refused both workspace sizes with `LoadLimitExceeded`, exit 1,
  without an allocator abort. The wide plan requested 217556406008 bytes and the narrow plan 214737522424 bytes.
- At the common four-slot, 32768-context geometry with `GLM_LOAD_LIMIT_GB=240`, the parent default workspace
  refused admission: 235727027864 resident bytes plus 8589934592 margin exceeded the 230553780224-byte allowance.
  The comparison therefore pins **4096 workspace rows in both arms** instead of raising the machine's limit.

The 73-test family record predates the final core fallback fix. The final standalone projection target,
`zig build test-exl3-gpu -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8 --summary all`, ran on the qualification tree:
**36 passed, 3 failed, 39 total; exit 1**. The parent target returned **21 passed, 6 failed, 27 total; exit 1**.
All three remaining failing test cases also fail on the parent: mixed-width routed/dense bit parity,
packed-sign filtered/dense BF16 parity, and grouped/scattered versus separate projection bit parity.
They remain merge gates; neither complete projection suite is reported as passing.
The indexed odd/wide scalar-reference check, tiled/table bit checks and new chain/MoE fusion checks pass.
The final native rebuild passed 66/66 build steps. Raw build/server logs are not published because they contain
local filesystem paths; [operator-checks.json](operator-checks.json) records the scoped outcomes.
Both historical served revisions reported a 16-byte allocator leak on shutdown. The backend now frees its per-slot
`last_depth` allocation, and the allocator regression passes; final served shutdown with explicit drafts remains a separate gate.

## Reproduce the served comparison

Use separate checkouts at the parent and serving revisions, the same pack and original Engram tables, Zig 0.17.0,
an installed Metal compiler, and Python 3.10 or later. Build each with:

```sh
zig build native -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8
python3 docs/receipts/dsv41-exl3-metal/prompts.py /tmp/exl3-prompts
```

Run the prompt generator from the candidate checkout; the parent does not contain it.
Run only one server at a time. For each checkout, use this common command; on the candidate add
`TF_DSV41_CONE_CHECK=4096,16384` to run full/cone logits, cursor and rollback checks before readiness.
The checker restores the configured cone mode afterwards; any mismatch fails startup.

```sh
GLM_LOAD_LIMIT_GB=240 TF_DSV41_PREFILL_STEP=4096 \
./zig-out/native/bin/tensorfold-native serve <pack> \
  --engram-stream-dir <orig> --chat-template <pack>/chat_template.jinja \
  --context 32768 --parallel 4 --prompt-cache-gib 0 \
  --host 127.0.0.1 --port 8941 --name exl3-qualification --keep-warm 0
```

Once `/v1/models` is ready, run the existing client from a checkout containing it; set `ARM=base` or `ARM=candidate`
for the currently running server:

```sh
python3 tools/bench_concurrent.py http://127.0.0.1:8941 exl3-qualification \
  --tokens 64 --reps 1 --temperatures 0,1.0 --levels 1,4 --alone --serial \
  --prompts /tmp/exl3-prompts/short-prompts.json --label "$ARM-short" --output "$ARM-short.json"
python3 tools/bench_concurrent.py http://127.0.0.1:8941 exl3-qualification \
  --tokens 64 --reps 1 --temperatures 0 --levels 1,4 --alone --serial \
  --prompts /tmp/exl3-prompts/long-prompts.json --label "$ARM-long" --output "$ARM-long.json"
```

The client can finish without rejecting unequal tokens. Require complete full-token hashes and compare both arms:

```sh
python3 tools/bench_exl3_compare.py base-short.json candidate-short.json
python3 tools/bench_exl3_compare.py base-long.json candidate-long.json
```

## Formal projection and batched-state checks

On the same 64-core M5, the projection contract passed 219 width/codebook registrations over 13 entrypoints:
all gates 1–4 passed, with 876/876 real GPU fault controls caught and no known findings.
The fixtures compare every output column to independently scalar-decoded FP64 rows and require the replaced
sixteen-warp reduction's exact bytes. KU2 prompt tiles additionally equal the KU1 template's bytes.
Structured full-sized matrices vary three K bands independently; precision plants repeat a random decoded tile.
The reported FP64 worst column and squared error are retained separately for each shape and comparator.
The native grouped-output bindings use 344 named `wo_a.slice` matrices: eight per backbone/DSpark block.
They are 4096-by-1024 dots (320 at ten half-bits, 24 at eight), already included in the geometry list;
joining the eight groups does not introduce a different unqualified reduction width.

```sh
zig build tf-kernel-contract -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8
./zig-out/bin/tf-kernel-contract --only exl3-metal --no-limits
./zig-out/bin/tf-kernel-contract --only exl3-metal --no-gates --offline /tmp/exl3-offline
```

The separate resource run passed all 13 live launches, with zero limit problems and both resource controls caught.
These projection pipelines are selected only with M5 tensor units; M1/M2 projections are explicitly unserved,
not claimed to have passed offline or runtime checks there. The offline resource controls did run.
`zig build test-exl3-contract-host test-kernel-contract -Dcpu=apple_m1 -j4 --summary all`
passed 232/232 host tests in 33/33 build steps.

Finite packed-key fixtures exposed a batched split-merge bug at 16 heads: the merger retained the 64-head defaults.
It now receives the actual head count and width. The same test checks four independent slots at 16, 32 and 64 heads,
including wrapped/long positions, both RoPE directions, KV glue and publication against the per-slot path.
`zig build test-deepseek-v41-rows -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j8 --summary all`
failed before the repair and then passed 6/6 tests in 41/41 build steps.
The fixture excludes E4 NaN codes from its finite-data leg and explicitly requires finite attention outputs.

The M4 source-built suite found a separate shared-down fusion mismatch: a separate dense call uses split sums,
while joining it to routed rows selects an unsplit body. Shared-down merging now requires the qualified
sixteen-warp FP32 path for both calls; other devices, row counts and incompatible knobs keep separate calls.
The focused `test-exl3-moe` run on M4 passed 3 tests and skipped the unserved merged-down path (4 total).
The full M4 run before this eligibility repair was 35 passed, 4 failed; three are the inherited parity failures above.
The focused M5 `test-exl3-moe` run passed 4/4 tests, including the merged-down path.
On M4, `test-deepseek-v41-lifetime` passed 2/2 tests in 43/43 build steps: backend host-state release and real Metal
draft-ID capacity growth without corrupting pending Markov picks. It does not prove a served MTP shutdown.

The later projection-order repair passes `test-exl3-gpu` on M4: **38 passed, one unserved tensor-path skip**.
The fused-QKV target also passes both tests, with **1792/1792 output cells bit-identical** to separate projections.
Indexed scalar windows now use the same split order and sufficient scratch as dense calls, with per-descriptor
column bounds. A scratch-size assertion tied to the previous allocation formula was removed; numerical bounds
and the valid parity assertions were not changed. M5 validation of the sixteen-warp path is recorded separately.
The M5 standalone suite passes **39/39**, and its fused-QKV target passes **2/2** with all 1792 cells exact.
The expanded tensor-unit contract covers every valid width/codebook tuple for the dense base and multi-row bodies:
**261 registrations pass gates 1–4; all 1044 planted controls are caught, with zero real failures or known findings**.
The static-specialized variants retain their narrower registered domain.

The scalar split-window shader is registered separately on both M4 and M5: **27 registrations and 108 planted
controls pass gates 1–4 per chip**. Together, the M5 tensor and scalar projection suites cover **288 registrations,
14 entrypoints and 1152 caught numerical controls**. No numerical tolerance was relaxed.
Scalar gate 5 passes with 32 launch threads, 1536 threadgroup bytes and a live 1024-thread limit on both chips.
The actual M1/M2 offline compiler also reports 1024 for this body; that is not M1/M2 runtime qualification.
Final integrated M5 projection/chain/native checks pass 47/47 tests in 63/63 build steps. Their resource checks
report no findings; each limits run catches both real oversized-resource controls.

## Chain contract on M4

The eight chain/MoE entrypoints pass the same gates 1–4 on an M4 Pro:

```text
SUMMARY 8 kernels: 8 pass gates 1-4, 0 fail, 0 known; 34 controls: 34 red, 0 missed
```

```sh
zig build test-exl3-chain-contract test-exl3-chain-host \
  -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j4 --summary all
```

The source-only runner uses the shared kernel-contract harness, without unrelated bundled shader dependencies.
It checks exact unfused GPU outputs, FP64 classes, explicit activation rounding, all three accumulator replay orders,
input immutability, guards and untouched padding. Cases include 384 experts, six picks plus shared rows, full
32768-column outputs, grouped output rotation, in-place RMS and independent long RoPE positions.
FP16/BF16 Hadamard and ordered-combine plants miss by at least 256 times the declared bound;
separate rounded-norm controls are caught by the exact unfused comparison.
The BF16 midpoint/subnormal oracle and accumulator plants also passed two host tests.
M4 reflection reports 1024 maximum threads for all eight pipelines; static threadgroup storage is 32 bytes for RMS,
4144 for down-combine, 2592 for QKV, and zero for the other five.
The M5 chain run also passed all eight registrations and 34 controls, with the same live resource limits.
The resource pass measured M1/M2 compiled maxima: 832 for out/in rotation, 928 for RMS and 1024 for the other six.
RMS launch sizing adapts to its pipeline limit. Both offline resource controls were caught; no M1/M2 runtime numeric claim is made.

The earlier CUDA host subset passed 5540 tests with 32 skips in 345/345 build steps, before the new chain launch records.
The current source is checked again after the final-row repair: **345/345 steps**, **131 reported passed tests**,
with unchanged work cached. This is the device-free `test-cuda-host` target on M5/macOS, not CUDA GPU execution or Linux-runtime qualification.

## Native batched rotation and packed-cache contract on M4

The native batched RoPE, KV-glue and ring-publication entrypoints pass gates 1–4, with separate forward/inverse
rotation registrations: **4 passed, zero failures or known findings; 13/13 actual GPU fault controls caught**.
`zig build test-deepseek-v41-rows-contract -Dcpu=apple_m1 -Doptimize=ReleaseSafe -j4 --summary all`
passed 4/4 tests in 4/4 build steps without requiring unrelated offline-compiled shaders.

The exact per-slot GPU comparisons cover 1–8 slots, 16/32/64 heads, padded strides and positions through 1048575.
Publication permutes the indirect ring addresses and checks each complete ring image, including all untouched slots.
Packed non-finite inputs retain the per-slot byte encoding; raw cache bytes are not claimed to propagate IEEE classes.
The FP16 norm plant misses by 1398101 times the bound across all three replay orders. No BF16 separation is claimed
for that plant: balanced positive-square sums can be exact, and FP8 storage hides small norm errors.
M4 reflection reports 1024 maximum threads for each entrypoint and static storage of 0/32/0 bytes respectively.
The M5 rotation/KV run also passed all four registrations and 13 controls, with the same live limits.
The resource pass measured 1024 compiled threads for all three entrypoints on both M1 and M2, with zero limit problems.
The native attention runner compiles separately on M4 and correctly skips its unserved tensor-unit test.
`test-deepseek-v41-attention-rows-host` passed two independent decoder/precision-plant tests in 3/3 build steps.

## Native attention and whole-model fidelity

The M5 dense/scan contract passes eight registrations and all 34 real GPU controls.
Cases cover 1–8 slots, 16/32/64 heads, independent positions, indirect owner planes and both split-merge launch forms.
The exact per-slot baseline is paired with independent packed-code classes and an FP32 value-cancellation oracle.
M5 reflection reports 1024 threads for both entrypoints and 0/8448 bytes of static threadgroup storage.

The cancellation oracle exposed a real bug in both scan bodies: the final 32-key tile could cross `j_end`,
so adjacent splits counted some keys repeatedly. Both now mask at the split boundary, including the per-slot baseline.
Two controls restore the old predicate and reproduce the failures at 32 and 8 splits.
Reserved E4M3 NaN codes and NaNs passed through BF16 rounding now retain their non-finite class.
The fixture's own dense decoder also needed braces to bind its `else` to the visibility test rather than the inner loop.
No tolerance was widened.

The full 128-token check is still **failing**, despite the attention repair:

| Measure | Before repair | After repair |
|---|---:|---:|
| NRMS against independent FP64 | 0.279974 | 0.172070 |
| Ratio to the fixed FP32/FP64 floor | 1.854916 | 1.140019 |
| Mean KL | 0.099111 | 0.062856 |
| Worst-row KL | 1.502481 | 2.151304 |
| Greedy disagreements | 7/128 | 6/128 |
| Execution-time disk reads | 4096 bytes | 0 bytes |

The clear/review boundaries remain 1.05/1.10; 1.140019 fails. Some remaining disagreements have substantial reference
margins, so they are not dismissed as ties. The original full FP32 and FP64 CPU captures are reused explicitly:
their equations, checkpoint and 128-token history are unchanged, and the floor's input/logit hashes were verified.
An offline comparator reproduces the original native NRMS and mean KL within 1e-10 before assessing the repaired GPU capture.
The GPU-only command's exit 0 is not a numeric verdict. Full source fingerprints, margins and memory scope are in
[fidelity-128.json](fidelity-128.json). CPU-interleaved and contiguous GPU timings are not a controlled speed comparison.

A later full 128-token replay after the projection-order repair gives the same fidelity metrics and six disagreements.
Its residency gate fails: **4096 execution-time disk bytes, zero pageins**. The earlier zero-read window is retained
as historical evidence, not substituted for this result. Neither model qualification nor publication is cleared.

The comparator now also runs the actual #2373 parent, `bec140d545aa50b765f26cf295b43179c6b2834f`:
all 4695 production-source files were checked byte-for-byte against that tree. The parent reproduces the table's
before-repair metrics; the candidate reproduces its after-repair metrics. Both retain the default 1048576-token
model context and run one teacher token per forward. Identical throwaway CLI harnesses reserve 128 workspace rows
instead of the unused 15872-row prefill capacity, which otherwise prevents the parent from fitting admission.
No production source or admission limit was changed for this control; the original refusal is preserved.
Both controlled execution windows read 4096 disk bytes and fail residency. The parent already fails the numerical
band; the candidate improves NRMS and mean KL but worsens worst-row KL, and still fails the unchanged band.
Teacher argmax differences are reported separately from the engine's own-path token-equality invariants.

## Served trained MTP and ownership

Draft embedding IDs grow for the requested arm before host writes without replacing pending Markov picks.
Model teardown releases Markov IDs, taps, conditioning and draft-logit scratch. The repaired scratch has one
error cleanup. Five directly executed lifetime cases cover host ownership, real Metal ID-buffer growth, trained-only
adaptive depth, committed-context suffix lookup and lookup precedence over a bound trained proposal.
The final M5 full corpus includes these cases; the matching policy fixtures also pass on M4.

The adaptive cap no longer falls to zero merely because lookup is disabled. Lookup no longer keeps a second token
history: it borrows the stream's canonical committed context, including accepted windows and restored prefixes.
A real matching suffix now takes precedence over paying for a trained proposal, without bypassing the cost governor.
The served measurements below predate these final policy/context/selection repairs; they are not current-source speed claims.

The first MTP-only served probe drafted nothing: backend capability admission incorrectly depended on lookup depth.
The repair reads the registered `TF_DSV41_MTP_DRAFTS` boolean once at model open and admits trained proposals when
DSpark weights exist, independently of lookup drafting. Proposal depth cannot exceed the four-ID proposal buffer.
The corrected run uses lookup unset, MTP and taps enabled, context 65536, four lanes and zero retained prompt cache.
The standard `bench_openai.py --tokens 256 --reps 3 --temperatures 0` command passes.
Twelve plain/drafted pairs (code completion and prose chat, greedy and temperature 0.8, three seeds) have identical
full token and output hashes. Eight c4 replies match their solo hashes; shared rounds remain plain by contract.
A cancelled stream does not change the following 256-token sampled-state-independent greedy reply.
There are no trained-proposer-unavailable warnings.

This is **not a speedup**: greedy code accepts 3/10 proposals and decodes at about 51.60 tok/s versus plain 52.23;
prose accepts 6/10 and measures about 52.19 versus 52.20. The cost governor stops unsuccessful drafting after eight
rounds. Sampled code also loses throughput. The no-slowdown gate remains uncleared.
These token-equality checks do not clear the independent whole-model fidelity failure above.

Two actual idle-unload cycles pass after drafter scratch ownership is repaired. The 256-token sampled reply's full
token and output hashes remain exact after reload; trained proposals are active before and after it.
Current physical footprint falls below 121 MB on both completed unloads, and graceful SIGTERM exits 0.
The probe waits for `/v1/load` to report `parked=true`, published only after engine close. HTTP 503 can precede
completed unload; lifetime-maximum footprint cannot measure released memory. Exact bytes are in `operator-checks.json`.

## Cold default policy and retained native window

The actual default-policy server passes `bench_cache.py --require-cold` at exactly **2048, 8192, 32768 and 65536
input tokens**, with context 131072, four lanes, zero retained prompt cache, MTP off and no prefill-step override.
Each case has one first request, two repetitions and four concurrent requests; every reply has 256 tokens,
zero cached input tokens and identical full token/output hashes. Graceful shutdown exits 0.
The token arrays and exact request bodies are retained with the run. Timings and fixture/token SHA256 values
are recorded in `operator-checks.json`; this synthetic fixture is not a language-quality comparison.

The fixture uses the pinned checkpoint's `tokenizer.json` through `tokenizers.Tokenizer.encode` with
`add_special_tokens=True`, repeats the following block 12000 times, then takes the first requested number of IDs:

```text
A public GPU performance fixture. A matrix product combines rows and columns. Each independent request has its own token history, cache positions and reproducible seed. The benchmark measures prefill latency and exact repeated replies, not language quality.
```

Each block ends with a newline. For each array, the exercised command is
`bench_cache.py URL MODEL --tokens 256 --repeats 2 --streams 4 --prompt-file ARRAY.json
--expected-prompt-tokens N --require-cold --output RECEIPT.json`.

The native server also admits context **1048576 with four lanes, a 2 GiB retained cache and trained MTP enabled**.
An 8192-token prompt restores 8064 cached tokens. Eight greedy/sampled plain-versus-drafted pairs, eight c4 replies
and the sampled reply following cancellation have exact full token/output hashes.
The original one-off probe exits 1 because it incorrectly demanded reuse for a 22-token prose prompt.
Existing cache rules require more than 4096 tokens for tail/end marks; those short requests correctly remain fresh.
The original HTTP records were checked against that eligibility rule without changing production code or thresholds.
The raw failed assertions remain recorded; they are not presented as a passing command.

Before the final-row repair below, the requested corpus, native and diagnostic binaries, and five lifetime cases succeed:
**977/977 build steps**; the build reports **1603 passed and 53 skipped tests**, with other unchanged steps cached.
The preceding 973-step run reported 1529 passed and 53 skipped; the earlier 970-step run reported 11884 passed and
114 skipped before the final policy/context repairs. One QKV diagnostic has one differing cell out of 1792 and
passes the unchanged NRMS, mismatch-count and worst-error bounds; the earlier standalone exact fixture is not
a universal bit-exact claim.

`tf-dsv41-run` honors an explicit `TF_DSV41_PREFILL_STEP` using the server's existing parsing semantics,
records the effective workspace rows, and retains its original unset default. The initial workspace-admitted
run then failed with `InvalidCounterSample`; deferred mHC was opening a span before encoding any work.
Spans now begin at the actual mHC encoding site. Invalid-sample checks remain unchanged.

Before the final-row repair, the native-context 1048576, 16-workspace-row profile passes for **one input row**:
grouping on/off and profiled last logits are exact, with zero execution disk reads and zero page-ins in that invocation.
The two- and four-row commands fail **`ProfileChangedLogits`**; both report zero execution reads/page-ins.
Those negative results remain in `final_strict_profile_matrix`; they are not overwritten.

The final fused expansion dispatched one row for `.last` without shifting its input/output views to the selected last row.
It expanded row zero, then the head collapsed a different, unexpanded row. All per-row views now select the requested row;
the kernel arithmetic, launch width and strict bit-equality assertions are unchanged, and no extra rows are computed.
The corrected two- and four-row full-model profiles **pass**, including exact grouped/ungrouped/profiled-last logits,
zero execution disk reads and zero page-ins. The one-row logits are also exact, but that invocation's independent
residency gate **fails**: 3,080,192 execution disk bytes and 45 page-ins. It is not a passing qualification command.
After the repair, the complete requested corpus is rebuilt: **977/977 steps**, **1588/1641 reported tests passed, 53 skipped**;
unchanged steps may be cached. The native and diagnostic binaries and five lifetime cases are included.
The fresh full 128-token GPU capture is byte-identical to the preceding matched capture, so the independently measured
**1.140019 band still fails the unchanged 1.10 bound**. This invocation also fails execution residency:
**154,669,056 disk bytes and 8,912 page-ins**. CPU references are explicitly reused, not recomputed or silently replaced.
The separate earlier 128-row fidelity and execution-read residency failures remain recorded above.

Before the Sinkhorn repair below, the contribution's served path is checked at context **1048576**, four lanes and a 4 GiB retained cache:
**41 completed replies** retain full token/output hashes across plain/drafted, greedy/sampled and solo/shared paths.
An 8192-token request is cancelled after generated content; its following sampled reply matches the plain control.
Two actual idle unload/reload cycles preserve that sampled reply and release the model's bulk memory:
current process footprint falls from 226,962,856,920 / 222,969,100,760 bytes loaded to 147,244,328 / 149,833,024 bytes fully parked.
The parked flag and current footprint must both be observed; no lifetime-maximum RSS substitution is used.
With lookup disabled, real trained MTP proposes **370 tokens** and preserves the same full reply hashes.
Both mixed and trained-only servers exit gracefully. These current-source invariants do not waive the numerical band,
residency, source-matched no-slowdown, release or unavailable-chip/parent gates.

### Fused Sinkhorn reduction-order repair

The fused boundary used row XOR offsets 1,2 and column offsets 4,8, unlike the unfused split and pinned reference.
It now uses their FP32 butterfly order. An isolated retained-argument regression fails before the repair and passes
after it, with all 1728 FP32 pre/post/comb coefficients bit-identical to the unfused split and unchanged FP64 bounds.
The reference target reports **41 passed, 15 skipped of 56**. The complete requested build succeeds again:
**977/977 steps**, **422/435 reported tests passed, 13 skipped**; other unchanged work is cached.

The unchanged full 128-token comparison now clears the numerical band: **0.967305**, below the existing 1.05 clear
boundary. Mean KL is **0.042165**, worst-row KL **0.832581**, with **five** teacher argmax differences; all margins
are recorded in [fidelity-128.json](fidelity-128.json). CPU reference files and thresholds are unchanged.
The command still exits 1 on **4096 execution disk bytes**, with zero page-ins. Strict one-, two- and four-row
profile commands all pass exact logits and zero-read/zero-page-in execution on this source. These short windows
do not replace the failed full-history residency result. Post-repair serving also passes 41 full-hash replies, cancellation recovery,
two real idle unload/reloads and trained-only lookup0 with 367 actual proposals; both servers exit0. These runs were under filesystem
tracing, not controlled performance measurements. The fused boundary's contract registration and the final tree's results follow.

### Final qualification of source `efdf7a464e`

Every command below ran on that exact tree, synced from the commit, on an M5 Ultra (M4 Pro where stated).

- Build and test corpus, reported counts: **977/977 steps; 4760 of 4845 tests passed, 85 skipped**. The device-free CUDA host
  subset passes **345/345 steps, 131 tests**.
- Kernel contract on the M5 Ultra: `--only dsv41` **102 kernels pass gates 1-4**, 237 controls red as planted, 0 missed;
  `--only exl3` **296 kernels pass**, 1186 controls red as planted, 0 limit problems. On the M4 Pro: dsv41 94 and exl3 35 pass.
  The six dsv41 offline M1/M2 limit problems (`xp_quant`, `fp4pack`, `fp4rt`, `hcmix`, `hcpost`, `joinpost`) are identical at the
  parent `bec140d545` (89 kernels pass there), so this delta does not introduce them. The unfiltered contract aborts in
  `metal_lfm3_attention_sums.zig:39` (integer overflow in `valuesDepth`), a file this delta does not touch, so it was not completed.
  The exhaustive rotary sweep covers 302,008,320 cases.
- Strict one-, two- and four-row profiles return identical grouped and ungrouped logits. The first run, after the build had
  evicted the file cache, read 7,700,480 / 7,782,400 / 15,450,112 bytes (96 / 97 / 193 page-ins) and failed the residency gate.
  Two consecutive reruns pass for all three, with 0 bytes and 0 page-ins. The gate therefore measures cache temperature at
  execution time; a warm pass does not prove that every Engram table page is resident.
- The unchanged full 128-token comparison: band ratio **0.967305** (clear at 1.05), mean KL 0.042165, worst-row KL 0.832581,
  five teacher argmax differences; execution residency passes with 0 bytes and 0 page-ins in that warm run.
- Served checks on this tree, as above: 41 full-hash replies, cancellation, two idle unload/reload cycles (loaded
  226,963,086,272 / 222,898,212,784 bytes, parked 749,290,720 / 151,176,440), trained-only lookup0 with 367 proposals.
- Parent against this tree, same machine, 4096 workspace rows and prefill step 4096 in both arms, 64 tokens, three repetitions.
  Solo, serial and concurrent hashes are equal inside each build. Tokens differ between the builds because the arithmetic
  changed, so `tools/bench_exl3_compare.py` refuses the pair and this is an observation, not a no-slowdown qualification:

| Cell | Parent tok/s | This tree tok/s |
|---|---|---|
| short, 1 stream | 40.6 | 52.5 |
| short, 4 streams, aggregate / steady | 43.5 / 61.0 | 59.0 / 93.9 |
| long, 1 stream | 38.5 | 49.1 |
| long, 4 streams, aggregate / steady | 6.1 / 58.1 | 11.1 / 88.4 |
| long TTFT max, 1 / 4 streams | 12.34 s / 48.87 s | 6.53 s / 26.47 s |

Still open: the parent's review gates (code provenance and copy rule, integration), ATC coverage (the board is incomplete),
runtime on M1, M2 and M3 chips, and a source-matched no-slowdown comparator against the parent.



## Limits

These are single-run synthetic correctness/timing probes, not representative throughput or quality gates.
The long fixture repeats public prose; the short fixture mixes completion and chat requests.
The client warms the server; disabling prompt retention does not make disk/Engram pages cold.
`aggregate_tps` sums per-stream decode rates; use `end_to_end_tps` for completed tokens per concurrent request window.
Historical pinned-workspace comparisons, current default-policy cold runs and explicit trained-MTP checks have
different source/settings scopes; none substitutes for another. The final section above is the current scope. Unavailable-chip
runtime checks, a token-equal no-slowdown comparator and the parent's remaining integration gates remain uncleared.
The new family controls use shared strict validation, not a global profile.
