# PR 1381 standalone review fixes

## Composition with requested main `cb036e00f`

The shared-reservation stack is integrated without restoring the rejected shared-round guard.
Historical grants and new reservations remain distinct:

1. `granted_depth` reads the last **verified** grant, its short-grant flag and configured slack. Its periodic
   zero-grant probe is evaluated once. This bounds what the next reservation may request.
2. The shared allocator reserves rows using the drafter's cached chain chances, even when that historical
   depth bound exists. The reservation decision queries a stateful copy proposer once.
3. The actual reservation is capped to the hold pass's padded capacity, then `payHolds` prices only its
   positive holds. Expected gain is recomputed from the usable chance prefix. These steps only reduce depths.
4. A fresh zero reservation therefore never offers a hold. A historical zero grant may still probe, but only
   if the new reservation supplies positive rows. Context absorption continues before the depth-zero skip,
   so later holds retain continuous history.

`limit_shared_drafts` keeps upstream's legacy head-step-cost meaning. LFM2's target facts set it, but upstream's
attached-drafter classification excludes its legacy cap for node prices and confidence floors. The opt-in
`granted_depth` path composes its own bound with reservations independently; it neither reinterprets
`Stream.granted` nor replaces the drafter's chain chances with generic rates. Confidence and other-family
legacy caps retain their existing paths.

The focused regressions cover bounds, zero-to-positive transitions, chance pricing, padded-pass capacity,
reserved gain, probe cadence and copy lookup. The mechanical merge without the composition fails seven of
the nine new regressions. An independently compiled host API smoke also reproduces an unpaid hold pass
before the fix: four positive holds survive a 1000 ms hold price. With the composition they are all rejected;
profitable holds remain and zero reservations stay zero, without mutating historical grant fields.

Current-main scoped checks use Zig 0.17.0 and `-Dcpu=apple_m1 -Dversion=1.2.0`:

```sh
zig build native test-native test-shared-lane-regressions-host \
  test-shared-head-cost-metal-host test-draft-metal-host test-contract-images-host \
  -Dcpu=apple_m1 -Dversion=1.2.0 -j2
```

The lane suite passes 299/299, including the complete absorbed-history comparison before every later hold;
the Metal family head-cost checks pass 48/48, draft-host checks 11/11, and native checks 93 with three skips.
The pinned main checker (`985356b424269b8d11e8a486223cf3c9aa1e726bb4761ad4a414482352024367`)
reports 424 inherited findings on both trees, zero new. Only obsolete comparisons against deliberately wasted
holds/full-block counts were removed: reservations now optimize both sides of those comparisons. Token
equality, copy-skip, bound, price and zero-row gates remain.

The historical results below retain their original source and device scope. They do not qualify the new
DSpark chain's complete no-slowdown matrix.

## Historical device receipt

The lane changes are rebased onto main `d33dcfc767ee093393c6bfa00b3949bd7b4d8e88`, without the rejected guard ancestry. Source `745fe1088ab7293b5bc4f28c6cdedb989e83ec4f` retains the opt-in chance/grant/hold features, reuses the hold-order buffer, interpolates adjacent measured depths and queries a stateful copy proposer only once per granted-depth budget decision.

Host checks: `zig build test-shared-lane-regressions-host test-draft-metal-host -Dcpu=apple_m1 -Dversion=1.1.8 --summary all` passes 57/57 steps, 290/290 tests. Plain `zig build -Dcpu=apple_m1 -Dversion=1.1.8 --summary all` passes 236/236 steps. Zig is 0.17.0. Formatting and whitespace pass. The same lean checker reports 425 findings on current main and this branch, zero new.

The standalone PR changes no family adapter or GPU operator. DSpark device measurements require the downstream adapter from PR 1394. The local integration used here is `f73bb5590cc458b9b21b86352d65f23b76cac7c1` (PR 1394 head `98f5cc72b` merged with the same current main). Its tree is `1ce4a92c3ebd0ae6637872af2b3aec3fdbf1e73b`, reproducible by merging those two public parents. Its `config.zig`, `depth.zig`, `drafted.zig`, `drafter.zig`, `hold_price.zig` and `stream.zig` are byte-identical to this standalone head; it also contains PR 1394's final-buffer allocation cleanup and pipelined-code extraction. These measurements are explicitly integration evidence, not a claim that standalone main already loads DSpark.

Device: Apple M5 Ultra, 256 GiB, macOS 27.0 (26A428), Metal 32023.921.6; native binaries built with Zig 0.17.0, `-Dcpu=apple_m1 -Dversion=1.1.8`. Target: `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` revision `667408d494bde191c2cb9dd4f4549827816b9275`. Drafter safetensors SHA-256: `7d06a78556b0b1f386134508f09b00a3a7e0e5e0ec2cc73119229f4ef7c9b299`. BF16 drafter mode, one rank, context 4096, parallel capacity 16, prompt cache zero, greedy seed 1234, ignore EOS, 256 generated tokens. Code and prose prompts are retained verbatim in the driver. Each arm gets a 32-token warm-up per prompt. Four rotated runs compare main without an external drafter, standalone without an external drafter, and the DSpark integration. Main's normal suffix copying remains enabled. The integration also checks explicit `draft:false` oracles.

Hold retention is measured on a separate diagnostic build: count positive stream budgets entering and leaving `payHolds`, after the one-pass cap. Report both kept/offered and kept/sum-of-stream-budget-opportunities. The latter includes zero-budget slots in calls that reach this pricing path; it does not count rounds that never call this path. Zero-budget streams do not enter its denominator, probes follow normal pricing, and tail rounds can have fewer live streams than the request batch. Per-call events are retained. Diagnostic output is excluded from throughput runs.

To reproduce, use `device_report.py` with Python's standard library and the `qualification.py` helper from PR 1394's receipts. Set `TF_QUAL_HELPER`, `TF_QUAL_DRAFTER`, `TF_REPORT_MAIN`, `TF_REPORT_STANDALONE`, `TF_REPORT_INTEGRATION` (uninstrumented), and `TF_REPORT_DIAGNOSTIC`. Optional `TF_REPORT_OUT` controls output. The driver uses `/tmp/tensorfold-model.lock`, refuses an existing recognized model process through the helper, and aborts on memory-pressure admission changes. Reserve the GPU exclusively. The diagnostic build adds only before/after positive-budget counters around `payHolds`; the executed instrumentation is retained in the receipt.

No new M1/M2/M3/M4, CUDA or HIP device qualification is claimed. Cold-prefill scaling, long-context and cancellation were not rerun for this standalone review report; historical PR 1394 checks are separately scoped to their recorded sources.

## Device results

All 72 throughput cells, six diagnostic batches, and explicit `draft:false` oracles agree token for token. All 13 servers shut down gracefully. Rates are median aggregate decode tokens/s across four rotations; diagnostic logging is absent from these timings.

| Prompt | Streams | Main | Standalone | DSpark integration | Integration vs main | Holds kept/offered | Kept/budget slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 4 | 651.92 | 652.29 | 640.37 | -1.77% | 12/742 | 12/886 (1.35%) |
| code | 8 | 830.98 | 831.20 | 815.77 | -1.83% | 28/1226 | 28/1836 (1.53%) |
| code | 16 | 831.29 | 829.54 | 888.75 | +6.91% | 374/1891 | 374/3258 (11.48%) |
| prose | 4 | 623.40 | 623.08 | 616.75 | -1.07% | 0/960 | 0/996 (0.00%) |
| prose | 8 | 804.91 | 804.87 | 795.60 | -1.16% | 40/40 | 40/2006 (1.99%) |
| prose | 16 | 799.75 | 798.35 | 873.10 | +9.17% | 225/3260 | 225/3813 (5.90%) |

Standalone/no-external-drafter changes are between -0.21% and +0.06% versus main in this run. DSpark holds are sparse at four and eight streams, and its integration loses 1.07–1.83% there; it gains 6.91% code / 9.17% prose at 16 streams. These results document the remaining downstream pricing/performance tradeoff rather than claiming universal gains. In prose at eight streams, every offered positive hold is kept, but holds occur in only 1.99% of recorded budget slots; kept/offered alone would conceal that sparsity.

[Request-level data, per-call counters, binary/source hashes and diagnostic patch](device-report.json). The interpolation regression fails on the old PR head (expected 17.5 ms, obtained 24 ms) and passes here.
