# Decode equivalence follow-up

The earlier 64-token ABAB gate established output equality for its cases and small observed timing differences. It did not establish performance equivalence. This follow-up fixes its protocol before measurement and keeps those earlier observations separate.

## Fixed gate

The [protocol](protocol.json) specifies six matched pairs of fresh baseline and candidate servers, with order AB, BA, BA, AB, AB, BA. Three guarded windows each contain two pairs, with production restored and health-checked after every window. The binary, checkpoint and GPU-image hashes remain those in the [existing environment receipt](../model/environment.json).

Each server produces 256-token replies for three public chat fixtures, temperatures zero and one, and one and two streams. There are two repetitions per cell, one excluded warmup and three excluded sampled solo references: 40 responses per server, 480 in total. The additional seed-1235 references and measured single-stream cases provide solo comparisons for every concurrent output. Full tokens, text, cache evidence and timings are retained. The earlier 64-token drafted/plain check remains separate; this follow-up makes no 256-token plain equality claim.

For each of the twelve cells, the analysis uses six paired log ratios of candidate/baseline per-server median aggregate decode rates. A two-sided 95% Student t interval with five degrees of freedom is exponentiated back to a rate ratio. Practical equivalence requires every interval to lie wholly within `[0.98, 1.02]`, together with complete observations and exact output equality. These are per-cell intervals, not simultaneous confidence intervals; the all-cells gate is an intersection-union decision.

The paired t calculation assumes independent, approximately normal block log ratios. Six blocks can leave a wide interval. A bound-crossing interval is inconclusive; an interval wholly outside the band is not equivalent. Neither outcome is automatically called a regression. There is no early acceptance, sample extension or replacement pair selected from the results. A safety or exactness failure stops the experiment.

## Results

**All twelve cells pass the fixed ±2% equivalence gate.** The geometric mean paired changes range from −0.561% to +0.222%. Across the twelve per-cell intervals, the lowest bound is −1.680% and the highest is +1.073%. This supports practical decode-throughput equivalence within the chosen tolerance on this checkpoint, hardware and request grid. It does not assert identical timings or equivalence on untested workloads.

All 480 responses are complete, with zero unequal or unverified outputs and zero cached tokens: 432 measured responses plus 48 excluded warmup/solo-reference responses, totaling 122,880 reply token IDs. Full tokens and content agree across repetitions, solo/concurrent cases, builds and blocks. Six tokenization snapshots per run also agree. The previous 64-token plain equality gate remains separate.

Baseline and PR rate columns are arithmetic means of six per-run medians. Paired change and interval columns use the geometric mean of the six paired ratios, so they need not equal the ratio of the displayed arithmetic means. Acceptance uses unrounded intervals.

| Fixture | T | Streams | Baseline tok/s | PR tok/s | Paired change | 95% interval | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Code | 0 | 1 | 63.44 | 63.39 | -0.080% | [-0.539%, +0.381%] | Pass |
| Code | 0 | 2 | 111.79 | 111.17 | -0.561% | [-1.490%, +0.376%] | Pass |
| Code | 1 | 1 | 62.50 | 62.47 | -0.049% | [-0.368%, +0.271%] | Pass |
| Code | 1 | 2 | 81.49 | 81.67 | +0.222% | [-0.393%, +0.841%] | Pass |
| 34-message chat | 0 | 1 | 61.02 | 61.07 | +0.090% | [-0.883%, +1.073%] | Pass |
| 34-message chat | 0 | 2 | 85.20 | 84.85 | -0.412% | [-1.377%, +0.563%] | Pass |
| 34-message chat | 1 | 1 | 54.82 | 54.80 | -0.027% | [-0.177%, +0.122%] | Pass |
| 34-message chat | 1 | 2 | 54.88 | 54.95 | +0.113% | [-0.114%, +0.341%] | Pass |
| Prose | 0 | 1 | 56.27 | 56.30 | +0.058% | [-0.142%, +0.260%] | Pass |
| Prose | 0 | 2 | 103.85 | 103.32 | -0.514% | [-1.680%, +0.666%] | Pass |
| Prose | 1 | 1 | 47.91 | 47.86 | -0.100% | [-0.547%, +0.349%] | Pass |
| Prose | 1 | 2 | 63.09 | 63.17 | +0.135% | [-0.330%, +0.602%] | Pass |

The [summary](summary.json) contains each raw per-run median and paired ratio. [A1](A1.json), [B1](B1.json), [B2](B2.json), [A2](A2.json), [B3](B3.json), [A3](A3.json), [A4](A4.json), [B4](B4.json), [A5](A5.json), [B5](B5.json), [B6](B6.json), and [A6](A6.json) retain the full response evidence in execution order. JSON whitespace was compacted for publication; all parsed values are unchanged. Original byte hashes and published artifact identities are recorded in [environment.json](environment.json).

## Execution and verification

Protocol commit `df80cdc60` preceded measurement. The complete fixed sample was collected without replacement pairs, extra runs or interim confidence calculations. All 27 initial/before/after observations have identical source, native binary, GPU asset, checkpoint, client and controller identity. The 103 flat GPU images remain identical between baseline and candidate. Native code was not rebuilt for this follow-up.

All three [window outcomes](window-outcomes.json) record success with no cleanup errors. The existing guard remained active; no guard/OOM event was observed. Production was restored and freshly health-checked after each window, including the final one. Memory observations are boundary measurements and server-reported gauges, not continuous whole-host footprint sampling.

The frozen comparator reconstructs hashes, output equality, chronology, every raw timing and cell median before calculating the six-pair intervals. An independent high-precision Decimal calculation reconstructs all twelve intervals directly from the raw timestamps. Offline validation covers the full 480-response shape, five statistical cases, seventeen malformed-receipt rejections, missing-pair rejection and cross-run drift. Partial/single-run modes produce no intervals.

## Reproduction

Use the pinned native builds and server command from the [earlier model receipt](../model/README.md). Start a fresh server in the exact order specified by `protocol.json`; restore production between its three windows under the machine's existing guard. For example, on the first baseline server:

```sh
python3 docs/receipts/template-accumulation/decode-equivalence/client.py \
  --url http://127.0.0.1:30001 --tools tools --block 1 --side A --output A1.json \
  --shared-client docs/receipts/template-accumulation/model/client.py
```

Repeat for all twelve named runs with the corresponding pinned baseline or candidate binary. After all six pairs are collected:

```sh
python3 docs/receipts/template-accumulation/decode-equivalence/compare.py /path/to/receipts --output summary.json
python3 docs/receipts/template-accumulation/decode-equivalence/compare.py --self-test
```

The statistical scope is served aggregate decode on the twelve cases above. TTFT, longer contexts, resumed caching, other checkpoints, more than two streams and Metal are not qualified by this equivalence result.
