# Decode equivalence follow-up

The earlier 64-token ABAB gate established output equality for its cases and small observed timing differences. It did not establish performance equivalence. This follow-up fixes its protocol before measurement and keeps those earlier observations separate.

## Fixed gate

The [protocol](protocol.json) specifies six matched pairs of fresh baseline and candidate servers, with order AB, BA, BA, AB, AB, BA. Three guarded windows each contain two pairs, with production restored and health-checked after every window. The binary, checkpoint and GPU-image hashes remain those in the [existing environment receipt](../model/environment.json).

Each server produces 256-token replies for three public chat fixtures, temperatures zero and one, and one and two streams. There are two repetitions per cell, one excluded warmup and three excluded sampled solo references: 40 responses per server, 480 in total. The additional seed-1235 references and measured single-stream cases provide solo comparisons for every concurrent output. Full tokens, text, cache evidence and timings are retained. The earlier 64-token drafted/plain check remains separate; this follow-up makes no 256-token plain equality claim.

For each of the twelve cells, the analysis uses six paired log ratios of candidate/baseline per-server median aggregate decode rates. A two-sided 95% Student t interval with five degrees of freedom is exponentiated back to a rate ratio. Practical equivalence requires every interval to lie wholly within `[0.98, 1.02]`, together with complete observations and exact output equality. These are per-cell intervals, not simultaneous confidence intervals; the all-cells gate is an intersection-union decision.

The paired t calculation assumes independent, approximately normal block log ratios. Six blocks can leave a wide interval. A bound-crossing interval is inconclusive; an interval wholly outside the band is not equivalent. Neither outcome is automatically called a regression. There is no early acceptance, sample extension or replacement pair selected from the results. A safety or exactness failure stops the experiment.

## Execution status

Protocol and client are frozen before measurement. Results will be added after the fixed run and independent verification. No performance-equivalence claim is made yet.
