| invocation | prompt | streams | M1 | B0a | B1a | B1b | B0b | M2 | B0 vs M | B1 vs M | B1 vs B0 | tokens a round M / B0 / B1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai_t0 | fibonacci-raw | 1 | 73.9 | 71.8 | 72.1 | 73.0 | 72.7 | 72.1 | -1.0% level | -0.7% level | +0.3% level | 5.224 / 5.224 / 5.224 |
| openai_t0 | gpu-chat-no-think | 1 | 47.3 | 46.0 | 46.1 | 46.7 | 46.6 | 46.1 | -0.7% level | -0.6% level | +0.2% level | 2.813 / 2.813 / 2.813 |
| conc_t0 | chat | 1 | 46.5 | 45.9 | 46.7 | 46.7 | 46.6 | 46.2 | -0.2% level | +0.8% level | +1.0% level | 2.813 / 2.813 / 2.813 |
| conc_t0 | chat | 2 | 86.6 | 85.7 | 87.0 | 87.1 | 86.3 | 86.1 | -0.4% level | +0.8% level | +1.2% faster | 2.510 / 2.510 / 2.510 |
| conc_t0 | chat | 4 | 156.6 | 154.7 | 158.0 | 158.2 | 156.4 | 155.4 | -0.3% level | +1.3% faster | +1.6% faster | 2.510 / 2.510 / 2.510 |
| conc_t0 | chat | 6 | 210.7 | 208.6 | 213.7 | 213.6 | 210.9 | 209.3 | -0.1% level | +1.7% faster | +1.9% faster | 2.510 / 2.510 / 2.510 |
| conc_t0 | chat | 8 | 257.9 | 255.3 | 262.0 | 262.2 | 255.1 | 255.9 | -0.7% level | +2.0% faster | +2.7% faster | 2.510 / 2.510 / 2.510 |
| conc_t0 | code | 1 | 73.5 | 71.9 | 72.9 | 72.7 | 72.6 | 72.2 | -0.8% level | -0.1% level | +0.8% level | 5.224 / 5.224 / 5.224 |
| conc_t0 | code | 2 | 139.2 | 137.2 | 139.0 | 138.8 | 137.9 | 137.1 | -0.4% level | +0.5% level | +1.0% level | 4.830 / 4.830 / 4.830 |
| conc_t0 | code | 4 | 243.7 | 239.8 | 245.5 | 245.2 | 242.1 | 240.7 | -0.5% level | +1.3% faster | +1.8% faster | 4.830 / 4.830 / 4.830 |
| conc_t0 | code | 6 | 307.5 | 304.4 | 312.1 | 312.0 | 307.8 | 305.5 | -0.1% level | +1.8% faster | +1.9% faster | 4.491 / 4.491 / 4.491 |
| conc_t0 | code | 8 | 373.9 | 369.1 | 379.3 | 379.8 | 371.9 | 369.0 | -0.3% level | +2.2% faster | +2.4% faster | 4.491 / 4.491 / 4.491 |

Verdicts (first pair): B0 vs M: 0 faster, 12 level, 0 slower; B1 vs M: 6 faster, 6 level, 0 slower; B1 vs B0: 7 faster, 5 level, 0 slower

| streams | cells | B0 vs M mean | verdicts | B1 vs M mean | verdicts |
|---|---|---|---|---|---|
| 1 | 2 | -0.52% | 0 faster, 2 level, 0 slower | +0.34% | 0 faster, 2 level, 0 slower |
| 2 | 2 | -0.42% | 0 faster, 2 level, 0 slower | +0.68% | 0 faster, 2 level, 0 slower |
| 4 | 2 | -0.40% | 0 faster, 2 level, 0 slower | +1.32% | 2 faster, 0 level, 0 slower |
| 6 | 2 | -0.12% | 0 faster, 2 level, 0 slower | +1.77% | 2 faster, 0 level, 0 slower |
| 8 | 2 | -0.46% | 0 faster, 2 level, 0 slower | +2.10% | 2 faster, 0 level, 0 slower |

Rerun (one per flagged cell; a repeat of slower is a loss):
No cell was slower in the first pair, so no rerun was taken.
Losses: 0
