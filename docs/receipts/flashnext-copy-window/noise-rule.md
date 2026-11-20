| invocation | prompt | case | streams | M1 | M2 | B0a | B0b | B1a | B1b | B0vsM | B1vsM | B1vsB0 | tokens a round, before / after: B0vsM, B1vsM, B1vsB0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai_t0 | fibonacci-raw |  | 1 | 72.9 | 72.8 | 72.6 | 72.7 | 72.1 | 72.7 | -0.3% level | -0.6% level | -0.3% level | 5.22 / 5.22 / 5.22 / 5.22 / 5.22 / 5.22 |
| openai_t0 | gpu-chat-no-think |  | 1 | 46.7 | 46.7 | 46.6 | 46.6 | 46.3 | 46.6 | -0.2% level | -0.5% level | -0.4% level | 2.81 / 2.81 / 2.81 / 2.81 / 2.81 / 2.81 |
| conc_t0 | chat |  | 1 | 46.7 | 46.6 | 46.8 | 46.6 | 46.5 | 46.6 | +0.1% level | -0.2% level | -0.3% level | 2.81 / 2.81 / 2.81 / 2.81 / 2.81 / 2.81 |
| conc_t0 | chat |  | 2 | 86.6 | 86.6 | 87.0 | 86.7 | 86.2 | 86.4 | +0.3% level | -0.3% level | -0.6% level | 2.51 / 2.51 / 2.51 / 2.51 / 2.51 / 2.51 |
| conc_t0 | chat |  | 4 | 156.0 | 156.0 | 156.8 | 156.3 | 156.0 | 156.0 | +0.4% level | +0.0% level | -0.4% level | 2.51 / 2.51 / 2.51 / 2.51 / 2.51 / 2.51 |
| conc_t0 | chat |  | 6 | 210.4 | 210.4 | 211.4 | 210.7 | 210.2 | 210.3 | +0.3% level | -0.1% level | -0.4% level | 2.51 / 2.51 / 2.51 / 2.51 / 2.51 / 2.51 |
| conc_t0 | chat |  | 8 | 257.2 | 257.2 | 258.9 | 257.5 | 256.8 | 257.4 | +0.4% level | -0.0% level | -0.4% level | 2.51 / 2.51 / 2.51 / 2.51 / 2.51 / 2.51 |
| conc_t0 | code |  | 1 | 72.9 | 72.7 | 72.6 | 72.4 | 72.3 | 72.6 | -0.4% level | -0.5% level | -0.1% level | 5.22 / 5.22 / 5.22 / 5.22 / 5.22 / 5.22 |
| conc_t0 | code |  | 2 | 138.5 | 138.3 | 137.9 | 138.4 | 137.8 | 138.1 | -0.2% level | -0.3% level | -0.1% level | 4.83 / 4.83 / 4.83 / 4.83 / 4.83 / 4.83 |
| conc_t0 | code |  | 4 | 242.7 | 242.4 | 242.1 | 242.8 | 242.0 | 242.8 | -0.0% level | -0.1% level | -0.0% level | 4.83 / 4.83 / 4.83 / 4.83 / 4.83 / 4.83 |
| conc_t0 | code |  | 6 | 307.0 | 308.0 | 309.2 | 306.8 | 307.1 | 307.7 | +0.2% level | -0.0% level | -0.2% level | 4.49 / 4.49 / 4.49 / 4.49 / 4.49 / 4.49 |
| conc_t0 | code |  | 8 | 371.5 | 373.0 | 373.5 | 371.7 | 372.1 | 371.6 | +0.1% level | -0.1% level | -0.2% level | 4.49 / 4.49 / 4.49 / 4.49 / 4.49 / 4.49 |
| copy | copy.txt | changed | 1 | 246.8 | 249.2 | 250.2 | 250.7 | 287.9 | 287.9 | +1.0% level | +16.1% faster | +15.0% faster | 23.27 / 23.27 / 23.27 / 28.44 / 23.27 / 28.44 |
| copy | copy.txt | verbatim | 1 | 255.5 | 256.8 | 256.0 | 257.7 | 295.8 | 296.2 | +0.3% level | +15.6% faster | +15.2% faster | 23.27 / 23.27 / 23.27 / 28.44 / 23.27 / 28.44 |

Verdicts: B0vsM: 0 faster, 14 level, 0 slower, 0 unmeasured; B1vsM: 2 faster, 12 level, 0 slower, 0 unmeasured; B1vsB0: 2 faster, 12 level, 0 slower, 0 unmeasured

| comparison | streams | cells | mean change | faster | level | slower |
|---|---|---|---|---|---|---|
| B0vsM | 1 | 2 | -0.15% | 0 | 2 | 0 |
| B0vsM | 2 | 2 | +0.05% | 0 | 2 | 0 |
| B0vsM | 4 | 2 | +0.16% | 0 | 2 | 0 |
| B0vsM | 6 | 2 | +0.24% | 0 | 2 | 0 |
| B0vsM | 8 | 2 | +0.24% | 0 | 2 | 0 |
| B1vsM | 1 | 2 | -0.35% | 0 | 2 | 0 |
| B1vsM | 2 | 2 | -0.34% | 0 | 2 | 0 |
| B1vsM | 4 | 2 | -0.03% | 0 | 2 | 0 |
| B1vsM | 6 | 2 | -0.05% | 0 | 2 | 0 |
| B1vsM | 8 | 2 | -0.07% | 0 | 2 | 0 |
| B1vsB0 | 1 | 2 | -0.20% | 0 | 2 | 0 |
| B1vsB0 | 2 | 2 | -0.39% | 0 | 2 | 0 |
| B1vsB0 | 4 | 2 | -0.19% | 0 | 2 | 0 |
| B1vsB0 | 6 | 2 | -0.29% | 0 | 2 | 0 |
| B1vsB0 | 8 | 2 | -0.31% | 0 | 2 | 0 |

Rerun (one per flagged cell; a repeat of slower is a loss):
No cell was slower in the first pair, so no rerun was taken.
Losses: 0
