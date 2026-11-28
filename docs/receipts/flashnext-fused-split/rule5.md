# The noise rule, cell by cell

M1 and M2 are main, F1 and F2 the branch with `TF_FLASHNEXT_FUSED_SPLIT=1`, U1 and U2 the branch with it unset, run
in the order M1 F1 U1 M2 U2 F2. Aggregate tok/s; each verdict compares the mean of that branch arm's two runs with the
mean of main's two. Tokens a round are the tools' tokens a streamed chunk.

| invocation | prompt | temp | seed | streams | M1 | M2 | F1 | F2 | U1 | U2 | F vs main | U vs main | tokens a round M / F / U |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai_t0 | fibonacci-raw | 0.0 |  | 1 | 72.5 | 72.4 | 72.7 | 72.6 | 72.6 | 71.7 | +0.3% level | -0.4% level | 5.22 / 5.22 / 5.22 |
| openai_t0 | gpu-chat-no-think | 0.0 |  | 1 | 46.6 | 46.5 | 46.7 | 46.6 | 46.5 | 46.0 | +0.2% level | -0.6% level | 2.81 / 2.81 / 2.81 |
| conc_t0 | chat | 0.0 |  | 1 | 46.5 | 46.6 | 46.8 | 46.6 | 46.6 | 46.0 | +0.3% level | -0.5% level | 2.81 / 2.81 / 2.81 |
| conc_t0 | chat | 0.0 |  | 2 | 86.4 | 86.3 | 87.0 | 86.8 | 86.7 | 85.8 | +0.6% level | -0.1% level | 2.51 / 2.51 / 2.51 |
| conc_t0 | chat | 0.0 |  | 4 | 155.7 | 154.7 | 157.0 | 156.8 | 156.2 | 154.5 | +1.1% faster | +0.1% level | 2.51 / 2.51 / 2.51 |
| conc_t0 | chat | 0.0 |  | 6 | 210.0 | 208.1 | 212.8 | 212.7 | 210.1 | 208.4 | +1.8% faster | +0.1% level | 2.51 / 2.51 / 2.51 |
| conc_t0 | chat | 0.0 |  | 8 | 257.2 | 254.2 | 260.5 | 260.3 | 258.1 | 254.8 | +1.8% faster | +0.3% level | 2.51 / 2.51 / 2.51 |
| conc_t0 | code | 0.0 |  | 1 | 72.6 | 72.3 | 72.8 | 72.5 | 72.5 | 71.6 | +0.3% level | -0.6% level | 5.22 / 5.22 / 5.22 |
| conc_t0 | code | 0.0 |  | 2 | 138.1 | 137.8 | 139.0 | 138.5 | 138.0 | 136.4 | +0.6% level | -0.5% level | 4.83 / 4.83 / 4.83 |
| conc_t0 | code | 0.0 |  | 4 | 242.7 | 242.4 | 244.9 | 244.2 | 242.1 | 240.4 | +0.8% level | -0.5% level | 4.83 / 4.83 / 4.83 |
| conc_t0 | code | 0.0 |  | 6 | 307.6 | 306.7 | 313.0 | 311.5 | 308.3 | 304.3 | +1.7% faster | -0.3% level | 4.49 / 4.49 / 4.49 |
| conc_t0 | code | 0.0 |  | 8 | 371.8 | 371.6 | 378.1 | 377.4 | 373.4 | 369.4 | +1.6% faster | -0.1% level | 4.49 / 4.49 / 4.49 |
| conc_t1_s1234 | chat | 1.0 | 1234 | 1 | 44.4 | 43.8 | 44.8 | 44.7 | 44.6 | 44.6 | +1.5% faster | +1.1% faster | 2.39 / 2.39 / 2.39 |
| conc_t1_s1234 | chat | 1.0 | 1234 | 2 | 69.9 | 68.9 | 70.1 | 70.0 | 70.0 | 70.0 | +0.9% level | +0.9% level | 2.45 / 2.45 / 2.45 |
| conc_t1_s1234 | chat | 1.0 | 1234 | 4 | 104.3 | 103.4 | 105.2 | 104.8 | 104.8 | 104.8 | +1.1% faster | +0.9% level | 2.46 / 2.46 / 2.46 |
| conc_t1_s1234 | chat | 1.0 | 1234 | 6 | 127.9 | 126.2 | 128.6 | 128.3 | 127.7 | 127.9 | +1.1% faster | +0.6% level | 2.43 / 2.43 / 2.43 |
| conc_t1_s1234 | chat | 1.0 | 1234 | 8 | 148.3 | 146.6 | 149.6 | 149.1 | 148.4 | 148.3 | +1.3% faster | +0.6% level | 2.43 / 2.43 / 2.43 |
| conc_t1_s1234 | code | 1.0 | 1234 | 1 | 69.9 | 69.3 | 70.3 | 70.3 | 70.2 | 69.5 | +1.0% faster | +0.4% level | 4.57 / 4.57 / 4.57 |
| conc_t1_s1234 | code | 1.0 | 1234 | 2 | 88.2 | 87.4 | 88.9 | 88.6 | 88.7 | 87.8 | +1.1% faster | +0.5% level | 3.77 / 3.77 / 3.77 |
| conc_t1_s1234 | code | 1.0 | 1234 | 4 | 129.0 | 127.8 | 130.0 | 129.6 | 129.5 | 127.7 | +1.1% faster | +0.2% level | 3.83 / 3.83 / 3.83 |
| conc_t1_s1234 | code | 1.0 | 1234 | 6 | 159.7 | 157.9 | 160.9 | 160.8 | 159.7 | 157.7 | +1.3% faster | -0.1% level | 3.72 / 3.72 / 3.72 |
| conc_t1_s1234 | code | 1.0 | 1234 | 8 | 184.8 | 182.8 | 187.4 | 187.0 | 185.3 | 182.8 | +1.8% faster | +0.1% level | 3.72 / 3.72 / 3.72 |

Verdicts in the first pair: TF_FLASHNEXT_FUSED_SPLIT=1: 14 faster, 8 level, 0 slower; TF_FLASHNEXT_FUSED_SPLIT unset: 1 faster, 21 level, 0 slower
