# The noise rule, cell by cell

M1 and M2 are main, F1 and F2 the branch with `TF_FLASHNEXT_FUSED_SPLIT=1`, in the order M1 F1 M2 F2. Aggregate tok/s; the verdict compares the mean of the branch's two runs with the mean of main's two. Tokens a round are the tools' tokens a streamed chunk.

| invocation | prompt | temp | seed | streams | M1 | M2 | F1 | F2 | F vs main | tokens a round M / F |
|---|---|---|---|---|---|---|---|---|---|---|
| openai_t0 | fibonacci-raw | 0.0 |  | 1 | 72.3 | 71.7 | 72.8 | 71.8 | +0.4% level | 5.22 / 5.22 |
| openai_t0 | gpu-chat-no-think | 0.0 |  | 1 | 46.5 | 46.0 | 46.6 | 46.2 | +0.4% level | 2.81 / 2.81 |
| conc_t0 | chat | 0.0 |  | 1 | 46.4 | 46.0 | 46.8 | 46.1 | +0.5% level | 2.81 / 2.81 |
| conc_t0 | chat | 0.0 |  | 2 | 86.5 | 85.9 | 87.0 | 85.9 | +0.3% level | 2.51 / 2.51 |
| conc_t0 | chat | 0.0 |  | 4 | 155.8 | 154.6 | 156.6 | 154.8 | +0.3% level | 2.51 / 2.51 |
| conc_t0 | chat | 0.0 |  | 6 | 210.3 | 208.4 | 212.1 | 209.3 | +0.6% level | 2.51 / 2.51 |
| conc_t0 | chat | 0.0 |  | 8 | 257.5 | 254.5 | 259.7 | 256.6 | +0.8% level | 2.51 / 2.51 |
| conc_t0 | code | 0.0 |  | 1 | 72.3 | 71.6 | 73.0 | 71.8 | +0.6% level | 5.22 / 5.22 |
| conc_t0 | code | 0.0 |  | 2 | 137.7 | 136.3 | 138.3 | 136.8 | +0.4% level | 4.83 / 4.83 |
| conc_t0 | code | 0.0 |  | 4 | 242.3 | 239.6 | 245.5 | 242.2 | +1.2% level | 4.83 / 4.83 |
| conc_t0 | code | 0.0 |  | 6 | 307.0 | 303.5 | 312.5 | 308.2 | +1.7% faster | 4.49 / 4.49 |
| conc_t0 | code | 0.0 |  | 8 | 372.4 | 367.2 | 377.4 | 373.1 | +1.5% faster | 4.49 / 4.49 |
| conc_t1_s1234 | chat | 1.0 | 1234 | 1 | 44.3 | 44.0 | 44.8 | 44.1 | +0.7% level | 2.39 / 2.39 |
| conc_t1_s1234 | chat | 1.0 | 1234 | 2 | 69.7 | 69.1 | 70.1 | 69.2 | +0.4% level | 2.45 / 2.45 |
| conc_t1_s1234 | chat | 1.0 | 1234 | 4 | 104.4 | 103.1 | 105.0 | 103.5 | +0.5% level | 2.46 / 2.46 |
| conc_t1_s1234 | chat | 1.0 | 1234 | 6 | 127.4 | 125.9 | 128.4 | 126.6 | +0.7% level | 2.43 / 2.43 |
| conc_t1_s1234 | chat | 1.0 | 1234 | 8 | 148.1 | 146.0 | 149.2 | 146.8 | +0.6% level | 2.43 / 2.43 |
| conc_t1_s1234 | code | 1.0 | 1234 | 1 | 70.0 | 69.2 | 70.7 | 69.6 | +0.8% level | 4.57 / 4.57 |
| conc_t1_s1234 | code | 1.0 | 1234 | 2 | 88.1 | 87.4 | 89.1 | 87.7 | +0.7% level | 3.77 / 3.77 |
| conc_t1_s1234 | code | 1.0 | 1234 | 4 | 128.8 | 127.3 | 129.9 | 127.9 | +0.7% level | 3.83 / 3.83 |
| conc_t1_s1234 | code | 1.0 | 1234 | 6 | 159.3 | 157.3 | 161.0 | 158.2 | +0.8% level | 3.72 / 3.72 |
| conc_t1_s1234 | code | 1.0 | 1234 | 8 | 184.9 | 182.3 | 187.3 | 184.3 | +1.2% level | 3.72 / 3.72 |

Verdicts in the first pair: TF_FLASHNEXT_FUSED_SPLIT=1: 2 faster, 20 level, 0 slower
