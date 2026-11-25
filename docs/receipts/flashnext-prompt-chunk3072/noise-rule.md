# The noise rule, and the cells it judged

A cell is SLOWER when both after runs are under both before runs (for a latency: over) and the mean gap is over 1%; FASTER when both are over both and the gap is over 1%; LEVEL otherwise. Each arm booted twice per timed pass (`M1 B1 B2 M2`). A flagged cell gets one rerun of its invocation on fresh boots inside the same window, and a repeat of slower is a loss; this window flagged none.

- cold prompts: 9 faster, 0 level, 0 slower, 0 unmeasured
- decode: 0 faster, 12 level, 0 slower, 0 unmeasured
- one-stream control: 0 faster, 5 level, 0 slower, 0 unmeasured
- losses: none

## Cold prompts, four slots

| prompt | load | M1 | M2 | B1 | B2 | B vs M |
|---|---|---|---|---|---|---|
| cold2k-off | idle | 1.070 | 1.071 | 0.972 | 0.960 | -9.8% faster |
| cold2k-off | load3 | 1.273 | 1.262 | 1.053 | 1.046 | -17.2% faster |
| cold8k-off | idle | 4.035 | 3.986 | 3.664 | 3.685 | -8.4% faster |
| cold8k-off | load3 | 4.653 | 4.632 | 3.989 | 3.987 | -14.1% faster |
| cold32k-off | idle | 16.015 | 16.044 | 14.980 | 15.039 | -6.4% faster |
| cold32k-off | load3 | 18.249 | 18.420 | 16.114 | 16.094 | -12.2% faster |
| cold64k-off | idle | 33.058 | 33.175 | 30.665 | 30.682 | -7.4% faster |
| cold64k-off | load3 | 37.582 | 37.718 | 32.780 | 32.786 | -12.9% faster |
| first_ttft_s | idle | 1.241 | 1.221 | 1.142 | 1.143 | -7.2% faster |

## The one-stream control

| prompt | load | M1 | M2 | B1 | B2 | B vs M |
|---|---|---|---|---|---|---|
| cold2k-off | idle | 0.979 | 0.968 | 0.969 | 0.966 | -0.6% level |
| cold8k-off | idle | 3.458 | 3.434 | 3.453 | 3.414 | -0.4% level |
| cold32k-off | idle | 14.144 | 14.150 | 14.124 | 14.100 | -0.2% level |
| cold64k-off | idle | 29.025 | 28.967 | 28.896 | 28.984 | -0.2% level |
| first_ttft_s | idle | 1.148 | 1.150 | 1.143 | 1.141 | -0.6% level |

## Decode

| prompt | streams | M1 | M2 | B1 | B2 | B vs M | tokens a round, M / B |
|---|---|---|---|---|---|---|---|
| chat | 1 | 46.9 | 46.2 | 46.5 | 46.3 | -0.3% level | 2.81 / 2.81 |
| chat | 2 | 86.8 | 86.2 | 86.3 | 86.3 | -0.2% level | 2.51 / 2.51 |
| chat | 4 | 156.6 | 155.1 | 155.7 | 155.7 | -0.1% level | 2.51 / 2.51 |
| chat | 6 | 211.0 | 209.2 | 209.8 | 209.5 | -0.2% level | 2.51 / 2.51 |
| chat | 8 | 258.1 | 256.1 | 256.9 | 256.5 | -0.2% level | 2.51 / 2.51 |
| code | 1 | 73.1 | 71.9 | 72.3 | 72.5 | -0.1% level | 5.22 / 5.22 |
| code | 2 | 138.8 | 136.9 | 138.0 | 137.5 | -0.1% level | 4.83 / 4.83 |
| code | 4 | 244.9 | 240.8 | 242.7 | 241.6 | -0.3% level | 4.83 / 4.83 |
| code | 6 | 310.5 | 306.5 | 307.7 | 306.2 | -0.5% level | 4.49 / 4.49 |
| code | 8 | 374.5 | 370.5 | 370.8 | 371.7 | -0.3% level | 4.49 / 4.49 |
| fibonacci-raw | 1 | 73.4 | 72.2 | 72.7 | 72.6 | -0.2% level | 5.22 / 5.22 |
| gpu-chat-no-think | 1 | 47.0 | 46.3 | 46.5 | 46.5 | -0.3% level | 2.81 / 2.81 |
