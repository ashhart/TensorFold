# Cold-start series: 36 cells, the first request of each fresh server

## short prompt (34 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 6 | 127.20, 127.02, 126.99, 127.11, 126.88, 126.68 | 127.01 | 126.98 | 15.00 | 0.09 | 1 |
| fixed 15 | 6 | 127.09, 127.21, 127.18, 127.07, 127.15, 127.04 | 127.12 | 127.12 | 15.00 | 0.09 | 1 |
| fixed 7 | 6 | 120.08, 119.89, 120.00, 119.97, 120.18, 120.04 | 120.02 | 120.03 | 7.00 | 0.09 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 15, 127.12): 0.999 on the medians (BELOW); the slowest adaptive server is 0.997 of it.
Adaptive / fixed 15 (head against main's default): 0.999.

## long prompt (7588 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 6 | 67.24, 67.22, 67.31, 67.27, 67.30, 67.32 | 67.29 | 67.28 | 7.00 | 4.01 | 1 |
| fixed 15 | 6 | 59.19, 59.16, 59.14, 59.18, 59.20, 59.17 | 59.17 | 59.17 | 15.00 | 4.01 | 1 |
| fixed 7 | 6 | 67.27, 67.29, 67.18, 67.24, 67.34, 67.35 | 67.28 | 67.28 | 7.00 | 4.01 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 7, 67.28): 1.000 on the medians (at or above); the slowest adaptive server is 0.999 of it.
Adaptive / fixed 15 (head against main's default): 1.137.
