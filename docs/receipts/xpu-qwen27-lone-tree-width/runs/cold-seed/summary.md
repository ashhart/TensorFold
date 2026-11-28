# Cold-start series: 48 cells, the first request of each fresh server

## short prompt (34 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 127.01, 126.94, 126.70 | 126.94 | 126.89 | 15.00 | 0.09 | 1 |
| fixed 15 | 3 | 127.18, 127.18, 127.12 | 127.18 | 127.16 | 15.00 | 0.09 | 1 |
| fixed 7 | 3 | 120.06, 119.82, 120.14 | 120.06 | 120.01 | 7.00 | 0.09 | 1 |
| fixed 3 | 3 | 85.05, 84.94, 85.04 | 85.04 | 85.01 | 3.00 | 0.09 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 15, 127.18): 0.998 on the medians (BELOW); the slowest adaptive server is 0.996 of it.
Adaptive / fixed 15 (head against main's default): 0.998.

## long prompt (7588 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 67.24, 67.23, 67.16 | 67.23 | 67.21 | 7.00 | 4.01 | 1 |
| fixed 15 | 3 | 59.19, 59.17, 59.12 | 59.17 | 59.16 | 15.00 | 4.01 | 1 |
| fixed 7 | 3 | 67.32, 67.33, 67.33 | 67.33 | 67.32 | 7.00 | 4.01 | 1 |
| fixed 3 | 3 | 58.00, 58.06, 58.00 | 58.00 | 58.02 | 3.00 | 4.01 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 7, 67.33): 0.999 on the medians (BELOW); the slowest adaptive server is 0.998 of it.
Adaptive / fixed 15 (head against main's default): 1.136.

## p28946 prompt (28946 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 50.35, 50.32, 50.33 | 50.33 | 50.34 | 3.00 | 16.95 | 1 |
| fixed 15 | 3 | 35.38, 35.38, 35.37 | 35.38 | 35.38 | 15.00 | 16.95 | 1 |
| fixed 7 | 3 | 45.74, 45.72, 45.74 | 45.74 | 45.73 | 7.00 | 16.97 | 1 |
| fixed 3 | 3 | 50.35, 50.34, 50.37 | 50.35 | 50.35 | 3.00 | 16.97 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 50.35): 1.000 on the medians (BELOW); the slowest adaptive server is 0.999 of it.
Adaptive / fixed 15 (head against main's default): 1.423.

## p121022 prompt (121022 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 26.90, 26.86, 26.88 | 26.88 | 26.88 | 3.00 | 101.36 | 1 |
| fixed 15 | 3 | 13.13, 13.11, 13.12 | 13.12 | 13.12 | 15.00 | 101.36 | 1 |
| fixed 7 | 3 | 20.92, 20.91, 20.90 | 20.91 | 20.91 | 7.00 | 101.35 | 1 |
| fixed 3 | 3 | 26.88, 26.89, 26.88 | 26.88 | 26.88 | 3.00 | 101.34 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 26.88): 1.000 on the medians (BELOW); the slowest adaptive server is 0.999 of it.
Adaptive / fixed 15 (head against main's default): 2.049.
