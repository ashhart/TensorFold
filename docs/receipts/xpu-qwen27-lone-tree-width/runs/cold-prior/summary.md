# Cold-start series: 48 cells, the first request of each fresh server

## short prompt (34 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 127.06, 126.93, 126.69 | 126.93 | 126.89 | 15.00 | 0.09 | 1 |
| fixed 15 | 3 | 127.15, 127.21, 126.96 | 127.15 | 127.11 | 15.00 | 0.09 | 1 |
| fixed 7 | 3 | 119.90, 119.84, 119.95 | 119.90 | 119.89 | 7.00 | 0.09 | 1 |
| fixed 3 | 3 | 84.84, 84.92, 84.84 | 84.84 | 84.87 | 3.00 | 0.09 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 15, 127.15): 0.998 on the medians (BELOW); the slowest adaptive server is 0.996 of it.
Adaptive / fixed 15 (head against main's default): 0.998.

## long prompt (7588 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 67.11, 67.16, 67.16 | 67.16 | 67.14 | 7.00 | 4.01 | 1 |
| fixed 15 | 3 | 59.15, 59.16, 59.21 | 59.16 | 59.17 | 15.00 | 4.01 | 1 |
| fixed 7 | 3 | 67.31, 67.21, 67.28 | 67.28 | 67.26 | 7.00 | 4.01 | 1 |
| fixed 3 | 3 | 58.03, 57.98, 57.98 | 57.98 | 58.00 | 3.00 | 4.01 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 7, 67.28): 0.998 on the medians (BELOW); the slowest adaptive server is 0.998 of it.
Adaptive / fixed 15 (head against main's default): 1.135.

## p28946 prompt (28946 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 50.08, 50.09, 50.09 | 50.09 | 50.09 | 3.00 | 16.94 | 1 |
| fixed 15 | 3 | 35.42, 35.40, 35.38 | 35.40 | 35.40 | 15.00 | 16.96 | 1 |
| fixed 7 | 3 | 45.77, 45.75, 45.73 | 45.75 | 45.75 | 7.00 | 16.97 | 1 |
| fixed 3 | 3 | 50.32, 50.33, 50.31 | 50.32 | 50.32 | 3.00 | 16.96 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 50.32): 0.995 on the medians (BELOW); the slowest adaptive server is 0.995 of it.
Adaptive / fixed 15 (head against main's default): 1.415.

## p121022 prompt (121022 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 26.85, 26.86, 26.86 | 26.86 | 26.85 | 3.00 | 101.30 | 1 |
| fixed 15 | 3 | 13.13, 13.13, 13.13 | 13.13 | 13.13 | 15.00 | 101.24 | 1 |
| fixed 7 | 3 | 20.93, 20.92, 20.93 | 20.93 | 20.93 | 7.00 | 101.28 | 1 |
| fixed 3 | 3 | 26.89, 26.89, 26.89 | 26.89 | 26.89 | 3.00 | 101.25 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 26.89): 0.999 on the medians (BELOW); the slowest adaptive server is 0.998 of it.
Adaptive / fixed 15 (head against main's default): 2.045.

