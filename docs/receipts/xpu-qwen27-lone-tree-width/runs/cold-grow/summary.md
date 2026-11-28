# Cold-start series: 48 cells, the first request of each fresh server

## short prompt (34 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 127.13, 127.04, 126.84 | 127.04 | 127.00 | 15.00 | 0.09 | 1 |
| fixed 15 | 3 | 127.25, 126.78, 127.24 | 127.24 | 127.09 | 15.00 | 0.09 | 1 |
| fixed 7 | 3 | 119.85, 120.01, 119.85 | 119.85 | 119.91 | 7.00 | 0.09 | 1 |
| fixed 3 | 3 | 84.95, 84.93, 84.95 | 84.95 | 84.94 | 3.00 | 0.09 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 15, 127.24): 0.998 on the medians (BELOW); the slowest adaptive server is 0.997 of it.
Adaptive / fixed 15 (head against main's default): 0.998.

## long prompt (7588 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 67.31, 67.22, 67.28 | 67.28 | 67.27 | 7.00 | 4.00 | 1 |
| fixed 15 | 3 | 59.22, 59.16, 59.19 | 59.19 | 59.19 | 15.00 | 4.00 | 1 |
| fixed 7 | 3 | 67.34, 67.29, 67.30 | 67.30 | 67.31 | 7.00 | 4.00 | 1 |
| fixed 3 | 3 | 58.02, 58.03, 58.01 | 58.02 | 58.02 | 3.00 | 4.01 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 7, 67.30): 1.000 on the medians (BELOW); the slowest adaptive server is 0.999 of it.
Adaptive / fixed 15 (head against main's default): 1.137.

## p28946 prompt (28946 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 50.31, 50.36, 50.37 | 50.36 | 50.35 | 3.00 | 16.95 | 1 |
| fixed 15 | 3 | 35.44, 35.42, 35.43 | 35.43 | 35.43 | 15.00 | 16.95 | 1 |
| fixed 7 | 3 | 45.78, 45.78, 46.14 | 45.78 | 45.90 | 7.00 | 16.95 | 1 |
| fixed 3 | 3 | 50.26, 50.34, 50.38 | 50.34 | 50.33 | 3.00 | 16.95 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 50.34): 1.000 on the medians (at or above); the slowest adaptive server is 1.000 of it.
Adaptive / fixed 15 (head against main's default): 1.422.

## p121022 prompt (121022 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 26.89, 26.90, 26.90 | 26.90 | 26.89 | 3.00 | 101.07 | 1 |
| fixed 15 | 3 | 13.15, 13.14, 13.15 | 13.15 | 13.15 | 15.00 | 101.11 | 1 |
| fixed 7 | 3 | 20.94, 20.94, 20.94 | 20.94 | 20.94 | 7.00 | 101.10 | 1 |
| fixed 3 | 3 | 26.92, 26.91, 26.91 | 26.91 | 26.91 | 3.00 | 101.09 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 26.91): 0.999 on the medians (BELOW); the slowest adaptive server is 0.999 of it.
Adaptive / fixed 15 (head against main's default): 2.046.
