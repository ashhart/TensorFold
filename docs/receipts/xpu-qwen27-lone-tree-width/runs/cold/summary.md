# Cold-start series: 48 cells, the first request of each fresh server

## short prompt (34 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 128.98, 126.98, 126.71 | 126.98 | 127.56 | 15.00 | 0.09 | 1 |
| fixed 15 | 3 | 127.59, 128.82, 126.73 | 127.59 | 127.71 | 15.00 | 0.09 | 1 |
| fixed 7 | 3 | 120.21, 120.20, 119.87 | 120.20 | 120.09 | 7.00 | 0.09 | 1 |
| fixed 3 | 3 | 85.11, 85.16, 84.77 | 85.11 | 85.01 | 3.00 | 0.09 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 15, 127.59): 0.995 on the medians (BELOW); the slowest adaptive server is 0.993 of it.
Adaptive / fixed 15 (head against main's default): 0.995.

## long prompt (7588 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 65.61, 65.87, 65.53 | 65.61 | 65.67 | 7.75 | 4.01 | 1 |
| fixed 15 | 3 | 59.14, 59.14, 59.22 | 59.14 | 59.17 | 15.00 | 4.01 | 1 |
| fixed 7 | 3 | 67.28, 67.29, 67.32 | 67.29 | 67.30 | 7.00 | 4.01 | 1 |
| fixed 3 | 3 | 57.98, 58.02, 58.01 | 58.01 | 58.01 | 3.00 | 4.01 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 7, 67.29): 0.975 on the medians (BELOW); the slowest adaptive server is 0.974 of it.
Adaptive / fixed 15 (head against main's default): 1.109.

## p28946 prompt (28946 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 47.86, 47.81, 47.82 | 47.82 | 47.83 | 5.12 | 16.95 | 1 |
| fixed 15 | 3 | 35.39, 35.37, 35.36 | 35.37 | 35.37 | 15.00 | 16.98 | 1 |
| fixed 7 | 3 | 45.74, 45.73, 45.72 | 45.73 | 45.73 | 7.00 | 16.96 | 1 |
| fixed 3 | 3 | 50.37, 50.33, 50.33 | 50.33 | 50.34 | 3.00 | 16.96 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 50.33): 0.950 on the medians (BELOW); the slowest adaptive server is 0.950 of it.
Adaptive / fixed 15 (head against main's default): 1.352.

## p121022 prompt (121022 tokens)

| arm | servers | tok/s, servers 1..N | median | mean | nodes a round | TTFT s (median) | hashes |
|---|---:|---|---:|---:|---:|---:|---:|
| adaptive (head) | 3 | 23.60, 23.56, 23.55 | 23.56 | 23.57 | 3.53 | 101.26 | 1 |
| fixed 15 | 3 | 13.12, 13.12, 13.13 | 13.12 | 13.13 | 15.00 | 101.27 | 1 |
| fixed 7 | 3 | 20.92, 20.92, 20.92 | 20.92 | 20.92 | 7.00 | 101.29 | 1 |
| fixed 3 | 3 | 26.87, 26.89, 26.89 | 26.89 | 26.88 | 3.00 | 101.29 | 1 |

Token hashes across every cell of this prompt: 1 distinct (EQUAL).
Adaptive / best fixed tree (fixed 3, 26.89): 0.876 on the medians (BELOW); the slowest adaptive server is 0.876 of it.
Adaptive / fixed 15 (head against main's default): 1.795.

