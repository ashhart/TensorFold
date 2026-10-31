## Cold prefill (max_tokens 1, prompt cache off, fresh server per run)
| arm | prompt tokens | context | prefill tok/s (median) | per round | TTFT (s) | wall (s) | cached tokens | GTT+VRAM peak over idle (GiB) | RSS peak (GiB) | server |
|---|---|---|---|---|---|---|---|---|---|---|
| new | 2,048 | 10240 | 673.7 | 681.4, 666.0 | 3.040 | 3.041 | 0 | 82.45 | 102.9 | ready |
| base | 2,048 | 10240 | 293.2 | 290.0, 296.4 | 6.986 | 6.996 | 0 | 78.21 | 76.81 | ready |
| new | 8,192 | 16384 | 558.2 | 559.2, 557.2 | 14.68 | 14.68 | 0 | 82.62 | 105.0 | ready |
| base | 8,192 | 16384 | 261.3 | 261.3, 261.3 | 31.35 | 31.35 | 0 | 78.39 | 77.56 | ready |
| new | 32,768 | 40960 | 513.5 | 512.6, 514.5 | 63.81 | 63.81 | 0 | 83.33 | 105.0 | ready |
| base | 32,768 | 40960 | 260.4 | 260.1, 260.6 | 125.9 | 125.9 | 0 | 79.09 | 79.81 | ready |
| new | 65,536 | 73728 | 496.7 | 497.2, 496.2 | 132.0 | 132.0 | 0 | 84.27 | 104.3 | ready |
| base | 65,536 | 73728 | 259.3 | 259.3, 259.3 | 252.7 | 252.7 | 0 | 80.03 | 81.64 | ready |
| new | 131,072 | 139264 | 471.6 | 471.7, 471.5 | 277.9 | 277.9 | 0 | 86.14 | 103.8 | ready |
| base | 131,072 | 139264 | 258.0 | 258.1, 257.8 | 508.1 | 508.1 | 0 | 81.91 | 83.66 | ready |
| new | 261,888 | 262144 | 437.3 | 437.5, 437.1 | 598.8 | 598.8 | 0 | 89.68 | 103.9 | ready |
| base | 261,888 | 262144 | 244.8 | 245.1, 244.5 | 1,070 | 1,070 | 0 | 85.45 | 86.69 | ready |

new / base, prefill tok/s
| prompt tokens | new | base | ratio |
|---|---|---|---|
| 2,048 | 673.7 | 293.2 | 2.298 |
| 8,192 | 558.2 | 261.3 | 2.136 |
| 32,768 | 513.5 | 260.4 | 1.972 |
| 65,536 | 496.7 | 259.3 | 1.915 |
| 131,072 | 471.6 | 258.0 | 1.828 |
| 261,888 | 437.3 | 244.8 | 1.786 |

## Decode, one stream (bench_openai.py, 256 tokens, greedy, drafts on)
| arm | prompt | decode tok/s (median) | per round | TTFT (s) | wall (s, derived) | spec-decode hit rate | GTT+VRAM peak (GiB) | RSS peak (GiB) |
|---|---|---|---|---|---|---|---|---|
| new | fibonacci-raw | 69.04 | 69.15, 68.92 | 0.09908 | 3.793 | 76.2 % | 84.38 | 104.7 |
| new | gpu-chat-no-think | 55.92 | 55.44, 56.39 | 0.1684 | 4.729 | 76.2 % | 84.38 | 104.7 |
| base | fibonacci-raw | 50.80 | 50.94, 50.66 | 0.1237 | 5.143 | 70.9 % | 81.33 | 76.80 |
| base | gpu-chat-no-think | 45.47 | 46.23, 44.70 | 0.2396 | 5.848 | 70.9 % | 81.33 | 76.80 |

## Prefill, serial decode and Default MTP decode (one public-code prompt, greedy, cache off, drafts as stated)
| arm | decode | output tokens | decode tok/s (median) | every rep | TTFT (s) | wall (s) | spec-decode hit rate | GTT+VRAM peak over idle (GiB) | RSS peak (GiB) | cached tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| base | Default MTP | 256 | 40.27 | 34.91, 40.64, 41.07, 34.86, 41.02, 39.91 | 46.48 | 52.84 | 68.1 % | 78.48 | 78.67 | 0 |
| base | Default MTP | 1024 | 38.26 | 35.44, 38.42, 38.33, 34.39, 38.77, 38.19 | 46.54 | 73.32 | 68.1 % | 78.48 | 78.67 | 0 |
| new | Default MTP | 256 | 51.86 | 52.11, 51.89, 52.71, 51.83, 51.75, 50.15 | 30.91 | 35.86 | 61.8 % | 82.72 | 103.8 | 0 |
| new | Default MTP | 1024 | 49.05 | 49.44, 49.46, 49.22, 48.11, 48.03, 48.88 | 30.94 | 51.86 | 61.8 % | 82.72 | 103.8 | 0 |
| base | serial (--no-drafts) | 256 | 29.64 | 27.14, 30.41, 29.92, 26.43, 29.41, 29.86 | 46.38 | 55.19 | - | 78.48 | 78.63 | 0 |
| base | serial (--no-drafts) | 1024 | 29.99 | 27.83, 30.16, 30.29, 26.94, 29.99, 29.99 | 46.55 | 80.70 | - | 78.48 | 78.63 | 0 |
| new | serial (--no-drafts) | 256 | 35.55 | 35.72, 35.58, 35.53, 35.48, 35.45, 35.86 | 30.71 | 37.86 | - | 82.68 | 104.9 | 0 |
| new | serial (--no-drafts) | 1024 | 35.71 | 35.75, 35.95, 35.66, 35.48, 35.77, 35.67 | 30.75 | 59.39 | - | 82.68 | 104.9 | 0 |

| arm | prefill tok/s (cold, 16K) | serial decode tok/s (256) | serial decode tok/s (1024) | Default MTP decode tok/s (256) | Default MTP decode tok/s (1024) | MTP token hash == serial's |
|---|---|---|---|---|---|---|
| new | 533.5 | 35.55 | 35.71 | 51.86 | 49.05 | yes/yes |
| base | 353.3 | 29.64 | 29.99 | 40.27 | 38.26 | yes/yes |

## Concurrent (bench_concurrent.py --levels 1,4 --alone --serial)
| arm | prompt | streams | aggregate tok/s | per-stream tok/s | concurrent==alone (eq/ne/failed) | alone==draft:false (eq/ne/failed) | failed requests |
|---|---|---|---|---|---|---|---|
| new | chat | 1 | 57.20 | 57.20 | 6/0/0 | 2/0/0 | 0 |
| new | chat | 4 | 99.75 | 25.85 | 24/0/0 | 2/0/0 | 0 |
| new | code | 1 | 68.45 | 68.45 | 6/0/0 | 2/0/0 | 0 |
| new | code | 4 | 117.4 | 30.00 | 24/0/0 | 2/0/0 | 0 |
| base | chat | 1 | 45.30 | 45.30 | 6/0/0 | 2/0/0 | 0 |
| base | chat | 4 | 72.95 | 18.65 | 24/0/0 | 2/0/0 | 0 |
| base | code | 1 | 51.15 | 51.15 | 6/0/0 | 2/0/0 | 0 |
| base | code | 4 | 89.65 | 22.70 | 24/0/0 | 2/0/0 | 0 |
