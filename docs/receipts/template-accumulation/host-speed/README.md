# Template render speed

On an arm64 macOS host (Darwin 27.0.0), the synthetic interleaved workload rendered faster with the candidate than the base. This is a host microbenchmark, not a model or served benchmark.

| Workload | Base median | Candidate median | Change |
| --- | ---: | ---: | ---: |
| Short chat, 8 messages | 8.35 µs | 8.38 µs | +0.2% |
| Interleaved, 128 messages | 0.949 ms | 0.196 ms | −79.3% |
| Interleaved, 256 messages | 3.597 ms | 0.474 ms | −86.8% |
| Interleaved, 512 messages | Scratch cap exceeded | 0.839 ms | Not comparable |

The base was `f350cd3ea` and the candidate was `a8dd85606`, compiled with Zig 0.17.0 `ReleaseFast`. Each interleaved message has 1,024 request bytes and a literal separator. An independent 4,096-byte concatenation runs between namespace appends. The short control has eight 32-byte messages and direct output. Each measured render checks every output byte and provenance offset. Compilation, input construction and validation are outside the timed interval.

Five alternating ABBA/BAAB blocks produced ten runs per revision. Each run has two warmups and three timed renders per cell; the table reports the median of the ten per-run medians. [results.json](results.json) holds the normalized per-render timings and exit codes; [summary.json](summary.json) gives the aggregation. The [harness](speed_test.zig) caps tracked template scratch at 240 MiB. In all ten base `ReleaseFast` runs, 512 messages hit the cap before a valid render completed. The candidate's 512 timing therefore has no paired speed comparison.

The earlier Debug regression receipt measured 225,995,904 bytes of peak base scratch at 512 messages and completed under the cap. In `ReleaseFast`, the base requested a 102,437,542-byte arena allocation with 194,488,720 bytes already live. The accounting measures live allocation, not cumulative allocation. The reason for the mode-dependent allocation difference was not established.

To reproduce, make both commits available locally and use Zig 0.17.0 and Python 3:

```sh
python3 docs/receipts/template-accumulation/host-speed/run.py --repo . --zig /path/to/zig --output /path/to/new-output-directory
```

The runner creates temporary source snapshots, compiles both binaries, runs the alternating blocks and writes per-run logs and normalized timings into a new output directory. It fails on output or provenance mismatches, unexpected test failures or incomplete results; the documented base cap refusal is recorded without treating it as a timing.
