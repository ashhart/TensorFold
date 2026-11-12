# A startup receipt on Metal

For Qwen3.8-27B MLX affine 4-bit g64, `TF_STARTUP_RECEIPT=1` adds two JSON lines on stderr,
each prefixed `startup-receipt `. Unset or `0` leaves the usual output unchanged.
Other deployments have not been qualified with this harness.

The first record follows model worker start, socket listen and successful accept-thread creation.
A failed or interrupted load produces no ready record. The second follows the first successful
non-streaming OpenAI completion's whole HTTP response write. Streaming completions do not emit it.
The receipt carries the existing full reply token SHA-256, never the prompt or reply text.

`startup.schema` is `1`; `event` is `ready` or `first_completion`. Times are nanoseconds on the
awake monotonic clock, relative to native `main` entry. Runtime initialization before `main` is
outside this clock. The harness separately measures parent launch to observed readiness.
Each named phase carries `start_ns`, `end_ns`, `total_ns` and `calls`. Null means no recorded call.
Parent phases include child phases; tokenizer and model loading overlap. Do not add their times.
For repeated Metal source and pipeline builds, first start and last end include gaps; `total_ns`
sums only the calls. These totals also include work deferred until the first request.

`resources` uses the existing Darwin process counters. `physical_footprint_bytes` is a sample;
`lifetime_peak_physical_footprint_bytes` includes peaks between samples. `disk_io_bytes_read` is
process-lifetime physical disk I/O, including non-checkpoint files. A warm page-cache read can
contribute zero physical bytes. It is not a logical checkpoint byte count. Unsupported reads
produce null resources; the harness refuses to call that a complete device receipt.

Build the candidate native server with the Metal toolchain before running the harness. It does
not build, install, download, purge the OS cache or generate prepared weights. From the source tree:

```sh
python3 tools/startup_receipt.py \
  --binary zig-out/bin/tensorfold-native --model "$MODEL" \
  --output "$OUTPUT" --case warm --runs 3
```

Choose an existing local checkpoint for `MODEL` and a new output directory for `OUTPUT`.
The harness uses loopback, an ephemeral port, one stream, context 4096, no drafts and no prompt
cache. ANE remains explicitly off, which is already this family's default. It hashes the binary
and checkpoint files, sequentially reads the checkpoint before each fresh process, sends the
fixed greedy 64-token request in `startup_receipt_format.py`, then reaps only its own child.
Sequential prewarming is best effort: the kernel may evict pages before they are read again.
Inherited registered environment overrides still apply; record them alongside device results.

Every attempt writes `receipt.json`, including failures and interruption. `summary.json` preserves
all attempts, identities and the common reply hash. Inspect `server-private.log` locally; it can
contain checkpoint paths from normal startup messages and is not a public report.

The `--case cold` option writes an unsupported receipt and launches nothing. This harness has no
unprivileged per-checkpoint OS-cache eviction on macOS. This loader rebuilds target layouts for
each process; there is no engine-managed persistent target pack hit/miss to select. Metal's driver
shader cache is opaque here, so its hit/miss is uncontrolled, not a measured prepared-artifact case.

Host checks, with no GPU or model:

```sh
zig build test-startup-receipt-host test-server-cpu -j1 --summary all
python3 -m unittest discover -s tools/tests -p test_startup_receipt.py -v
```
