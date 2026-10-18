# PR836 native CUDA TTFT inputs

These collectors produced the rows in [../requests.csv](../requests.csv).
Point `run_panel.py`, `run_startup.py` and `run_safety.py` at a local
OpenAI-compatible server. Python 3.11 or newer. The published rows are the
15 October 2026 runs; this directory does not rerun them.

These are versioned copies of the earlier HTTP collectors. The Qwen3.5 CUDA
route has no cross-request prompt-state cache: both arms must report zero
`cached_tokens` on fresh, changed, exact-repeat and long-history requests.
The engine's per-stream GPU buffers are a different kind of reuse.

Before any timed run, use `prepare_fixture.py` against the chosen baseline's
`/tokenize` route with `literal-corpus-source.json` and
`literal-long-source.json`. Pin the resulting `corpus.json` and
`long-manifest.json` by SHA-256. The two servers must render exactly those
IDs; the collectors check this before panel timing. A fresh-process startup
collector sends its first request before any tokenization HTTP call and uses
the already pinned corpus. If the long messages no longer share 4096 rendered
tokens, fixture preparation fails rather than changing the benchmark silently.

`panel-abba-unbound.json`, `smoke-unbound.json`,
`startup-forward-unbound.json` and `startup-reverse-unbound.json` are the
four protocol templates. Each has `ready=false`, empty build identity and no
input hashes, so none of them is an executable plan. The host controller that
filled those placeholders is not in this directory: it is tied to one
machine's production service paths. Bind a replay by hand:

- Build both arms from the source archives in the receipt and keep the
  recorded native-server SHA256.
- Serve the pinned target and drafter, BF16 KV, 32,768-token context, 16
  slots, and zero restored prompt-cache tokens. Both arms use the same
  settings.
- Fill `{python}`, `{inputs}`, `{port}`, `{run}` and, for startup,
  `{startup}` in the template's `client_argv`. Pass `--repeat-cache zero`
  and the pinned `corpus.json`. All four templates expect the complete
  contents of this directory at `{inputs}/integration/pr836/harness/`,
  including the Python helpers and JSON fixtures. `{inputs}` is the absolute
  staging root, not the harness directory itself. Alternatively, set
  `{inputs}` to the checkout root and replace every
  `integration/pr836/harness/` prefix in `client_argv` with
  `docs/receipts/qwen-cuda-first-token/harness/`. Apply the same mapping to
  script and fixture paths in all four templates, then pin the staged inputs.
- For a startup job, write the complete receipt described below before the
  client runs. Capture every `*_ns` value as an integer from
  `time.perf_counter_ns()` on the server host. The client must run on that
  same host and use the same monotonic clock; its `time.perf_counter()`
  timestamps are in seconds. It sends the first burst before any tokenize
  or primer call.
- Keep the smoke and safety gate ahead of the panel. Run the two startup
  orders only after those checks pass. `startup-forward` is baseline-first;
  `startup-reverse` is candidate-first.

## Startup receipt contract

`{startup}` names a JSON file accepted by
[`read_startup()` in run_startup.py](run_startup.py#L31). It must contain all
of the following fields; the two anchors in `launches.csv` alone are not a
complete replay receipt.

| Field | Required value or meaning |
| --- | --- |
| `schema` | `"native_ttft_startup_v1"` |
| `complete` | JSON `true`, set after the successful readiness probe |
| `clock` | Exact string `"time.perf_counter_ns on the Spark host"`, as required by this collector |
| `launch_before_ns` | Integer clock reading immediately before launching the server process |
| `launch_after_ns` | Integer clock reading immediately after the process-launch call returns |
| `ready_probe_begin_ns` | Integer clock reading immediately before the first successful readiness probe |
| `ready_probe_end_ns` | Integer clock reading immediately after that successful probe |
| `server_pid` | Positive integer PID of the launched server process |
| `binary_sha256` | SHA-256 of that server binary, as 64 lowercase hexadecimal characters |
| `binary` | Object with `sha256` equal to `binary_sha256` |
| `file_cache_policy` | Exact string `"warm_or_uncontrolled_os_cache"`; these runs do not establish a cold filesystem cache |
| `readiness_probe_count` | Number of entries in `readiness_probes` |
| `readiness_probes` | Nonempty array of probe records in execution order, as specified below |

The four top-level clock readings must be positive and strictly ordered:
`launch_before_ns < launch_after_ns < ready_probe_begin_ns < ready_probe_end_ns`.
Do not use wall-clock/Unix timestamps or floating-point seconds for these
fields.

Each readiness probe records integer `begin_ns` and `end_ns` readings from
the same clock, with `begin_ns < end_ns`, and a Boolean `ready`. Every probe
before the last must have `ready=false`; the final probe must have
`ready=true`. The first probe begins at or after `launch_after_ns`, and each
later probe begins at or after the preceding probe ends. The final probe's
`begin_ns` and `end_ns` must equal `ready_probe_begin_ns` and
`ready_probe_end_ns`. Stop probing at the first success and retain all failed
probes that preceded it.

The collector additionally checks that the first generation requests begin
at or after readiness. Keep this complete receipt, the source/build/binary
pins and the client rows together for each process. The exports below retain
only the two anchors needed to recompute the published startup durations.

## Recomputing the published timings

[../launches.csv](../launches.csv) is the 24 startup receipts reduced to those
two anchors. Joined to the `first_burst` rows in
[../requests.csv](../requests.csv):

```
launch = launch_before_ns / 1e9
ready = ready_probe_end_ns / 1e9
launch_to_ready_s = ready - launch
launch_to_first_sse_s = min(first) - launch
launch_to_last_done_s = max(done) - launch
```

Those durations rebuild the 18 launch-inclusive rows in
[../metrics.csv](../metrics.csv). Each summary cell is the geometric mean of
the baseline-first and candidate-first observations.

The panel records first text SSE, decode, last DONE, output/token hashes,
cache depth, and speculative work. Its smoke adds changed code and a growing
chat turn to fresh/exact-repeat anchors. The full panel also exercises n=1/4/16
and the genuine long changing conversation. `run_safety.py` compares seeded
drafted/undrafted output, disconnects after first text, and checks a following
healthy request. These checks are not GPU state proofs; source/host ownership
tests and production restoration remain separate gates.
