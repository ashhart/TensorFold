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
  and the pinned `corpus.json`.
- For a startup job, write the receipt before the client runs. Take
  `launch_before_ns` on `time.perf_counter` before exec, and
  `ready_probe_end_ns` at the end of the first successful readiness probe.
  The client must use that same clock. It sends the first burst before any
  tokenize or primer call.
- Keep the smoke and safety gate ahead of the panel. Run the two startup
  orders only after those checks pass. `startup-forward` is baseline-first;
  `startup-reverse` is candidate-first.

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
