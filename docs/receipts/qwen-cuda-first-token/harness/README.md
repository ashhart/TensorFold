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

The four `*-unbound.json` files are protocol templates, not executable plans.
They deliberately have `ready=false`, no model/build/binary or generated-input
hashes, and an invalid unbound schema. A reviewed PR836 controller must bind
the same target, drafter, tokenizer, server settings and inputs to both arms.
The older v2 controller contains prior model paths and cannot be assumed to
provide that binding. Keep the local smoke and safety gate ahead of the full
panel; run the startup forward/reverse plans only after correctness passes.

The panel records first text SSE, decode, last DONE, output/token hashes,
cache depth, and speculative work. Its smoke adds changed code and a growing
chat turn to fresh/exact-repeat anchors. The full panel also exercises n=1/4/16
and the genuine long changing conversation. `run_safety.py` compares seeded
drafted/undrafted output, disconnects after first text, and checks a following
healthy request. These checks are not GPU state proofs; source/host ownership
tests and production restoration remain separate gates.
