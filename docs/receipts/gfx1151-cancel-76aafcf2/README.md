# gfx1151 cancellation follow-up

Measured on 2026-10-18, following the unresolved 27B case in [#994](https://github.com/ashhart/TensorFold/pull/994).
This is a fresh A/B run on public source `76aafcf274e60a5e31a59ce145b2f71f7b366e9d` and the runtime patch
in `5429554e708fdf7a9e0017301a00eb87a4182acd`. It is not a full release qualification.

The single GPU is a BOSGAME M5's Radeon 8060S (`gfx1151`). The checkpoint is
`TensorFold/Qwen3.8-27B-MLX-4bit` at `70ae7fac63274ff2eac54152031433374cb80f2f`.
All 17 checkpoint files match their previously pinned public Git-blob/LFS digests; see `checkpoint.json`.

## Cancellation result

The same gate runs three trials per build. Each trial separately measures the time from closing a long request's
socket until the engine's `requests_running` metric reaches zero, then the next short request sent 50 ms behind
another cancellation. Every long request uses a fresh nonce first, preventing long-prefix reuse.
The client closes 0.4 seconds after sending it. All twelve cancelled prompts fit within the 16,384-token window
and end with `reason=client-left`, zero emitted tokens. Actual prompt lengths are in `audit.json`.

| Measurement | Unmodified base | Fixed |
| --- | ---: | ---: |
| Idle short baseline, best of three | 1.899 s | 1.913 s |
| Disconnect to idle metric, three trials | 4.720–4.744 s | 0.032–0.042 s |
| Next short request, three trials | 6.532–6.548 s | 1.902–1.907 s |
| Cancelled request lifetime, arrival to end | 5.08–5.14 s | 0.43–0.44 s |
| Gate: release <=0.5 s and short <=idle+0.5 s | **3/3 failed** | **3/3 passed** |

All fourteen short replies across the two builds have the same complete native token SHA-256:
`ad11ca37c7fcabaf13c8fc1142956890438c4ca49c58d6f579c3eef5485b77d5`.
`*-http.jsonl` retains each short request/response and each cancelled request's exact body.
`*-server.log` independently retains six client-left completions per build.
Arrival-to-end lifetime is not used as disconnect-to-release time.

The harness's optional `--baseline-27b` also prints the historical v1.0.12 receipt for comparison.
Those historical numbers are labelled separately; the table above is the fresh unmodified-versus-fixed comparison.

## Correctness and timing controls

An identical uncancelled 14,975-token prompt produces the same full native token SHA-256 on both builds:
`677caeac8d5e87f26c67e37ff1551667fc3217c4690d7a276204d22c2504f0a1`.
Prefill takes 79.021 s before and 78.867 s after. This is one timing control, not a broad performance claim.
The complete requests and replies are in `*-uncancelled.json`.

The unmodified `serve_check.py` then passes its 13 exercised comparisons on the fixed server, including
concurrent/solo and resumed/fresh checks. It prints 15 successes, but two drafting comparisons are unexercised:
this checkpoint has no draft head. The raw output retains all fifteen lines without inflating exercised coverage.

The host build passes 292/292 steps; cached results are reused. It reports 126 new passes and two skips, and the
HIP Qwen family reports 32 passes/one skip. Before the implementation, retaining the old 2,048-row threshold and
the null cancellation hook makes both new CPU regressions fail. Remaining work in that deliberately failing build
was stopped once those two failures were captured; it is not a completed full-suite run. See `cpu-*.log`.

## Reproduction and scope

`state.json` retains exact build/server/test commands and both binary SHA-256 values. Use Zig 0.17.0,
HIP 7.1 and clang 21; the HIP build targets `gfx1151` with explicit compiler/include/device-bitcode paths.
The server uses `--context 16384 --parallel 4 --prompt-cache-gib 1 --no-thinking --metrics-open` on loopback.
Run from each source version, using the same pinned checkpoint:

```sh
python3 zig/tests/qwen35/serve_cancel_release.py http://127.0.0.1:8092 amd-cancel-27b \
  --runs 3 --server-log SERVER_LOG --record HTTP_JSONL --baseline-27b
python3 zig/tests/qwen35/serve_check.py http://127.0.0.1:8092 amd-cancel-27b 64
```

Only single-rank HIP Qwen prompt passes of at least 512 rows now poll cancellation at layer boundaries.
Shorter passes still finish their current chunk; tensor-parallel groups retain between-chunk cancellation so
ranks do not diverge at a collective. This run does not qualify every prompt cut, other models, multi-GPU,
drafting, or all release gates. It does not implement DFlash2 or resolve the separate benchmark-publication work.

This directory is a separate sanitized publication copy. Machine-local paths/hostname are replaced with
placeholders and trailing whitespace normalized. Request/response values and native hashes are unchanged.
`publication.json` records source/published hashes; `SHA256SUMS` covers every published file except itself.
Original raw evidence and service-control state remain private and unchanged.
