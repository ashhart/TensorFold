# Contributing to TensorFold

TensorFold's main line is the Zig engine and server.
Python 0.6.6 is maintained on `python-0.6`; Python clients and references under `tools` are development tools.
Start from the release branch named in the issue, and keep one change per pull request.
Open an issue before a new model family, backend or arithmetic change so we can agree on its gate.

## Build and host tests

Use the exact Zig version in `.zig-version`, currently 0.17.0.
Metal builds need Xcode's macOS SDK and Metal toolchain.

```sh
zig build native -Dcpu=apple_m1 -Dversion=1.0.0
zig build test test-golden -Dcpu=apple_m1
```

`native` writes `zig-out/native/bin/tensorfold-native`.
The `apple_m1` CPU target keeps the macOS binary compatible with M1 through M5.
`test` runs host checks; `test-golden` compares the frozen server corpus and records known differences separately.
Neither host result substitutes for an actual model/device gate.

Linux release builds use the CUDA backend and qualified GPU kernel assets.
[Packaging](packaging/README.md) lists the flat fatbin set and optional captured CUDA files required by `dist`.
Compile success alone does not qualify a GPU or checkpoint.

## Exactness and precision

A draft is accepted only when it equals the same engine's plain token.
Keep shared lane verification, caches and per-stream state in the core; add model operators and weight handling in the family.
A new family does not add another scheduler beside the engine.

Verify these contracts at the same checkpoint, backend, runtime and settings:

- Drafted output equals `"draft": false` output, token for token.
- A supported resumed path equals a fresh execution of the full prompt.
- Each concurrently requested reply equals its solo run, including engines that queue requests.
- Prompt chunks and decode use the required row arithmetic, with recurrent state, KV and final logits checked at seams.
- Cancellation, reset and partial acceptance leave the next request's state valid.

Keep the checkpoint's activation precision and the required accumulation precision.
A changed reduction order needs an independent FP64 reference for each affected projection class, with native error no worse than the current reference.
Compare prompt and decode references separately on identical input rows.
Report worst per-column differences as well as aggregate error; an average must not hide a worse column.
For model fidelity, use teacher-forced histories and record all top-token differences and margins.
A documented one-step tie policy applies only to the same two contenders with both margins within that policy.

## Performance receipts

Measure before and after on the same machine, using the same checkpoint, public prompt IDs and settings.
Name the commit/version, GPU, OS, checkpoint revision, runtime libraries and request parameters.
Record pass, fail and skip counts and explain any unrun platform or feature.

For served decode, the existing standard-library client needs Python 3.11 or newer:

```sh
python3 tools/bench_openai.py http://127.0.0.1:8080 local-model \
  --tokens 256 --reps 3 --temperatures 0 --label candidate --output decode.json
python3 tools/bench_concurrent.py http://127.0.0.1:8080 local-model \
  --tokens 256 --reps 3 --temperatures 0 --levels 1,4 --alone --serial --output concurrent.json
```

The server in these commands is the native binary.
The clients measure requests; Python is not part of native inference or the native build.
Include ordinary greedy prose and code in the measured prompts.
For cold prefill, attach exact 2k, 8k, 32k and 64k token arrays, request bodies, timings and cached-token counts.
Disable supported prefix retention with `--prompt-cache-gib 0` and use fresh KV state for each measurement.
Record 64k, 128k and the native window separately where the model/platform admits them.
Compare against the previous release and the applicable standard server, such as `mlx_lm` or `vLLM`.
When a difference is only a few percent, alternate before/after runs to separate it from temperature and load changes.

Memory receipts include peak physical process footprint and device/pinned allocations where the backend reports them.
State the hardware RAM class and context; a budget on a larger host is not a measurement on a smaller machine.
Use a single model process per test machine, the existing GPU lock and admission guard, and record load/cleanup boundaries.
Never raise a shared machine's limits to get a test through without its owner's authorization.

## Lean code and review

Each module has one job, source files stay at or below 600 lines, and comments/docstrings are single lines of at most 120 columns.
Measurements and history belong in a receipt or recipe, not source comments.
The lean checker must add zero problems over the branch's pinned baseline:

```sh
python3 tools/zig/lean_check.py
```

Attach the baseline count and identify inherited findings instead of claiming a nonzero checker exit is clean.
Write tests that demonstrate a real failure and its correction; keep the failing receipt when changing arithmetic or lifecycle behavior.
Run the host suite and the device/model checks for every affected platform.
We review a candidate, preserve its authorship and run the release gates before landing it.

## Public identity and privacy

Use your approved public Git identity and GitHub noreply address.
Keep every contributor's authorship through a port or merge; tool co-author trailers do not belong in the history.
Keep private emails, machine names, local user paths, addresses, keys and confidential data out of source, fixtures and receipts intended for publication.
Examples use fictional identities and reserved documentation addresses.
Inspect the complete publication tree, outgoing history and packaged artifacts before uploading them.
A clean diff or a passing test suite does not replace that review.

## Bug reports and license

Include the native version, model revision, GPU/OS, exact command, startup error and a public request that reproduces the problem.
A token mismatch report should include both drafted and plain token hashes at the same settings.

TensorFold is Apache-2.0; a contribution follows that license.
Model weights keep their own licenses, and the archive retains the project's required third-party notices.
