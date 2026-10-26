# Contributing to TensorFold

TensorFold's main line is the Zig engine and server.
Python 0.6.6 stays available at the `v0.6.6` tag and takes no further changes; Python clients and references under `tools` are development tools.
Start from `main`, the only branch, and keep one change per pull request.
Open an issue before a new model family, backend or arithmetic change so we can agree on its gate.

## Coordinate before editing

TensorFold uses [ATC](https://github.com/ashhart/ATC) to coordinate contributors and coding agents; check the [agent radar](https://tensorfold.dev/atc) before choosing an issue.
Follow the [README workflow](README.md#contributing-with-agents) to check, claim, publish your session, pulse while working and land when you finish.
With ATC 0.7.0 or later, a successful new claim automatically starts an anonymous public session; keep its returned session number and avoid starting a second one.
Showing a GitHub name is an explicit opt-in with `--github-name`; `--local` keeps a new session on your machine.
If an issue is taken, offer to pair, split or review instead of opening a competing implementation.
Run `atc overlap --json` before opening a pull request and resolve conflicts with overlapping work.
Agents must also read the [ATC contributor instructions](docs/atc/SKILL.md).

## Build and host tests

Use the exact Zig version in `.zig-version`, currently 0.17.0.
Metal builds need Xcode's macOS SDK and Metal toolchain.

```sh
zig build native -Dcpu=apple_m1 -Dversion=1.0.0
zig build test test-golden -Dcpu=apple_m1
```

`native` writes `zig-out/native/bin/tensorfold-native`.
The `apple_m1` CPU target keeps the macOS binary compatible with M1 through M5.
`test` runs unit tests; on macOS it also runs available Metal device checks, including GLM MLA and KDA checks.
Run the full suite on an isolated machine when the local GPU is serving a model.
`test-golden` compares the frozen server corpus and records known differences separately.
These results do not substitute for an actual checkpoint/model gate.

For the CUDA host-test subset without the macOS Metal family tests:

```sh
zig build test-cuda-host
```

This subset does not replace the complete `test` suite.

HIP is an explicit build opt-in. Add `-Dhip=true` to enable its host tests and named HIP targets:

```sh
zig build test -Dhip=true
zig build native-hip -Dhip=true -Dhipcc=/opt/rocm/bin/hipcc -Dhip-arch=gfx1100
```

The default graph omits HIP. Enabling it does not select another backend or change CUDA, Metal or XPU runtime options.
The HIP backend uses the same shared core and serving contracts. Device qualification remains specific to its
checkpoint, compiler, runtime and architecture; a host build does not establish those receipts.

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

A new or changed GPU kernel registers with the [kernel contract](docs/kernel-contract.md), and its pull request posts each backend's `SUMMARY` line.
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

### Speed claims: an optional Grill A/B/A2 receipt

Maintainers' own A/B measurements on Macs and GB10s decide whether a speed change lands.
[The Grill](https://github.com/plotarmordev/thegrill) is one optional way to produce an A/B/A2 receipt, alongside the `tools/bench_*` instructions above; contributors need not install it.
The receipt measures `main` at your branch point (A), your pull request (B), then the same `main` again (A2), on one machine with one checkpoint and the same serve flags.
To use this route, install The Grill 0.11.0 with its [install guide](https://github.com/plotarmordev/thegrill/blob/a551fdf2b63059f56df6287e72ed573ca06cca2f/docs/performance/INSTALL.md), including the checksum and release receipt checks; it sets `GRILL_PERF`.
Then run this from your pull request's checkout, with the model pulled and `origin` pointing at this repository.
On Linux, leave out `-Dcpu=apple_m1`.

```sh
WORK=$(mktemp -d)
MODEL=mlx-community/Qwen3.5-2B-MLX-4bit QUANT="MLX 4-bit"
FLAGS=(--context 8192 --no-thinking --prompt-cache-gib 0)
PR=$PWD A=$(git merge-base origin/main HEAD) B=$(git rev-parse HEAD)
git worktree add --detach "$WORK/main" "$A"
(cd "$WORK/main" && zig build native -Dcpu=apple_m1 -Dversion=1.0.0)
zig build native -Dcpu=apple_m1 -Dversion=1.0.0
up() { "$1/zig-out/native/bin/tensorfold-native" serve "$MODEL" --name grill --port 18080 "${FLAGS[@]}" > "$WORK/server-$2.log" 2>&1 & SERVER=$!
  until curl -fs http://127.0.0.1:18080/v1/models > /dev/null; do kill -0 $SERVER || return 1; sleep 1; done
  curl -fs http://127.0.0.1:18080/v1/benchmark/manifest > "manifest-$2.json"; }
down() { kill -TERM $SERVER; wait $SERVER; }
mkdir -p "$WORK/receipt/attempt-1" && cd "$WORK/receipt/attempt-1"
up "$WORK/main" A
jq -n --arg m "$MODEL" --arg q "$QUANT" --arg r "TensorFold $A" --arg s "serve ${FLAGS[*]}" 'input as $f | $f.hardware as $h | {
  model_revision: "\($m)@\($f.model.revision // "unknown") (\($q))", runtime: $r, settings: $s,
  hardware: "\($h.gpus[0].name), \($h.gpus[0].cores // "unknown") GPU cores, \($h.system_memory_bytes // "unknown") bytes, \($h.cpu_model // "unknown")"}' manifest-A.json > a.json
jq --arg r "TensorFold $B" '.runtime = $r' a.json > b.json
"$GRILL_PERF" baseline --workload portable-v1 --client-placement same-host --endpoint http://127.0.0.1:18080/v1/chat/completions --local-http --model grill --deployment a.json --out A
down; sleep 60; up "$PR" B
"$GRILL_PERF" check A --change runtime --deployment b.json --out B
down; sleep 60; up "$WORK/main" A2
"$GRILL_PERF" check A --change none --deployment a.json --out A2
down
"$GRILL_PERF" compare A B --reference A2 | tee compare.txt
"$GRILL_PERF" outputs A B | tee outputs.txt
tar -czf ../attempt-1.tgz A B A2 a.json b.json manifest-*.json compare.txt outputs.txt
```

For a later attempt, reuse the same `$WORK` and move its worktree with `git -C "$WORK/main" checkout --detach "$A"` instead of a second `git worktree add`; remove it at the end with `git worktree remove "$WORK/main"`.
`FLAGS` is the README's serve line with prefix retention off; add only the flags your claim is about.
For a new opt-in flag, serve all three runs from your build, so `a.json` and `b.json` both carry your pull request's commit as `runtime` and differ only in `settings`; add the flag to B's server and to `settings` in `b.json`, and use `--change settings`.
Before measuring, fill in any field that came out `unknown` or `null` in `a.json` and `b.json`.
A concurrency claim uses `--workload portable-c2-v1`, `portable-c4-v1` or `portable-c8-v1` and gives all three captures the same `--seconds` from the install guide's table.
Keep both `sleep 60` lines: `check` accepts a shorter gap, but `compare` then refuses the three captures.
Attach the archive and paste `compare.txt`; the server logs stay out of the archive.
`MEASURED FASTER` means B beat both A and A2; quote the smaller of the two changes.
A script reads `"result": "IMPROVED"` from `compare --json`, not the exit code.
`MEASURED SLOWER`, `INCONCLUSIVE` or an error means the receipt supports no speed claim.
Attach every attempt, failed ones included; after updating the commits and builds as needed, a later attempt starts again from the `mkdir` line with the next number, reusing the same `$WORK`.
On a checkpoint whose count prompt ends before 400 tokens, `portable-v1` cannot finish its captures; rerun all three with `--workload baseline-v2`, which fixes exactly 400 output tokens, and note the workload with the receipt.
`baseline-v2` is defined in the Grill repo's [workload JSON](https://github.com/plotarmordev/thegrill/blob/a551fdf2b63059f56df6287e72ed573ca06cca2f/crates/grill-perf/examples/baseline-v2.json) and described in its [default scope and budget](https://github.com/plotarmordev/thegrill/blob/a551fdf2b63059f56df6287e72ed573ca06cca2f/docs/performance/README.md#default-scope-and-budget), not in the install guide.
New code pushed after measuring needs a new receipt.
The Grill measures request throughput on its count prompt, prompt processing included, so its receipt sits beside the decode speed from `tools/bench_openai.py` and does not replace it.
Drafting stays on as shipped, and the count prompt favors it, so a drafting claim also keeps the prose and code numbers from `tools/bench_openai.py`.
The hardware and settings in the receipt are your word, and maintainers may re-run it.

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
