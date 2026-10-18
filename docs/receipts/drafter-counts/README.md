# Draft counts beside the receipt: gfx1151 runs and host controls

These are the receipts behind [#1045](https://github.com/ashhart/TensorFold/pull/1045): each benchmark attempt's
verify rounds, drafted tokens and accepted tokens, as the server reported them, kept in `RUN_ID.drafts.json` beside
the board's receipt. They show that the counts in the file are the server's own: for the two attached runs, every
sample's counts are the same numbers the server printed for that request in its log.

## Where and when

The GPU runs were measured on 18 October 2026 on a Bosgame M5 (AMD Ryzen AI Max+ 395, Radeon 8060S, `gfx1151`,
one GPU, Linux, HIP backend; the receipts record it as `rocm`). Times below are UTC, as in the receipts.

| Folder | Measured | Binary SHA-256 (`tensorfold-native`, HIP build) | Source |
| --- | --- | --- | --- |
| `out/` | 17:33 to 17:47 | `e279e29a54b56dc81ff70985f8c4cffaa0d9aff4a7456a234eb7247667492247` | see below |
| `out-r5b/` | ended 18:18 | `7fcaa74ec93561902ef97675f00db1b6d022392c665a7578960e18c92bcd2407` | `out/` plus the identity fix |

The `out/` binary was built after the merge that `heads.txt` records (`7b84f909`, made on the box and not published)
of `hip-refuses-drafter` and this branch. The bundle prepared for the run carries main at `76aafcf2`,
`hip-refuses-drafter` at `920d1348` and this branch at `1b4b2d7b`; the branch's later `d2dfb603` changes comments
only. The `out-r5b/` binary adds `benchmark-rocm-host-identity` at `624e2ce6` (merge `25a7201f`,
`out-r5b/heads-r5b.txt`). The second line of `heads.txt` is `git rev-parse` echoing a ref the bundle clone did not
have (the matching `fatal:` line is in `run.log`).

The checkpoints are `TensorFold/Qwen3.8-27B-MLX-4bit` at `70ae7fac63274ff2eac54152031433374cb80f2f` and
`TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP` at `f84b054c5677b6e59bdb969ae61915b0983d68bb` (`pull.log`). Every run is
community-v1: code and chat, temperatures 1 and 0, one warm-up and five repeats per cell, 256 tokens per reply.

## The runs

| Run | How | Drafter passed | Rounds | Drafted | Accepted | Server log |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `out/27b-auto` | managed | no | 5030 | 262 | 70 | not kept |
| `out/35b` | managed | no | 2495 | 4490 | 2615 | not kept |
| `out/35b-serial` | managed, `serial: true` | no | 5100 | 0 | 0 | not kept |
| `out/35b-attached` | attached to `out/35b-serve.log`'s server | unknown | 2441 | 4626 | 2668 | `out/35b-serve.log` |
| `out-r5b/35b-attached` | attached to `out-r5b/35b-serve.log`'s server | unknown | 2473 | 4576 | 2636 | `out-r5b/35b-serve.log` |

Each total is the sum of its 20 samples and the same as the run's own `Draft counts:` line (line 37 of each `.log`).
A managed run unlinks its server's log as it starts, so only the attached runs can be held against a server log.

## Files

- `out/<run>.json`: the board's receipt, community-v1 fields only. `out/<run>.drafts.json`: the counts file beside
  it. `out/<run>.log`: the run's console output. The same three for `out-r5b/35b-attached`.
- `out/35b-serve.log`, `out-r5b/35b-serve.log`: the attached server's log for each attached run.
- `out/35b-manifest.json`, `out-r5b/35b-manifest.json`: the attached server's benchmark manifest, saved with each run.
- `out/27b-dflash2.log`, `out/27b-serve-drafter.log`: two refusals of `--drafter` on HIP (below).
- `run.log`, `out-r5b/run-r5b.log`: the box's run script: merges, build exit, binary SHA-256, each step's exit, GPU
  memory and busy readings between steps, and the state of the box's other LLM service before and after.
- `heads.txt`, `out-r5b/heads-r5b.txt`: the merged source. `binary.sha256`: the first binary's SHA-256.
  `pull.log`: the two checkpoint downloads.
- `match.py`, `match.txt`: the check of the counts against the server logs, and its output (written for this folder).
- `host-controls/`: the CPU checks of the overflow fix. `SHA256SUMS`: see the last section.

## Counts against the server's log

Each attached server prints one line per finished request, for example `out/35b-serve.log` line 10:
`done cmpl-... tokens=256 sha=5d7effdcf06f finish=length rounds=106 accepted=150/246`, where `accepted=A/D` is
accepted over drafted. The POST line just before it names the route. In both server logs the 24 finished requests are
lines 8 to 54 (every even line), each after its POST line: one warm-up per cell at lines 8, 20, 32 and 44, then the
cell's five repeats. Warm-ups are not samples, so the file leaves them out.

`match.py` (Python 3, standard library) reads the sanitized copies here and checks, for every sample of both attached
runs, that the file's rounds, drafted, accepted and completion tokens are the server's for that request, that the
receipt's `token_sha` is the server's `sha=`, and that the route is `/v1/completions` for code and
`/v1/chat/completions` for chat. It prints each sample beside its log line, route and `finish=` (all `length`), since
the counts file records neither the route nor the finish reason and the receipt's schema stays as it is. For all five
runs it checks that the totals are the samples' sum and the run's printed line. Its output is `match.txt`. Two
controls run in the same script: a file with one drafted count off by one, and the server log read one request out of
step; it refuses both.

```sh
python3 docs/receipts/drafter-counts/match.py
```

## What else the GPU runs show

- `out/35b` and `out/35b-serial` load the same checkpoint. With `serial: true` the server drafted no tokens: 255
  rounds for each 256-token reply. The default run drafted 4490 tokens and accepted 2615 in 2495 rounds.
- `out/27b-auto.log` says the run passes no drafter, as the HIP engine runs no DFlash2 drafter yet, and the file says
  `drafter_passed: false`. The server still reported 262 drafted and 70 accepted tokens in 14 of the 20 attempts.
  These receipts do not show where those drafts came from: the managed run's server log is not kept.
- `out/27b-dflash2.log` and `out/27b-serve-drafter.log` are the refusals from `hip-refuses-drafter`: the benchmark and
  `serve` both refuse `--drafter` on HIP. `run.log` records both steps (R2a, R2b) exiting 1, and no `27b-dflash2.json`.
- `out/35b-manifest.json` and `out-r5b/35b-manifest.json` are the attached server's benchmark manifest, saved with
  each run; the second carries the identity fix. The counts match their server logs in both runs.

## Host controls

`host-controls/` holds the CPU-only checks, on a MacBook (Apple silicon, macOS, Zig 0.17.0) on 19 October 2026.
`commands.txt` says what each log ran.

- `host-tests.log`: the host test step on `616d3065`, the overflow fix: 292 of 292 steps succeed. Steps whose inputs
  did not change since the first run come from the cache, so its summary counts the 958 tests that ran again (957
  passed, 1 skipped), among them the 433 of 433 in the server's test binary, which holds the one new unit test and
  the one new case in the existing run test.
- `host-tests-full.log`: the first, uncached run of the same step, before the commit, on a tree that differs from
  `616d3065` only in one comment's wording: 292 of 292 steps, 4315 of 4331 tests passed, 16 skipped.
- `negative-control.log`: `zig/src/server/benchmark_drafts.zig` put back as `d2dfb603` has it, the unchecked sums,
  with the new run test kept. That test aborts with `panic: integer overflow` at `t.rounds += d.rounds`
  (`benchmark_drafts.zig:18`, from `totals`), in both binaries that hold the server's tests: 431 passed and 1
  crashed of 432 in each. The summary: 288 of 292 steps, 2 failed.
- `lean.txt`: the lean check finds 503 problems on `616d3065`, on main (`b7ce8121`) and on the merge base
  (`76aafcf2`), none new. `fmt.txt`: `zig fmt --check` passes.

In a passing log, a `failed command:` line is the build runner printing a test binary's stderr (expected warnings
from the tests) under its command; the `Build Summary` line is the result.

The GPU runs predate the overflow fix. For sums that fit in 64 bits the fix writes the same file and the same line.

## Sanitized copies

These are sanitized copies of the raw files. Changed: `/home/<user>` became `~` in `binary.sha256`, `run.log`,
`pull.log`, `out-r5b/run-r5b.log` and the `Receipt saved:` and `Draft counts:` lines of every run's `.log`;
trailing spaces were removed from line 1 of both `35b-serve.log` files; the host logs' local paths became `~` and
their trailing spaces were removed. Every receipt, counts file and manifest is byte for byte the raw file, so its hash
in `SHA256SUMS` is the original's. Counts, hashes, run IDs, times and the binary SHA-256 values above are unchanged.

The hashes of the raw files that were sanitized are not published. Such a file differs from its published copy only in
the replaced paths and the removed spaces, so its raw hash would let anyone who guesses a replaced path check the
guess. `SHA256SUMS` lists the published bytes of every file here except itself; run `shasum -a 256 -c SHA256SUMS` in
this folder. Not copied: an empty `build.log` (the build's exit, 0, is in `run.log`) and two scripts that check
receipts against the board's backend list, which is not about draft counts.
