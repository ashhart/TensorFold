# HIP refuses an external drafter (gfx1151)

Receipts for [#1044](https://github.com/ashhart/TensorFold/pull/1044). On the HIP engine, `serve --drafter` and
`tensorfold benchmark --drafter` each stop with one diagnostic before any weights load. Refusing the external
DFlash2 drafter does not turn drafting off: the engine's own drafts stay on.

Measured on 2026-10-18 from 17:33 to 17:42 UTC (19:33 to 19:42 CEST) on one Bosgame M5: AMD Ryzen AI MAX+ 395,
Radeon 8060S (`gfx1151`), Linux. The checkpoint is `TensorFold/Qwen3.8-27B-MLX-4bit` at
`70ae7fac63274ff2eac54152031433374cb80f2f`, read offline from the Hugging Face cache (`HF_HUB_OFFLINE=1`,
`run-script.sh` line 50). `z-lab/Qwen3.8-27B-DFlash2` was in the same cache: serve resolves `--drafter` to its
folder before the engine opens, and that lookup succeeded, since the serve refusal below comes from the engine.

The source was a local merge on the box, `7b84f909`, never pushed: `main` at `76aafcf2` with #1043 at `5906a3f7`,
this PR at `920d1348` and #1045 at `1b4b2d7b`. This PR's runtime code at `920d1348` is the code at its head; later
commits change comments, one test name, docs and these receipts only. The binary was built by
`zig build native-hip -Dhip-arch=gfx1151 ... -Doptimize=ReleaseFast` with Zig 0.17.0 (`run-script.sh` line 46,
`build_exit=0` at `run.log` line 38, an empty `build.log`). Its SHA-256, as recorded on the box (`run.log` line 39):
`e279e29a54b56dc81ff70985f8c4cffaa0d9aff4a7456a234eb7247667492247`. The `Draft counts` line and
`27b-auto.drafts.json` come from #1045, not from this PR.

## The refusals: one diagnostic each, before any weights load

Below, `$B` is that binary and `$M27` is `TensorFold/Qwen3.8-27B-MLX-4bit`, as in `run-script.sh`. Each refusal's
log is the command's whole output (stdout and stderr) and is one line long: the diagnostic is printed once, and
nothing is printed before or after it.

| Run | Command (`run-script.sh`) | Exit (`run.log`) | The whole log |
| --- | --- | ---: | --- |
| R2a | line 54: `$B benchmark $M27 --drafter z-lab/Qwen3.8-27B-DFlash2 --output .../27b-dflash2.json` | 1 (line 43) | `27b-dflash2.log` line 1 |
| R2b | line 55: `timeout 180 $B serve $M27 --backend rocm --host 127.0.0.1 --port 8093 --drafter z-lab/Qwen3.8-27B-DFlash2` | 1 (line 46) | `27b-serve-drafter.log` line 1 |

`27b-dflash2.log` line 1:
`tensorfold benchmark: --drafter: the HIP engine runs no DFlash2 drafter yet, so a rocm run takes none; --drafter none runs without it`

`27b-serve-drafter.log` line 1:
`tensorfold: --drafter: the HIP engine runs no DFlash2 drafter yet; serve without --drafter`

**R2a stops before the run starts.** A run's first output is its suite banner, as in `27b-auto.log` lines 1 and 2
(`community-v1: ...` and `This generates model replies; ...`). R2a's log has neither. No receipt was written
(`run.log` line 44: `ls: cannot access '~/tf-1001-runs/out/27b-dflash2.json'`). In the code the refusal is the
first check once the checkpoint's family is known, before the drafter cache lookup and before any serve child
(`zig/src/server/benchmark.zig` line 338, `if (rocm and given) return fail(ctx, rocm_drafter);`; the banner is
lines 381-382).

**R2b stops before any weights load.** The refusal is the first line of `checkGroup` (`zig/src/native/hip.zig`
line 54). `ready()` runs `checkGroup` first (line 84), before it reads the model's window, queries the GPU (line 92)
or prints the policy (line 102), and `openWith` runs `ready()` (line 218) before the family's `open` reads any
weights (line 224). A follower rank goes through `ready()` too (line 252). A HIP serve that gets past that point
prints three lines on its way to serving: the policy line `[tensorfold] HIP rank 0 of 1: policy ...`, the device
line naming the model's size, and `serving ... loaded in N s`. For this checkpoint on a Bosgame M5 they are
lines 2, 4 and 5 of `docs/receipts/gfx1151-qwen-833bbfcb/27b-server-8092.log` (`model 14.10 GiB`,
`loaded in 37.9s`). R2b's log has none of them.

**Both finished within one second.** `run-script.sh` prints a GPU reading (`gtt()`, line 12, stamped with
`date -u +%T`) after R1, before R2b and after R2b. All three are stamped `17:42:02` (`run.log` lines 42, 45
and 47), so R2a and R2b each started and exited within that second, against the 37.9 s this checkpoint took to
load in that earlier receipt. R2b exited 1 by itself, not 124 from `timeout`. The GTT readings (1,129,906,176
then 1,129,861,120 bytes) are taken outside the process, after it exited: they show nothing was left held, not
that nothing was allocated.
The evidence for "before any weights load" is the one-line logs, the timing and the code path above.

These runs are one rank on one GPU. The refusal on a follower rank (`--tp 2 --rank 1`) is covered by the host
test in `zig/src/native/hip.zig`, not by a GPU run.

## What still drafts: refusing DFlash2 does not mean plain decoding

R1 is `$B benchmark $M27 --output .../27b-auto.json` (`run-script.sh` line 52), with the default `--drafter auto`.
It ran from 17:33:19 to 17:42:02 (`run.log` lines 40 to 42) and exited 0. `27b-auto.log` line 3 says
`This run passes no drafter: the HIP engine runs no DFlash2 drafter yet.`, the receipt `27b-auto.json` has
`"drafter_repo_id": null`, and `27b-auto.drafts.json` has `"drafter_passed": false`.

The run still drafted. Its 20 attempts drafted 262 tokens and kept 70 of them, over 5,030 rounds
(`27b-auto.drafts.json` `totals`, and `27b-auto.log` line 37). These drafts did not come from an MTP head,
because this checkpoint has none. At this revision `model.safetensors.index.json` (git blob `05a3a3a1`, the digest
pinned in `docs/receipts/gfx1151-qwen-833bbfcb/checkpoints.json`) lists 2,180 tensors and none of them is an
`mtp` tensor, and the revision has no `mtp*.safetensors` file. The HIP engine looks for a head in
`mtp*.safetensors` or among the shards' own tensors, and the shards hold the same 2,180 (`Checkpoint.mtp` in
`zig/src/families/qwen3_5/weights/checkpoint.zig`), and the 1.0.12 model check on this checkpoint reports `the
model has no draft head` (`docs/receipts/gfx1151-qwen-833bbfcb/27b-model-check.log` line 29). The policy line's
`mtp drafts 3 under 0.3` is the policy, printed whether or not a head loads. An earlier reading in the PR
description took these drafts for MTP drafts; they are not.

What drafted is the lane core's copy proposer (`SuffixLookup`): it proposes tokens copied from the stream's own
context, on every request that drafts. The counts fit it: 6 of the 20 attempts drafted nothing in about 255
rounds, and the others drafted 15, 34 or 48 tokens each. An MTP head proposes on most rounds: the MTP checkpoint
in `docs/receipts/gfx1151-qwen-833bbfcb/35b-server-8092.log` shows `rounds=33 accepted=30/68`, for example.
On a checkpoint with MTP weights the HIP engine drafts with the head as well as with copies. `serve --mtp-drafts 0`
stops the head's drafts alone (the head still loads); `serve --no-drafts`, or `tensorfold benchmark --serial`,
turns both off and decodes plain.

## Host controls (`host-controls/`)

The two tests this PR adds were run with each part of the refusal taken out, then with everything put back.
Each control is commit `43125a8c` (this PR's code at its head) plus one patch, and each log starts with the patch it
ran. Host tests on an Apple M-series Mac, CPU only, no GPU: `zig build test -Dcpu=apple_m1 --summary all -j8` from
the repository root, Zig 0.17.0, on 2026-10-19 from 08:06 to 08:19 UTC. The controls reused the build cache,
so their summary lines count only the steps that ran again; the restored run used a fresh cache, so every test ran.

| Control | What was taken out | Result |
| --- | --- | --- |
| 1 | the `--drafter` line at the top of `checkGroup` (`zig/src/native/hip.zig`) | exit 1. `control-1-serve-refusal-removed.log` lines 21 and 22: `hip.test.--drafter is refused before any load, on every rank, until the HIP engine runs DFlash2` failed, `expected error.Refused, found void`, at `hip.zig:309` (line 26). Summary, line 88: 1 failed |
| 2 | the benchmark's named-drafter refusal (`zig/src/server/benchmark.zig`) | exit 1. `control-2-benchmark-refusal-removed.log` lines 79 and 90: `a rocm run refuses a named --drafter before any serve child, and auto names no drafter on rocm` failed at `benchmark_run_test.zig:232` (lines 83 and 94), once in each server test binary (428 of 429 pass in each). Summary, line 99: 2 failed |
| 3 | the `rocm` case in `autoDrafter` (`zig/src/server/benchmark.zig`) | exit 1. `control-3-auto-rocm-case-removed.log` lines 81 and 92: the same test failed at `benchmark_run_test.zig:238` (lines 85 and 96), once in each server test binary. Summary, line 101: 2 failed |
| restored | nothing | exit 0. `restored.log` line 545: `292/292 steps succeeded; 4308/4324 tests passed (16 skipped)`; both server test binaries pass 429 of 429 (lines 704 and 787) |

No other test failed in any control. Line 232 checks that the named-drafter run printed the refusal, printed no
banner and sent the fake server nothing; line 238 checks that `autoDrafter` names no drafter on rocm.

## Files

| File | What it is |
| --- | --- |
| `27b-dflash2.log` | R2a's whole output: the benchmark's refusal of a named `--drafter` |
| `27b-serve-drafter.log` | R2b's whole output: serve's refusal of `--drafter` |
| `27b-auto.log` | R1's output: the default `--drafter auto` run, with the line that it passes no drafter |
| `27b-auto.json` | R1's receipt (`drafter_repo_id` null) |
| `27b-auto.drafts.json` | R1's per-attempt draft counts, written by #1045's code |
| `run.log` | the session's script output: merge, preflight, build exit, binary SHA-256, exit codes, GPU readings |
| `run-script.sh` | the script that produced `run.log` and the files above, as kept on the Mac that sent it to the box |
| `heads.txt` | `git rev-parse` of the merged tree: `7b84f909`, then the literal `origin/main` it could not resolve |
| `binary.sha256` | the binary's SHA-256 as `sha256sum` printed it on the box |
| `build.log` | the HIP build's output: empty, the build printed nothing |
| `host-controls/control-*.patch` | the change each control applied to the commit's tree |
| `host-controls/control-*.log`, `host-controls/restored.log` | the host test runs for the controls and the restored tree |
| `host-controls/.gitattributes` | exempts the control patches and logs from git's whitespace check: verbatim tool output, and patch context lines must keep their trailing space |
| `SHA256SUMS` | SHA-256 of every file here except itself |

In `run.log`, lines 1 to 26 are git setting up the merge on the box; line 24 is the `rev-parse` error behind
`heads.txt`. Lines 27 to 35 are the preflight: disk, memory and the two checkpoints' cached revisions. Lines 36,
56 and 57 are the box's own LLM service, stopped for the run and restarted after it. R3 to R5 (lines 48 to 54) are
the same session's runs on `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP`; their files are not part of this folder.
`run-script.sh` was last changed at 17:13 UTC, before the session started at 17:31; the box's own copy was not
hashed.

## Sanitized copies

These are sanitized copies of the raw files, and `SHA256SUMS` covers the copies as published here. The changes:
the box user's home directory, `/home/<user>`, is written `~` in `27b-auto.log` (lines 36 and 37), `run.log`
(lines 31, 34, 39 and 44) and `binary.sha256` (line 1); `run-script.sh` line 2 says `the box user` where it named
the user; in the host-control logs the Mac's home directory is written `~`. Nothing else was changed: hashes,
counts, timings, exit codes and GPU readings are as recorded, and the binary's SHA-256 in `binary.sha256` and
`run.log` line 39 is the recorded value. The raw files' own SHA-256 values are not published, since a raw hash of
a file that differs only by that path would confirm the user name.

`27b-dflash2.log`, `27b-serve-drafter.log`, `27b-auto.json`, `27b-auto.drafts.json`, `heads.txt` and `build.log`
are unchanged originals, byte for byte. No file here holds an API key or token, a private or tailnet address, a
host name, a MAC address, a serial number, a Wi-Fi name or a proxy address.
