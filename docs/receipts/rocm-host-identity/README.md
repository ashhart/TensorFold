# ROCm host identity receipts (#1001, PR #1043)

Evidence for [#1043](https://github.com/ashhart/TensorFold/pull/1043): GPU receipts from a Bosgame M5 (one Radeon
8060S, `gfx1151`) and host-side test controls from a Mac (Apple M5 Max, CPU only). Every file here is a sanitized
copy; see [Sanitizing](#sanitizing) for what changed. `SHA256SUMS` covers every file in this folder except itself
(`shasum -a 256 -c SHA256SUMS` from this folder).

## What the GPU receipts show, and what they do not

The GPU receipts were measured on 18 Oct 2026, before the two visible-device fixes: `82237515` ("a ROCm
visible-device index counts every GPU agent") and `fb8eaf85` ("the ROCm probe keeps the GPUs HIP keeps, by HIP's own
rules"). They change what the probe records only when `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES` or
`ROCR_VISIBLE_DEVICES` is set, or when a GPU agent's name fails the allowlist. With none of those variables set and
one GPU agent with a reportable name, as on this box, the old and the new probe record the same GPU with source
`detected`. The run logs do not record those variables, so these receipts do not exercise either fix; the host tests
and the negative controls in [`host-controls/`](#host-controls-host-controls) do.

### `gpu-run/`: managed runs and the first attached run

Measured 18 Oct 2026, 17:31 to 17:47 UTC (the `t=` stamps in `run.log`; each receipt's `created_at` is UTC and
agrees with them). Hardware as the receipts record it: CPU `AMD RYZEN AI MAX+ 395 with Radeon 8060S`, GPU
`AMD Radeon 8060S Graphics` with 40 compute units and 128,849,018,880 bytes, 131,209,781,248 bytes of system memory,
`unified`, source `detected`.

Source: the git bundle sent to the box carried `main` `76aafcf2`, this branch at `5906a3f7` (its first four commits) and
the other two #1001 branches, `hip-refuses-drafter` `920d1348` and `benchmark-drafter-counts` `1b4b2d7b`. `run.log`
shows those two merged in; `heads.txt` line 1 is the merge commit made on the box, `7b84f909`. Line 2 reads
`origin/main` literally: `git rev-parse origin/main` failed on the box (the `fatal:` line in `run.log`).

| File | What it is |
| --- | --- |
| `heads.txt` | the source commit the binary was built from (above) |
| `binary.sha256` | sha256 of that build's `tensorfold-native`: `e279e29a54b56dc81ff70985f8c4cffaa0d9aff4a7456a234eb7247667492247` |
| `run.log` | the driver's log: merge, disk and memory before the run, the pinned snapshots, build exit, each run's exit and the GPU's busy/GTT stamps |
| `pull.log` | the two checkpoints pulled into the Hugging Face cache at pinned revisions |
| `27b-auto.json`, `.drafts.json`, `.log` | R1: managed `TensorFold/Qwen3.8-27B-MLX-4bit@70ae7fac...`, `--drafter auto` |
| `27b-dflash2.log`, `27b-serve-drafter.log` | R2a, R2b: the `--drafter` refusals for the `hip-refuses-drafter` PR (exit 1 by design; no receipt) |
| `35b.json`, `.drafts.json`, `.log` | R3: managed `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP@f84b054c...` |
| `35b-serial.json`, `.drafts.json`, `.log` | R4: the same, `--serial` |
| `35b-serve.log`, `35b-manifest.json`, `35b-attached.json`, `.drafts.json`, `.log` | R5: the same snapshot served, then benchmarked attached |
| `check.mjs` | the harness that runs the board's `protocol.js` on each receipt (see below) |

R1, R3, R4 and R5 each ran 20 attempts, and all 80 are `ok`. R1, R3 and R4 name the CPU and GPU above, and their
checkpoint's `revision`, `config_sha256` and `tokenizer_sha256`, with `local` false. None of their logs prints the
"board ranks" note. R5 found the bug commit 5 fixes: `35b-manifest.json` says `"backend": "hip"` with `"gpus": []`, so
the client refused the manifest and `35b-attached.json` has unknown hardware and no revision.

`check.mjs` imports the board's validator as `protocol.mjs`: `https://tensorfold.dev/benchmarks/protocol.js` as
fetched on 18 Oct 2026, sha256 `0252160a717f4f50e12f9b019beff760ee10e415503ee504d72fc6289fc9cc4b`. That file is the
board's own code and is not copied here. The output of the 18 Oct run on the box was not saved;
`host-controls/board-validator.log` is a re-run on 19 Oct on the Mac with the same two files (same sha256) on the
receipts in this folder. Each receipt is refused as `rocm` ("Unsupported benchmark runtime.") and accepted when only
`runtime.backend` is changed to `cuda`, as the PR body says. The `cuda` copies are a schema check only and exist only
in memory.

`build.log` was empty (0 bytes) and is not included; the build's exit is in `run.log`.

### `gpu-attached-rerun/`: the attached run with commit 5

Measured 18 Oct 2026 on the same Bosgame M5; the receipt's `created_at` is `2026-10-18T18:18:44Z`.
`heads-r5b.txt`: `25a7201f`, this branch at `624e2ce6` (commit 5) merged into `7b84f909`. `run-r5b.log`: build
exit 0, `tensorfold-native` sha256 `7fcaa74ec93561902ef97675f00db1b6d022392c665a7578960e18c92bcd2407`, run exit 0.

| File | What it is |
| --- | --- |
| `heads-r5b.txt`, `run-r5b.log` | source, build exit, binary sha256, run exit |
| `35b-serve.log` | the HIP server, `gfx1151` device 0, serving the same 35B snapshot on loopback |
| `35b-manifest.json` | `GET /v1/benchmark/manifest`: backend `rocm`, the same revision and hashes, the GPU with 40 cores |
| `35b-attached.json`, `.drafts.json`, `.log` | the attached receipt: 20 of 20 `ok`, backend `rocm`, the CPU, GPU, memory and revision above |

The manifest attests no rank count, so the receipt's `settings.rank_count` is null and it stays unranked by `main`'s
rule. The log's line names "a rank count and checkpoint revision" because the client prints one message when either
is missing; the revision is present in both the manifest and the receipt.

## Host controls (`host-controls/`)

Run on a Mac with an Apple M5 Max (macOS 27.0.1), CPU only, Zig 0.17.0, with private Zig caches. Times are UTC. The
full suite is `zig build test -Dcpu=apple_m1 --summary all -j8`. The negative controls run the server tests only,
`zig build test-server-cpu --summary all -j8` (the step that holds `benchmark_facts_test.zig` and, from `fb8eaf85`,
`benchmark_rocm_test.zig`; the 19 Oct runs add `-Dcpu=apple_m1`); each log starts with the undo as a diff. The
controls on `82237515` ran in the branch checkout, and the file was restored with `git checkout` after each. The three
controls on `fb8eaf85` ran in a scratch copy of that commit (`git archive`), one undo at a time; after each the file
was copied back and compared byte for byte with the commit's (the last line of each log), and the copy was discarded
afterwards.

| File | Tree | When | Build steps | Tests | Server test binary |
| --- | --- | --- | --- | --- | --- |
| `test-main-76aafcf2.log` | `main` `76aafcf2` | 18 Oct 23:12 | 292/292 | 4305 of 4321 passed, 16 skipped | 428 passed |
| `test-branch-82237515.log` | this branch, `82237515` | 19 Oct 07:56 | 292/292 | 4317 of 4333 passed, 16 skipped | 434 passed |
| `test-branch-fb8eaf85.log` | this branch, `fb8eaf85` | 19 Oct 08:51 | 292/292 | 4321 of 4337 passed, 16 skipped | 436 passed |

The server test binary runs twice in `zig build test` (in `test` and in `test-cuda-host`), so its new tests add twice
their number: six at `82237515` (five from commits 1 to 5, one from `82237515`) add 12, and two more from `fb8eaf85` add
four. Each branch run was on a tree with no tracked file modified and started from an empty local cache, so every test
ran. The `failed command:` lines in these logs follow warnings that tests print on stderr; each `Build Summary` line
counts no failed test.

| File | What was undone | When | Result | The failure |
| --- | --- | --- | --- | --- |
| `negative-control-agent-positions.log` | `82237515`: `benchmark_host.zig` as it was before the fix (an unreportable agent dropped before the indices apply), plus only the `rocmVisible` helper the new test calls | 19 Oct 07:52 | 433 of 434, 1 failed | the new test: `HIP_VISIBLE_DEVICES=1: listed GPUs [], want [AMD Radeon PRO V620]`, `HIP_VISIBLE_DEVICES=0: listed GPUs [AMD Radeon PRO V620], want []`, and the same two lines for `ROCR_VISIBLE_DEVICES` |
| `negative-control-agent-positions-minimal.log` | the same fix, a smaller undo: `amdAgents` skips an unreportable agent again | 18 Oct 23:28 | 433 of 434, 1 failed | the same four lines |
| `negative-control-backend-name.log` | commit 5: the manifest names the engine (`hip`, `metal`) | 18 Oct 23:30 | 433 of 434, 1 failed | the backend-name test: expected `rocm`, found `hip` |
| `negative-control-revision.log` | commit 3: the manifest's revision from `TF_CHECKPOINT_REVISION` only | 18 Oct 23:31 | 432 of 434, 2 crashed | the manifest-agreement and attested-revision tests: `attempt to use null value` at the revision |
| `negative-control-cores.log` | commit 3: the manifest sends `cores: null` and the client does not read cores | 18 Oct 23:32 | 432 of 434, 2 failed | the manifest-agreement and cores tests: `expected 40, found null` |
| `negative-control-w-slash.log` | commit 2: `w/` kept as is in CPU names | 18 Oct 23:33 | 433 of 434, 1 crashed | the rocm probe test: no CPU name where `AMD RYZEN AI MAX+ 395 with Radeon 8060S` is expected |
| `negative-control-ranking-note.log` | commit 4: the ranking note never names a missing fact | 18 Oct 23:35 | 433 of 434, 1 crashed | the rocm probe test: no note where `GPU` is expected |
| `negative-control-rocr-twice.log` | `fb8eaf85`, part 1: `rocmVisible` as in `82237515`, so `ROCR_VISIBLE_DEVICES` applies again and `CUDA_VISIBLE_DEVICES` and clr's reading of a value do not | 19 Oct 08:54 | 434 of 436, 2 failed | `ROCR_VISIBLE_DEVICES=1 (rocminfo lists 1 GPUs): recorded [] partial`, `ROCR_VISIBLE_DEVICES=1,2 (rocminfo lists 2 GPUs): recorded [AMD Instinct MI210] detected`, and the `CUDA_VISIBLE_DEVICES`, empty `HIP_VISIBLE_DEVICES` and quoted `"1"` rows; in the agent-position test, both `CUDA_VISIBLE_DEVICES` rows |
| `negative-control-hip-list.log` | `fb8eaf85`, part 2: the list parsed by `82237515`'s `select` again | 19 Oct 08:55 | 435 of 436, 1 failed | 12 of the 15 rows of HIP's list rules, among them `0,3,2: recorded [AMD Radeon PRO W7800, AMD Instinct MI210]`, `0,0: recorded [AMD Radeon PRO W7800, AMD Radeon PRO W7800]`, `2,0,2` with three cards, `1,x,0`, `0,`, `01`, `1` after a space, and every `GPU-` row that names a card |
| `negative-control-partial.log` | `fb8eaf85`, part 3: no `partial` when a GPU HIP keeps is unnamed or a `GPU-` entry is unresolved | 19 Oct 08:56 | 434 of 436, 2 failed | the agent-position test with no variable (printed as `=`) and with `0,1`: `listed GPUs [AMD Radeon PRO V620] detected, want [AMD Radeon PRO V620] partial`; and `1,GPU-b2b2` on `rocminfo` text with no `Uuid` lines: `detected`, want `partial` |

The first seven ran on `82237515`, after it was committed (23:22 UTC on 18 Oct): the five controls from the PR body
again, and the agent-position one twice. In both agent-position controls the cases with no selection and with `0,1`
still pass, because dropping an agent before or after the indices gives the same list when nothing or everything is
selected. Each selection that names one position fails, in both directions: `1` loses the card it names, and `0`
reports the next card. Their last two rows set `ROCR_VISIBLE_DEVICES`; `fb8eaf85` turned those rows into
`CUDA_VISIBLE_DEVICES` rows, because the probe no longer reads `ROCR_VISIBLE_DEVICES` (below), and gave every row the
source it expects.

The last three ran on `fb8eaf85`, one per part of that fix. In each, the rows that part does not decide still pass: with
part 1 undone, the two rows where `HIP_VISIBLE_DEVICES` holds an index (2 of 7); with part 2 undone, `-1`, the
upper-case `GPU-` row and `1,0` on text with no `Uuid` lines, which the old parser turns into the same list (3 of 15);
with part 3 undone, every row where the probe names every GPU HIP keeps. Each control's other failures are listed in its
log.

| File | What it is |
| --- | --- |
| `lint.log` | on `fb8eaf85`: `zig fmt --check build.zig zig`, exit 0; `tools/zig/lean_check.py` on `main` `76aafcf2` and on this branch, 503 problems each, and `lean_new.py`: 0 new |
| `board-validator.log` | the board's validator re-run on the GPU receipts, described above |

## The ROCm source `fb8eaf85` follows

Read on 19 Oct 2026 at ROCm/ROCR-Runtime `547bc139`, ROCm/clr `dfa6202d` and ROCm/rocm-systems `df181d33`, the `develop`
heads that day. In `rocm-systems`, `projects/clr`'s visible-device block in `rocdevice.cpp` and
`projects/rocr-runtime`'s `amd_filter_device.cpp` are the same as in the other two.

- ROCr applies `ROCR_VISIBLE_DEVICES` before anything lists a GPU. `BuildTopology` in
  `runtime/hsa-runtime/core/runtime/amd_topology.cpp` (lines 401-459) ranks the GPUs the variable names in its
  order and creates the others disabled; `RegisterAgent` in `runtime.cpp` (236-245) keeps disabled
  GPUs out of the GPU list, and `IterateAgent` (312-329), which `rocminfo` reaches through `hsa_iterate_agents`, walks
  the CPU, GPU and AIE lists. So `rocminfo` prints only the visible GPUs, renumbered, and the probe must not apply
  the variable again.
- HIP reads `HIP_VISIBLE_DEVICES`, else `CUDA_VISIBLE_DEVICES` (`rocclr/device/rocm/rocdevice.cpp` 390-393), over
  every GPU agent in `hsa_iterate_agents` order (`iterateAgentCallback`, 327-349), the order `rocminfo` prints. An
  entry with `GPU-` in it is matched as a substring of each agent's UUID, first match wins (408-421); an index must
  read back unchanged through `std::to_string(atoi(entry))` and be below the GPU count (422-426); the first invalid
  entry ends the list (428-431), and a repeated index is skipped (433-436).
- clr reads these variables in `rocclr/utils/flags.cpp`: a variable set to nothing is stored as one space (121-124),
  which no entry rule accepts, so HIP sees no GPU; `removeQuotes` (26-61, applied at 187) drops one pair of double
  quotes around the value.
- `rocminfo` prints every agent's UUID on its `Uuid:` line (`projects/rocminfo/rocminfo.cc` 450-453 and 665 in
  `rocm-systems`), the value HIP compares: `GPU-` and 16 lower-case hex digits, or `GPU-XX` for a device with none
  (`amd_gpu_agent.cpp` 2057-2075 in ROCR-Runtime). The probe compares it and never records it.

## Sanitizing

These are copies of the raw files, changed only as follows. `provenance.tsv` lists every file's published sha256 and
which change it had. A file marked `unchanged` (the JSON receipts and manifests among them) is byte-identical to the raw
file, so its raw and published sha256 are the same. For each file that was changed the raw sha256 is withheld
(`withheld: sanitized`): such a file differs from its published copy only in the replaced paths and the stripped
spaces, so its raw hash would let anyone who guesses a replaced path check the guess.

- `/home/<user>` and `/Users/<user>` became `~` (logs and `binary.sha256`).
- Trailing spaces were stripped, so the files pass `git diff --check`: the two `35b-serve.log` files, whose first line
  ended in a space, and five negative-control logs, where the undo diff's blank context lines and the test runner's
  blank lines were only spaces.

The binary sha256 values above are the recorded originals. No file names a hostname (the machine is called only
"Bosgame M5", here), a private address, MAC address, serial number, Wi-Fi name, proxy or credential; the only address
is loopback `127.0.0.1`.
