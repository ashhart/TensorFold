# Reviewer2 — cuda-checkpoint-cli branch review (RESOLVED: review fixes applied and verified)

Branch `cuda-checkpoint-cli` vs `main`: **3 files, +20/-2** (`build.zig`, `zig/build/cuda.zig`,
`zig/src/cli/cuda_main.zig`). Commit `d5afccf`. The branch wires the checkpoint subcommands
(`models`, `info`, `pull`) — which already exist on main for the Metal/native builds — into the
Linux CUDA `tensorfold` binary, and adds a GPU-free `--version`. The shared CLI modules
(`zig/src/cli/cli.zig`, `models.zig`, `info.zig`, `pull.zig`, `hub.zig`, `pull_file.zig`,
`pull_parts.zig`) are byte-identical to main.

## Resolution of the review findings (working tree, UNCOMMITTED — must be committed with the branch)

During the joint review pass, Reviewer1's + Reviewer3's + my findings were resolved in the
working tree on top of the branch (diff vs HEAD: `zig/build/cuda.zig`, `zig/src/cli/cuda_main.zig`,
`zig/src/cli/hub.zig`, `zig/src/cli/pull.zig`, `zig/src/cli/pull_file.zig`, `zig/src/server/hub.zig`,
`CHANGELOG.md`; +69/-7 before Reviewer2's one-line repair):

- **S-F1 CLOSED** — `pull_file.linkIntoSnapshot` now rejects empty/absolute/`.`/`..` components
  (`snapshotPathOk`) before any join/symLink/delete. Pre-fix escape proven on HEAD: a test
  against HEAD's `pull_file.zig` plants the symlink ("PRE-FIX ESCAPE CONFIRMED"); the same test
  against the working tree refuses with `error.SnapshotPath`. End-to-end: a mock hub serving a
  tree entry `"../../../../../pwned.txt"` (Review1's `malicious_hub.py`) leaves the cache intact.
- **S-F2 CLOSED** — `pull.zig resolveRevision` now requires the JSON `sha` to be 40-hex; both
  `hub.cachedSnapshot` and `server/hub.zig` only join a 40-hex `refs/main` revision into a cache
  path, else refuse/fall back. Reviewer3's post-fix `r3_cachedSnapshot_ref_traversal.zig`
  passes (2/2).
- **S-F3 CLOSED** — `revisionOk` validates `REPO[@REVISION]` before URL interpolation (no `..`,
  alphanumeric/-/_/. only); a bad revision exits 1 with a clear message.
- **S-F4 CLOSED** — the resume prefix is read in 64 KiB chunks, never allocated at
  network-controlled size (no OOM vector, no 32-bit `@intCast` panic).
- **F1/F2 (my nits) CLOSED** — `--version` writes to stdout via `writeStreamingAll`, accepts no
  trailing args (usage + exit 2), and any leading `--flag` prints usage + exit 2 without touching
  the GPU (no more leaked `FileNotFound`).
- **F3 (my nit) PARTIALLY CLOSED** — the `checkpoint_cli` module creation is deduped into a
  `checkpointCli()` helper used by `targets()`, `nativeServer()` and `hostTests()`. The unused
  `native_engines` import is still passed (cli.zig never imports it) — harmless, flag only.
- **F4 CLOSED** — CHANGELOG has the Unreleased entry.
- Reviewer2 repaired one defect in the fix itself: `zig/build/cuda.zig` line 150 kept a stray `)`
  from the replaced inline `createModule` (build broke). Fixed; `zig build install`,
  `zig build test` and `zig build test-cuda-host` all green.
- Reviewer1's `pull_traversal.sh` harness bug repaired by Reviewer2: `HF_ENDPOINT` was missing,
  so the pull hit the real hub and the mock hub was never exercised; the mock hub is now pointed
  at and started (with a port-in-use guard), and the second pull shows the refusal.

## What the change does

1. **`build.zig`**: `cuda_build.hostTests` now also receives `build_options`, so the host test
   executable can import the build version.
2. **`zig/build/cuda.zig`**:
   - `targets()`: builds a new `checkpoint_cli` module over `zig/src/cli/cli.zig`, imports it into
     the CUDA `tensorfold` executable module, and adds `build_options` to it. It also builds a new
     native-engines module via `engines(...)`.
   - `hostTests()`: same wiring for the host test executable (debug optimize, host target).
3. **`zig/src/cli/cuda_main.zig`**: `--version` handled first (exit 0, no driver/GPU), then
   `checkpoint_cli.wants(args[1..])` dispatches `models`/`info`/`pull` to `cli.zig` before any
   `cuda.Driver.open()`; the three subcommands are added to the usage text.

## Verification (all proven in `smoke_tests/`, ALL PASS on a real Linux+CUDA host, RTX 5090s)

- `smoke_tests/smoke_cuda_checkpoint_cli.sh` (Review2): 16/16 checks — build, version-on-stdout
  without driver/GPU, strace dispatch proof (models=0 vs run=6 libcuda opens), arg handling,
  post-fix --version ergonomics.
- `smoke_tests/version_stdout.sh` (Review1): PASS post-fix.
- `smoke_tests/pull_traversal.sh` + `malicious_hub.py` (Review1, harness repaired): refusal plus
  nothing outside the cache.
- `smoke_tests/r3_linkIntoSnapshot_traversal.zig`, `r3_cachedSnapshot_ref_traversal.zig`,
  `r3_negative_controls.zig` (Review3, post-fix versions): all pass; plus the HEAD-vs-working-tree
  comparison run above proving the pre-fix escape and the post-fix refusal.
- `zig build install`, `zig build test`, `zig build test-cuda-host`: green.
## Findings

### F1 (nit, non-blocking) — `--version` only with exactly one argument
`if (args.len == 2 and std.mem.eql(u8, args[1], "--version"))` — `tensorfold --version anything`
falls through into the run path and leaks a bare `error: FileNotFound` (exit 1) after touching the
driver open path. Proven in smoke check C4. Suggest accepting `--version` when it is `args[1]`
regardless of trailing args, or at least not letting a bare Zig error escape. Pre-existing exit
path style, cosmetic.

### F2 (nit, non-blocking) — `--version` prints to stderr
`std.debug.print` writes to stderr, while `models`/`info` write to stdout. `tensorfold --version
2>/dev/null` prints nothing. For a version query, stdout is the conventional stream and what
scripts capture. One-line fix: print via the stdout writer (`std.Io.File.stdout()`), consistent
with `cli.zig`'s writer plumbing.

### F3 (nit, non-blocking) — unused `native_engines` import in the `checkpoint_cli` module
`zig/build/cuda.zig` gives the new `checkpoint_cli` module a `native_engines` import, but
`cli.zig` never imports `native_engines` (only `hub.zig` does, and `cli.zig` does not import
`hub.zig`). Harmless to the binary (Zig analyzes lazily), but it drags `zig/src/native/cuda.zig`
and the CUDA runtime into the build graph for nothing; `internal_tests`-style callers don't need
it. Also `nativeServer()` creates a second structurally identical `checkpoint_cli` module — minor
build-graph duplication.

### F4 (doc gap, non-blocking) — CHANGELOG has no entry
CHANGELOG's Unreleased section has no line for "the CUDA `tensorfold` serves `models`/`info`/
`pull` and prints `--version`" despite the branch being user-visible. One line to add.

## Findings (all resolved in the working tree; see top section)

### F1/F2 — RESOLVED
`--version` stderr + `--version extra` leak + leading-flag leak: fixed on stdout/usage/exit 2;
smoke checks C4 green.

### F3 — PARTIAL (flag only)
Unused `native_engines` import in `checkpointCli()` (cli.zig never imports it). Harmless to the
binary; build-graph dead weight. One-line follow-up.

### F4 — RESOLVED
CHANGELOG entry added on top of d5afccf; check it reads correctly before the merge commit.

### Open questions from Reviewer1 (out of scope for the verdict)
- `pull_parts` fetch resume/marker corner cases: probed, no defect found in the reviewed paths
  (sha256-before-rename and size checks verified in Review3's negative controls).
- `info` sizeOf=0 on unreadable dirs (prints 0.00 GiB silently): cosmetic; suggest printing an
  error at `statFile` failure in a follow-up.
- Build-time cost of the second/`checkpointCli` module creation: deduped into one helper; the
  `native_engines` flag remains (F3).

## Security findings (folded from Reviewer3; severities graded pre-fix, all closed in the working tree)

Reviewer3's full security review is in `smoke_tests/README.md`; all severities were graded against
the pre-fix code and every finding is now closed by the working-tree fixes (see top section).
Reviewer2 independently re-ran all tests against the unmodified modules — **all pass**
(zig 0.17.0); the pre-fix escape was reproduced against HEAD for certification:

- `zig test --dep pull_file -Mroot=smoke_tests/r3_linkIntoSnapshot_traversal.zig -Mpull_file=zig/src/cli/pull_file.zig` → All 1 tests passed
- `zig test --dep pull_file --dep hub -Mroot=smoke_tests/r3_cachedSnapshot_ref_traversal.zig -Mpull_file=zig/src/cli/pull_file.zig -Mhub=zig/src/cli/hub.zig` → All 1 tests passed
- `zig test --dep pull_file --dep hub -Mroot=smoke_tests/r3_negative_controls.zig -Mpull_file=zig/src/cli/pull_file.zig -Mhub=zig/src/cli/hub.zig` → All 1 tests passed

- **S-F1 MEDIUM** — hub tree `path` (network JSON) reaches `pull_file.linkIntoSnapshot`'s
  `std.fs.path.join` unvalidated; a `"path": "../../outside/evil"` entry plants a symlink outside
  the cache and `deleteFile` can unlink an arbitrary user file first. Proven
  (`r3_linkIntoSnapshot_traversal.zig`). Not High: git forbids `..` filenames, so it needs a
  hostile `HF_ENDPOINT`/compromised hub; code identical on main (inherited, not introduced by
  this branch — the CUDA binary merely gains a second path to it).
- **S-F2 MEDIUM** — revision JSON `sha` and `refs/main` content are joined into paths raw;
  a `../..` revision resolves the serving snapshot outside the repo dir. Proven on real
  `hub.zig` (`r3_cachedSnapshot_ref_traversal.zig`). Same preconditions; inherited.
- **S-F3 LOW** — `REVISION` unvalidated in URL interpolation (path reshaping only, no SSRF).
- **S-F4 LOW** — resume prefix hashed via one `a.alloc` sized by network-controlled `e.size`
  (OOM on 64-bit; `@intCast` panic on 32-bit).
- **S-F5 INFO** — undigested (non-LFS) files unverified beyond git blob sha1 when present.
- No Critical/High under the team-lead's rule (both MEDIUMs needed hostile `HF_ENDPOINT`/
  compromised hub or local cache writer; git forbids `..` filenames). As graded, no P0 and no
  unresolved P1 existed — the MEDIUMs are additionally **closed** in the working tree now, along
  with the LOWs (S-F3, S-F4). S-F5 (undigested non-LFS files unverified beyond git blob sha1)
  remains informational, unchanged from main's posture.
- Positive controls verified in Review3's `r3_negative_controls.zig` (`isRepoIdLike`, `blobName`
  sanitization, sha256-before-rename, content-range start check) — all pass.
### Inherited (out of branch scope, for Reviewer3)

My inherited-F1 suspicion was confirmed by Reviewer3's proven proofs; see the security section above.

## Verdict

**APPROVE.** Reached with Reviewer1 (informed of the resolution) and Reviewer3 (confirmed
consensus in writing). The branch commit plus the reviewed working-tree fixes deliver exactly
what they claim; correctness and the security posture are proven on a real Linux+CUDA host.
MUST: commit the working-tree fixes (they are the review's resolutions) together with the branch
before merging. Remaining flags: the unused `native_engines` import (F3, one-line, optional) and
the `info` 0.00 GiB cosmetic (optional follow-up).
