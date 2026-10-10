# Reviewer2 — cuda-checkpoint-cli branch review

Branch `cuda-checkpoint-cli` vs `main`: **3 files, +20/-2** (`build.zig`, `zig/build/cuda.zig`,
`zig/src/cli/cuda_main.zig`). Commit `d5afccf`. The branch wires the checkpoint subcommands
(`models`, `info`, `pull`) — which already exist on main for the Metal/native builds — into the
Linux CUDA `tensorfold` binary, and adds a GPU-free `--version`. The shared CLI modules
(`zig/src/cli/cli.zig`, `models.zig`, `info.zig`, `pull.zig`, `hub.zig`, `pull_file.zig`,
`pull_parts.zig`) are byte-identical to main.

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

## Verification (all proven in `smoke_tests/smoke_cuda_checkpoint_cli.sh`, all PASS)

- `zig build install` and `zig build test-cuda-host` succeed with the new wiring.
- `tensorfold --version` prints `tensorfold 1.0.4` (the manifest version; `-Dversion` override
  flows through `build_options`), exits 0, no model/driver/GPU.
- **Dispatch before the driver opens is proven, not just asserted**: `strace -e openat` on
  `tensorfold models` shows **0** libcuda opens, while the control case (`run x --tokens 1`)
  shows **6**. `models` works on a machine with GPUs without ever opening the driver.
- Argument handling: `models extra`, `info` alone, `pull` alone → exit 2 with the checkpoint
  usage on stderr; `info NoSuchOrg/NoSuchModel` → exit 1 from the checkpoint CLI (cache miss,
  suggests `tensorfold pull`), not a driver/loader error; the bare `tensorfold` invocation still
  prints the full run usage and exits 2; `run`/`segments`/... unaffected.
- `@ptrCast(args[1..])` is layout-safe: `[][:0]const u8` and `[]const []const u8` both have
  16-byte elements, so the fat-pointer cast preserves count; `cli.zig` documents `argv` as
  `argv[1..]`, matching the call.

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

## Security findings (folded in from Reviewer3; verified by Reviewer2)

Reviewer3's full security review is in `smoke_tests/README.md`; Reviewer2 independently re-ran
all three test files against the unmodified sources — **all pass** (zig 0.17.0):

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
- No Critical/High; positives verified in negative controls (`isRepoIdLike`, `blobName`
  sanitization, sha256-before-rename, content-range start check).

Per the team-lead's rule, no P0 and no unresolved P1 security finding exists, so nothing here is
a hard blocker. The MEDIUMs are inherited from main and fixable in a follow-up: both close with
one shared `isPathLike`/`isRevisionLike` validation (reject `..`, leading `/`, NUL) between the
network JSON and every path sink in `pull.zig`/`pull_file.zig`/`hub.zig`.

### Inherited (out of branch scope, for Reviewer3)

My inherited-F1 suspicion was confirmed by Reviewer3's proven proofs; see the security section above.

## Verdict (pending Reviewer1/Reviewer3 agreement)

**Approve.** The change is small, contained, and does exactly what it claims; correctness is
proven on a real Linux+CUDA host (RTX 5090s). No blocking defect found. Nits F1–F4 are one-liners
worth folding in before merge if the others agree; the inherited `pull` path-traversal question
goes to Reviewer3.
