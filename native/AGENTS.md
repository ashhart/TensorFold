# Native Zig contributor instructions

- Run commands from the repository root. Follow `native/README.md` for setup.
  Use `.zig-toolchain/zig` and the stable release in `.zig-version`.
- Inspect `git status` and preserve other work. Prefer the codebase graph for
  discovery when available.
- Bootstrap with `bash scripts/fetch-zig.sh`, then
  `.zig-toolchain/zig run tools/setup_native.zig`. Use `--dry-run` to review the
  plan or `--check` to inspect an existing environment. No neighboring checkout,
  administrator access or model download is needed for setup.
- Never execute sudo or another administrator escalation mechanism. Missing
  Xcode/system prerequisites must be installed by the user.
- Ask before **new large model downloads**. Reuse checkpoints under
  `~/.models/<publisher>/<model>` and link only needed models into `build/models`.
  Do not duplicate or delete weights to make a test convenient.
- Write native code, tests and new build helpers in Zig. Keep Python changes in
  existing oracle/exporter tooling. Do not add ad hoc shell/Python wrappers.
- Format edited Zig files, build with `-Doptimize=safe -j1`, and run targeted
  checks. Baseline: `.zig-toolchain/zig build test test-checkpoint-files
  -Doptimize=safe -j1`. Arithmetic/cache changes also require affected Metal
  oracle checks. Avoid concurrent full-model loads.
- Keep MLX, MLX-C, Python references, JPEG and generated Metal aligned with
  `native/dependencies.json` and upstream requirements. Do not change one side
  to make a mismatch pass. Setup and sync share the dependency build recipe in
  `tools/sync_upstream.zig`.
- Sync main into Zig only on request, on a clean PR branch based on the latest
  TensorFold `zig`. It prepares an uncommitted merge, aligns dependencies and
  runs large tests; it never rebases or pushes. Review and submit changes through
  a PR against `ashhart/TensorFold:zig`, using a GitHub noreply commit address.
  Do not schedule sync or use it as a setup shortcut. Use SSH Git remotes.
- After every upstream merge or rebase, including manual branch syncs, handle
  native drift before committing or pushing. Verify dependency pins, regenerate
  with `.venv/bin/python tools/export_native_kernels.py`, and review the diff.
  Update native callers and fixtures when kernel inputs, templates or layouts
  change; regeneration alone does not establish runtime parity. Review source
  changes and feature bindings before `record-upstream-coverage`; never refresh
  hashes merely to silence a failure. Run the affected Metal oracles, setup
  checks and every verification step in `.github/workflows/native-macos.yml`
  through Latch. Resolve failures without disabling checks, then inspect the
  resulting GitHub CI run after the authorized push and address any failures.
- Preserve upstream arithmetic, dtype, layout, sampling positions and cache
  commit/rollback semantics. Compare intermediate arrays when output diverges;
  do not widen tolerances to conceal numerical drift.
- Schema/kernel passes do not verify a complete model. GLM's full checkpoint
  and DeepSeek's full inference are unverified on the 128 GiB development
  machine. Physical M1–M4 coverage is also unverified.
- Keep generated fixtures/traces in ignored `build/native-checks`. Report
  correctness separately from performance and identify skipped/unrun checks.
  Give concise findings in the conversation; update the existing build guide
  when needed, without adding investigation diaries.
