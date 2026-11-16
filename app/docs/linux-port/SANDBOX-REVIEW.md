# Linux sandbox qualification review

## Status

The bubblewrap backend is **implemented on this branch** and stays opt-in
(`TENSORFOLD_LINUX_SANDBOX=1`). This review is the record of what blocked it; the blockers below are
now either closed or stated as limits in [STATUS.md](STATUS.md). No product default was weakened:
without the opt-in, Linux `run_command` keeps the existing fail-closed behaviour.

The read-only subagent review identified issues missed during initial implementation. Prior claims
that the failing probe proved a nested-container limitation were not established: its namespace
lacked the ELF dynamic loader path. Passing argument-construction tests alone does not prove
confinement either; the conformance test exercises a real namespace.

## Release blockers

1. **Mount ordering (closed):** all read-only mounts precede all writable mounts. A writable TMPDIR parent hides a read-only workspace mount below it. Build one ordered mount plan with explicit permissions, or private scratch mounts; test effective writes, not argument presence.
2. **Git-control protection (closed, with stated limits):** existing top-level entries alone are insufficient. Missing controls, fresh/nested repositories, submodule metadata, and worktree pointers/common directories are not safely covered. A writable `.git` pointer can be retargeted in place. Enumeration cannot prevent future Git metadata creation in arbitrary writable directories. Choose an enforceable confinement design or reject unsupported writable execution; do not advertise SBPL parity.
3. **Capability probe (closed):** the `/usr`-only probe omitted `/lib64` (and `/lib` on some systems), causing ENOENT for dynamically linked executables. Derive probes from the production mount plan, use matching namespace flags, and preserve diagnostic reasons. Do not retry sandbox-denied tool operations through another route.
4. **Runtime visibility (open):** user-managed Node/Python/Rust/Nix installations remain in PATH but outside mounted roots. Add minimal explicit trusted runtime roots rather than mounting HOME wholesale.
5. **Validation (partly closed):** live tests skip when the probe fails; a Linux CI lane exists but does not install bubblewrap, so device-level conformance must run on a host that has it. Add a required Linux CI lane with known-working bubblewrap. Use owned outside-workspace files for denial tests, local listener positive/negative network controls, and actual Git repositories. Cover temp overlap, metadata creation/retargeting, worktrees, nested repositories, shell cancellation, usrmerge and runtime paths.

## Other corrections

- The bot VM installer baseline failure was caused by umask 0002, not a symlinked Node binary. The targeted test passed with umask 0022.
- The packaged-Mac profile replacement test should retain the old inode by renaming the fixture before recreating its path. This makes the test deterministic across filesystems without skipping it on Linux or changing production cleanup policy.
- Headless startup previously used `--no-sandbox`; that is not evidence that sandboxed desktop startup works. No sandbox-enabled GUI or clean-machine artifact qualification has been completed.
- Earlier references to PR A–D denote local branches/commits only. No GitHub pull requests or pushes have been made.
