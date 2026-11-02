# Linux sandbox qualification review

## Current integration

The bubblewrap backend is implemented with private workspace copies and checked host publication for writable commands
and direct MCP servers. Native Ubuntu checks cover existing and new Git controls, validated linked
worktrees, private-file denial, scratch mount ordering, explicit network admission, cancellation and
MCP descendant cleanup. The earlier mount-only writable prototype is not used.

The production mount plan also drives the capability probe. System and Node roots stay read-only;
external MCP runtime and data inputs receive private read-only views. Broad home, temporary and
workspace ancestor grants are refused. New or changed Git controls cannot be published. These are
development qualification results, not a claim of identical macOS filesystem or Git semantics.
See [command and MCP isolation](../process-isolation.md) for limits and Windows qualification status.

## Historical review

The original bubblewrap backend was scoped out after the review below
because Git metadata protection and runtime roots were incomplete. The findings explain why writable
execution now uses private staging and checked publication rather than direct writable mounts.

This read-only subagent review identified issues missed during initial implementation. Prior claims that the failing probe proved a nested-container limitation were not established: its namespace lacks the ELF dynamic loader path. Merely passing argument-construction tests does not prove confinement.

## Original review findings

1. **Mount ordering:** all read-only mounts precede all writable mounts. A writable TMPDIR parent hides a read-only workspace mount below it. Build one ordered mount plan with explicit permissions, or private scratch mounts; test effective writes, not argument presence.
2. **Git-control protection:** existing top-level entries alone are insufficient. Missing controls, fresh/nested repositories, submodule metadata, and worktree pointers/common directories are not safely covered. A writable `.git` pointer can be retargeted in place. Enumeration cannot prevent future Git metadata creation in arbitrary writable directories. Choose an enforceable confinement design or reject unsupported writable execution; do not advertise SBPL parity.
3. **Capability probe:** the `/usr`-only probe omits `/lib64` (and `/lib` on some systems), causing ENOENT for dynamically linked executables. Derive probes from the production mount plan, use matching namespace flags, and preserve diagnostic reasons. Do not retry sandbox-denied tool operations through another route.
4. **Runtime visibility:** user-managed Node/Python/Rust/Nix installations remain in PATH but outside mounted roots. Add minimal explicit trusted runtime roots rather than mounting HOME wholesale.
5. **Validation:** live tests currently skip when the bad probe fails. Add a required Linux CI lane with known-working bubblewrap. Use owned outside-workspace files for denial tests, local listener positive/negative network controls, and actual Git repositories. Cover temp overlap, metadata creation/retargeting, worktrees, nested repositories, shell cancellation, usrmerge and runtime paths.

## Other corrections

- The bot VM installer baseline failure was caused by umask 0002, not a symlinked Node binary. The targeted test passed with umask 0022.
- The packaged-Mac profile replacement test should retain the old inode by renaming the fixture before recreating its path. This makes the test deterministic across filesystems without skipping it on Linux or changing production cleanup policy.
- Headless startup previously used `--no-sandbox`; that is not evidence that sandboxed desktop startup works. No sandbox-enabled GUI or clean-machine artifact qualification has been completed.
- Earlier references to PR A–D denote local branches/commits only. No GitHub pull requests or pushes have been made.
