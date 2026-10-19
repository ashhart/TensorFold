# Linux desktop port — current status

Work lives on branch `linux-port` in the fork and is pushed there. All checks below were re-run after
rebasing onto current `upstream/main`.

## CI evidence (GitHub Actions, run 39265667644)

| Job | Result |
| --- | --- |
| `desktop` (macOS 14) | success — Mac regression evidence for the shared changes |
| `desktop-linux` (Ubuntu 24.04) | success — the Linux desktop lane |
| `mobile` | success |

The Linux lane installs bubblewrap, restores both lock files, runs the suite, runs the Linux
command-sandbox conformance test, typechecks and checks patch whitespace.

## Local evidence (Debian 13 x86-64, headless)

| Check | Evidence |
| --- | --- |
| Dev build + typecheck | `npm run build` and `tsc --noEmit` pass |
| Full test suite | `npm test` → 826/826 passed |
| Window chrome | native Linux frame; window, close/draft and spacing tests pass |
| Linux packaging | `npm run pack:linux:dev` → `release/linux/linux-unpacked` |
| Artifact contents | no `resources/engine`, no `resources/computer-use`; notices only |
| Packaged launch | `npm run smoke:linux:packaged` — isolated profile, health 200, UI 200, unauthenticated snapshot 401 |
| AppImage | `dist:linux` → `TensorFold-1.0.0-beta.8-linux-x86_64.AppImage`, same 200/200/401 when launched |
| Command sandbox | conformance test passes: filesystem containment, git-control pinning, network denial/allowed profile, prompt cancellation |
| VM installer fixture | passes with `umask 022` (environment cause) |
| Outlook fixture | passes with a workspace-extracted sqlite3 CLI |

Not verified: graphical launch on X11/Wayland, a real Mac-server streaming/tool round trip, and a
clean-machine install.

## Linux command sandbox

The bubblewrap backend is functional and conformance-tested, but it stays **opt-in**
(`TENSORFOLD_LINUX_SANDBOX=1`) rather than defaulting on, because three deviations from the macOS
sandbox profile remain:

1. Creating a brand-new `.git` where none exists cannot be denied; existing repositories pin `.git`
   against rename and pin `config`, `config.worktree`, `hooks` and `info` read-only.
2. Nested repositories and submodule gitdirs below the workspace root are not pinned.
3. A worktree `.git` pointer file is mounted writable, so its external gitdir is not protected.

Without the opt-in, Linux `run_command` keeps the existing fail-closed behavior; the
development-only `VONTRA_ALLOW_UNSANDBOXED_COMMANDS=1` escape hatch is unchanged. Details and review
history: [SANDBOX-REVIEW.md](SANDBOX-REVIEW.md).

## Connecting to a Mac engine

See [REMOTE-ENGINE.md](REMOTE-ENGINE.md) for the supported setup: serve with `--host` and an
`--api-key-file`, verify `/v1/models` from Linux, then add it as a local OpenAI-compatible provider.
The built-in engine entry stays unavailable on Linux by design.

## Deliberate scope boundaries

- No inference engine bundled or ported; providers are user-configured HTTP endpoints (remote Mac or self-managed).
- Apple-only features fail closed with their existing messages; MCP process sandboxing remains macOS-only.
- Mac signing, notarization and update trust are untouched. Linux artifacts are unsigned development builds and never publish.

## Remaining work

1. Decide whether the three sandbox deviations are acceptable to default on, or close them.
2. Exercise provider-only onboarding and a real streaming/tool round trip against a Mac endpoint.
3. Clean-machine install check for the AppImage.
