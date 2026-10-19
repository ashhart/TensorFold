# Linux desktop port — current status

All changes are local commits on branch `linux-port` in `upstream/`. Nothing has been pushed; no GitHub pull requests exist.

## Verified on this machine (Debian 13 x86-64, headless)

| Check | Evidence |
| --- | --- |
| Dev build + typecheck | `npm run build` and `tsc --noEmit` pass |
| Full test suite | `npm test` → 821/821 passed |
| Window chrome | native Linux frame; focused window, close/draft and spacing tests pass |
| Linux packaging | `npm run pack:linux:dev` → `release/linux/linux-unpacked` |
| Artifact contents | no `resources/engine`, no `resources/computer-use`; voice/licence notices only |
| Packaged launch | `npm run smoke:linux:packaged` — isolated profile, health 200, UI 200, unauthenticated snapshot 401 |
| AppImage | `dist:linux` → `TensorFold-1.0.0-beta.8-linux-x86_64.AppImage`; launched it and got the same 200/200/401 results |
| VM installer fixture | passes with `umask 022` (environment cause) |
| Outlook fixture | passes with a workspace-extracted sqlite3 CLI |

Not verified: graphical launch on X11/Wayland, a real Mac-server streaming/tool round trip, a clean-machine install, and Mac regression for the shared changes.

## Linux command sandbox — disabled by design

`canUseLinuxSandbox()` returns false, so Linux `run_command` keeps the existing fail-closed behavior. The draft bubblewrap backend now builds one parent-first mount plan with a read-only workspace decisive beneath writable temporary roots, and probes the production argv. Still unqualified: git-control confinement for absent, nested and worktree metadata; runtime roots for user-managed toolchains; and live adversarial evidence. Details: [SANDBOX-REVIEW.md](SANDBOX-REVIEW.md).

## Deliberate scope boundaries

- No inference engine bundled or ported; providers are user-configured HTTP endpoints (remote Mac or self-managed).
- Apple-only features fail closed with their existing messages; MCP process sandboxing remains macOS-only.
- Mac signing, notarization and update trust are untouched. Linux artifacts are unsigned development builds and never publish.

## Remaining work

1. Qualify or finish the Linux command confinement design.
2. Exercise provider-only onboarding and a real streaming/tool round trip against the user's Mac endpoint.
3. Clean-machine install check for the AppImage.
4. Linux CI lane and Mac regression evidence.
