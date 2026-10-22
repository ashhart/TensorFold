# Phase 1 Baseline — TensorFold desktop on Linux

Date: 2026-10-18 · Machine: Debian 13 (trixie) x86-64, headless (no DISPLAY/WAYLAND)
Pinned upstream commit: `d46d17a784758b8a8b66badde1f728f36bdc9541` (the contributor's fork of this repository; upstream remote: ashhart/TensorFold)

## Environment setup (reproducible)

The bring-up machine's environment was prepared by a script the operator kept outside this repository. It set:

- Node **22.19.0** (a mise install under the operator's own home) — satisfies `devEngines` `>=22 <23`
- npm **10.9.8** exactly, through a wrapper the operator kept outside this repository (mise's bundled npm is 10.9.3 and is rejected by `devEngines`; the mise install tree is read-only there)
- `npm_config_cache` and `electron_config_cache` redirected into the workspace (sandbox denies `~/.npm`, `~/.cache/electron`)
- For test runs: `HOME` and `TMPDIR` pointed at directories inside the workspace (sandbox denies `$HOME` writes; `/tmp` is a 2 GiB tmpfs which trips a 2 GiB free-space staging check)

Install sequence (worked):

```sh
cd app && npm ci
npm run deps:trusted-scripts   # Electron binary + @lydell/node-pty rebuild (gcc/g++/make present)
cd mobile && npm ci            # required; three scripts/*.test.mjs import expo/xcode via mobile/
```

## Baseline results

| Check | Result |
| --- | --- |
| `npm run install:check` | PASS |
| `npm run typecheck` | PASS |
| `npm run build` (renderer + main/preload + cluster node) | PASS |
| Headless launch `electron . --headless-server --ozone-platform=headless` | **PASS** — orchestrator listens on 127.0.0.1:8765, `/health` returns `{"ok":true}`; only benign dbus warnings |
| `npm test` (817 files) | **807 pass, 10 fail** — classified below |

The headless launch used `--no-sandbox`. No comparison run established that Chromium sandboxing could not work here. This is diagnostic startup evidence only, not sandbox-enabled desktop qualification. See [subsequent review](SANDBOX-REVIEW.md) for corrections and release blockers.

## Failure classification

### Genuine Linux port items (code)

1. **No command sandbox on Linux** — `electron/agent/sandbox.ts` `runSandboxedCommand` supports only macOS `sandbox-exec` and otherwise fails closed (`VONTRA_ALLOW_UNSANDBOXED_COMMANDS=1` bypass for development). Blocks agent `run_command` on Linux.
   - Failing: `tools.test.ts` (expects runnable sandboxed command), `sandbox.test.ts` (trap.pid never written), `room-cli-seat.test.ts` ("Planning-only mode needs the macOS sandbox").
   - Fix direction: bubblewrap backend mirroring the existing profile semantics (read roots, write roots, network toggle); bwrap is available even in this container.
2. **Hardcoded `/bin/zsh`** in `sandbox.ts` (both sandboxed and unsandboxed exec paths, and `SHELL` env). Debian stock has no zsh. `terminal-runtime.ts` already has `loginShellPath()` discovery; the sandbox path needs the same treatment.
3. **`owned-group.ts` process-identity bug** — parses `ps -o comm=` to verify benchmark group members; `comm` is a full path on macOS but a 15-char truncated name on Linux, so `path.resolve(member.command) !== executable` always fails → `cleanupFailed: true`. Failing: `engine/benchmark/process.test.ts`. Fix: resolve via `/proc/<pid>/exe` on Linux or use `args=` argv[0] handling.

### Test-portability items (tests assume macOS unconditionally)

4. `email-outlook-optin.test.ts` — Outlook probe is darwin-only by design (`outlook-connector.ts` returns null off-darwin); test does not skip.
5. `planning-sandbox.test.ts` — expected sandbox read-root list omits `/usr`, which exists on Linux; fixture assumes macOS path layout.
6. `build-computer-helper.test.mjs` — helper build throws "macOS only" before the mocked-xcrun case can run; needs platform skip or platform injection.

### Environment artifacts of this container (not code bugs)

7. `native-adapter.test.ts` — PASSES once `TMPDIR` points at the workspace (2 GiB free-space staging check vs 2 GiB tmpfs `/tmp`).
8. `outlook-connector.test.ts` — `spawn sqlite3 ENOENT`; sqlite3 CLI not installed here.
9. `bot-vm/installer.test.ts` — "Linked or writable runtime components are forbidden": the environment's umask 0002 makes fixtures group-writable. The targeted test passes with umask 0022. The earlier Node-symlink explanation was incorrect.

## Audit findings beyond the test suite

- **The codebase is already substantially platform-aware.** `cluster-node` supports `linux` (avahi discovery, systemd user services); `updates.ts` fails closed off-darwin; iMessage/Outlook/iPhone-mirroring/bot-VM all fail closed with clear messages; `app-window.ts` has a non-darwin branch; renderer platform checks include `'linux'` in places.
- **Secrets do not depend on macOS Keychain**: `electron/local-crypto.ts` is app-managed encryption (safeStorage only for legacy blob migration). Linux keyring is a non-issue.
- **Window controls IPC exists**: preload exposes `windowAction('minimize'|'maximize'|'close')`; renderer TitleBar assumes macOS traffic lights (`pl-[96px]` inset) and draws no Linux controls. Needed: renderer window-control buttons on Linux.
- **Engine-missing path is graceful**: `EngineService.binary()` returns null without the binary; `EngineChoice` shows `engine.missing` and disables model loading. Onboarding has a `setup.later` path and external-provider choices. Likely works as-is; needs a UI verification pass with a display.
- **Engine bundling is mac-only**: `build/engine.json` pins `macos-arm64`; `fetch-engine.mjs` stages it. For Linux dev, `TENSORFOLD_STUDIO_ENGINE_BIN` override exists; packaged Linux builds must simply not stage the engine (per plan: no bundled engine on Linux).
- **Packaging scripts are mac-only by design** (`package-mac.mjs` guards on darwin). A Linux packaging script is additive work, not a rewrite.
- **UI copy says "this Mac" pervasively** (dozens of strings incl. localization catalogues). Cosmetic; handle as a later polish PR, not mixed into the bring-up PR.

## Phase 1 exit assessment

- [x] Reproducible baseline: the toolchain the repository pins and the steps above
- [x] Bounded blocker list: items 1–3 (code), 4–6 (test portability)
- Conclusion: **no fundamental blocker**. The core app boots and 807/817 tests pass unmodified. The critical-path work is the Linux command sandbox (bubblewrap) + shell discovery, then chrome, then packaging.

## Next: Phase 2 entry point

First PR candidate (small, reviewable, per CONTRIBUTING.md "one change per PR"):

1. PR A: fix `owned-group.ts` Linux `comm` bug + test (bug fix, platform-neutral).
2. PR B: shell discovery in sandbox exec paths (`/bin/zsh` → resolved shell) + test-portability skips for items 4–6.
3. PR C: bubblewrap command-sandbox backend behind the existing profile abstraction.

Phase 2 (client-only startup/onboarding) appears to mostly exist already; confirm with a display (Xvfb) in Phase 3.
