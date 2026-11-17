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
| Full test suite | `npm test` → 845/845 passed (after merging `upstream/main`) |
| Window chrome | native Linux frame; window, close/draft and spacing tests pass |
| Linux packaging | `npm run pack:linux:dev` → `release/linux/linux-unpacked` |
| Artifact contents | no `resources/engine`, no `resources/computer-use`; notices only |
| Packaged launch | `npm run smoke:linux:packaged` — isolated profile, health 200, UI 200, unauthenticated snapshot 401 |
| AppImage | `dist:linux` → `TensorFold-1.0.0-beta.8-linux-x86_64.AppImage`, same 200/200/401 when launched |
| Command sandbox | conformance test passes: filesystem containment, git-control pinning, network denial/allowed profile, prompt cancellation |
| VM installer fixture | passes with `umask 022` (environment cause) |
| Outlook fixture | passes with a workspace-extracted sqlite3 CLI |

Not verified by this branch's author: a real Mac-server streaming/tool round trip (the user
confirmed a remote TensorFold engine and provider detection working on Fedora 42), and a
clean-machine install.

## Fixed from user testing on Fedora 42 (Wayland)

- A startup crash (`React #310`) left the window on the boot splash in every Linux build. `App`
  declared `useEngine()` below its loading-state return, so the hook count changed when settings
  arrived. The hook moved above the return and `App.hooks-order.test.ts` now fails if any hook is
  declared after that return.
- A frozen boot is no longer silent: renderer errors, unhandled rejections and a ten-second stall
  appear in a visible panel, and `did-fail-load` / `render-process-gone` report to the terminal and
  a dialog.
- Wayland cannot present Vulkan surfaces; the app now merges `Vulkan` into `--disable-features` when
  a Wayland session is detected, leaving X11 and macOS untouched.
- LAN providers stay **external**, as the maintainer asked: the TensorFold network preset is only a
  convenience for adding the endpoint, and bots/phone guard do not run against it.
- Thread and bot chrome no longer says "this Mac" on Linux; a trusted network engine is labelled
  "Network" rather than "On this Mac".

## Known product limits (upstream design, not platform bugs)

- Bot chats are `kind: 'chat'`, and chat mode has no `read_file`/`run_command` tools, so a bot cannot
  read the workspace on any platform. Attachments are the supported way to give a bot file content;
  workspace work belongs to a task or project chat.

## Linux command sandbox — follow-up branch

The bubblewrap backend is implemented on this branch and stays **opt-in**
(`TENSORFOLD_LINUX_SANDBOX=1`) until a maintainer decides otherwise. It pins every repository it
finds under the workspace — including repositories inside dependency folders and nested worktrees —
and refuses instead of protecting only some.

Limits, stated rather than hidden:

1. A workspace with no repository is refused: bwrap mounts paths and cannot deny creating `.git` by
   name, so the sandbox requires metadata it can pin.
2. The walk refuses at 32 directory levels or 50,000 directories rather than silently pinning a
   subset, and it refuses when it cannot inspect a path (permission denied, I/O error) instead of
   skipping a directory that could hold a repository. A workspace that trips any of these keeps
   commands refused.
3. Control entries a repository does not have yet (`config.worktree`, `hooks`, `info`) are mounted
   read-only from empty placeholders, which bwrap creates as empty files or directories inside
   `.git` on first use. A writable run therefore cannot add or edit them.
4. A worktree's `.git` pointer file is mounted read-only and its gitdir and shared directory must
   stay inside the workspace; anything else is refused before a command runs.

The development-only `VONTRA_ALLOW_UNSANDBOXED_COMMANDS=1` escape hatch is unchanged. Details and
review history: [SANDBOX-REVIEW.md](SANDBOX-REVIEW.md).

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
