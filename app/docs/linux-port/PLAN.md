# TensorFold Linux Desktop Contribution Plan

## Objective

Contribute Linux support to the existing TensorFold desktop application upstream. Run the desktop workspace and coding tools on Linux while using a TensorFold model server running on a Mac through the existing OpenAI-compatible provider interface.

This is not a new client, a permanent fork, or an inference-engine project. Only the desktop application is ported: the TensorFold engine already supports Linux, and this contribution neither bundles nor manages any engine. Inference comes from a user-configured endpoint—typically a TensorFold server on a Mac in this plan, or an engine the user runs and manages themselves on the Linux machine.

## Intended architecture

```text
Linux desktop
  Existing Electron / React application
  Chat, streaming, approvals, local files, terminal, Git
             |
             | OpenAI-compatible API, preferably through SSH
             v
Mac
  TensorFold model server
  Model loading and inference
```

**Execution boundary:** model inference runs on the Mac; file access, edits, terminal commands, and Git operations run on Linux against the workspace selected there. Connecting a model endpoint does not grant access to the Mac's files or shell. Selecting a workspace is not an OS sandbox: terminal commands run with the Linux user's permissions and may access paths outside that workspace. Preserve existing file-tool enforcement and approval semantics without claiming shell confinement.

## Alternatives considered

The desktop already ships a headless orchestrator mode (`start:headless`) with browser pairing, so maintainers may ask why a Linux desktop port is needed at all.

- **Headless server on the Mac + Linux browser:** rejected. Tool execution (file access, edits, terminal, Git) follows the server, so the Mac would touch the code—this breaks the required execution boundary where work happens on the Linux workspace.
- **Headless server on Linux + Linux browser:** rejected as the deliverable. It still requires Electron and the app's Node runtime to work on Linux (most of the porting cost), while losing native chrome, notifications, and window integration. It remains a useful fallback and a development stepping stone.
- **Full Linux desktop (chosen):** the existing Electron/React app runs on Linux, tools act on the Linux workspace, and inference is served by an external endpoint—on a Mac, or a self-managed engine anywhere.

## Scope

### Required for the initial contribution

- Launch the existing Electron desktop application on Linux.
- Configure a remote OpenAI-compatible model endpoint without requiring a bundled local engine.
- Complete onboarding without downloading or loading a local model.
- Preserve streaming chat, tool calls, approval gates, workspace file operations, terminal execution, and core Git workflows.
- Clearly hide or mark unavailable unsupported native features.
- Produce one installable Linux artifact and document prerequisites.
- Add appropriate Linux tests/CI while preserving Mac behavior.
- Document the Linux-client-to-Mac-server workflow and its security boundary.

### Explicitly out of scope

- Bundled Linux inference engine, CUDA/ROCm integration, GPU discovery, or VRAM-based recommendations.
- Managed local model downloading/loading and local Living Weights on Linux.
- Linux implementations of Mac desktop automation, Apple integrations, or isolated bot VMs.
- Phone control, even though its Android path (adb over USB/wireless) is potentially portable to Linux. It is deferred, not rejected: feature gates should be capability-specific rather than a blanket `darwin` check, so it can be enabled later without unwinding platform assumptions. iPhone control stays Mac-only.
- Full feature parity across every Linux distribution and desktop environment.
- A new transport protocol or remote execution service.
- Automatic SSH tunnel management unless upstream explicitly requests it later.
- Broad UI refactoring or unrelated engine work.

External endpoints may still be on the same Linux machine, but this contribution does not manage those servers.

## Initial source-review findings

The desktop is Electron, React, TypeScript, Vite, Tailwind, and Zustand. It already supports OpenAI-compatible providers, making reuse of the current client the preferred path.

Observed concrete platform assumptions:

- The engine manifest selects a Mac ARM64 archive and a `.dylib`.
- Packaging scripts and release validation are Mac-oriented, including signing/notarization.
- Packaging has a Darwin/ARM64-specific ONNX voice-runtime unpack rule.
- The computer-use helper compiles Swift against Apple frameworks and explicitly refuses non-Mac builds.
- Desktop Living Weights support is gated to Darwin/ARM64.
- Engine service hardware/memory logic and messages contain Mac assumptions.

These are preliminary observations, not a complete platform audit. No Linux build or runtime test has been performed. Pin the implementation to an upstream commit before making changes; the repository can move rapidly.

## Contribution strategy

1. Open an upstream issue before substantial implementation.
2. Describe the narrow scope: Linux desktop with external inference; no bundled engine or Mac-only native integrations.
3. Ask maintainers to confirm the supported distro/architecture, preferred artifact, CI expectations, and whether a staged PR sequence is acceptable.
4. Keep changes in the existing application with small platform/capability boundaries; avoid copying the app into a separate Linux project.
5. Follow the repository's current contribution, agent, security, and testing instructions before writing source changes.
6. Submit reviewable commits with explicit Linux test evidence and Mac regression evidence. Do not claim Mac runtime validation without access to a Mac or maintainer-provided results.

Suggested issue title: **Linux desktop support using external OpenAI-compatible inference**.

## Implementation phases

### Phase 1 — Establish a baseline and inventory blockers

- Clone upstream into a separate checkout under this project folder after checking for existing files.
- Record the exact commit, Node/npm versions, Electron version, and Linux environment.
- Read current repository and app contribution instructions, including the pinned `.zig-version` and the contributor skill at `docs/recipes/SKILL.md`.
- Note the hard toolchain gate: `devEngines` requires Node >=22 <23 and exactly npm 10.9.8 with `onFail: error`, plus a `preinstall` package-manager guard. Native deps (`@lydell/node-pty`, `ssh2`) need Linux rebuilds, and `file:` dependencies under `mobile/vendor/` require a complete checkout. Install using the prescribed steps; do not bypass the repository's disabled lifecycle-script policy.
- Run baseline type checks, tests, renderer/main build, and a Linux Electron launch.
- Classify pre-existing failures separately from Linux portability failures.
- Audit startup, imports, packaging, and feature registration for:
  - Unconditional Mac commands or native helper initialization.
  - `process.platform`, architecture checks, Apple paths, and Mac-only dependencies.
  - Window controls, titlebar, keyboard modifiers, menu/tray behavior, and application lifecycle.
  - Provider credentials, OS keyring availability, and storage behavior.
  - Shell discovery, terminal native modules, process groups, cancellation, and GNU/BSD command differences.
  - Any onboarding route that requires a local model before an external provider can be used.
- Produce a blocker checklist separating required changes from optional features to disable.

**Exit criterion:** reproducible baseline and a bounded list of blockers for the external-provider Linux path.

### Phase 2 — Enable client-only startup and onboarding

- Treat bundled local-engine availability separately from desktop/provider availability.
- Prefer existing feature-availability mechanisms. Add a small shared capability layer only where needed; avoid scattered ad hoc platform checks.
- Ensure missing bundled engine resources do not cause app startup, settings, or provider selection to fail.
- Make external-provider setup an accessible onboarding path on Linux.
- Ensure initial launch never silently downloads or starts a model.
- Do not automatically initialize Apple helpers, VM components, or unavailable integrations on Linux.
- Hide or explain unsupported controls consistently. Preserve access to configured external providers.
- Enforce unavailable capabilities in backend IPC/tool handlers, not just the UI. Direct invocation and persisted settings must fail closed without spawning helpers or triggering downloads. Add regression tests and verify Mac capability paths remain unchanged.
- Keep voice disabled with an explanation if it cannot be safely supported within scope; do not make voice a launch prerequisite.
- Preserve existing Mac onboarding and managed local-engine behavior.

**Exit criterion:** a fresh Linux profile reaches a configured remote provider without any local model or engine.

### Phase 3 — Validate core Linux workspace behavior

- Verify streamed content, reasoning if supported, tool-call argument accumulation, error handling, and cancellation against the Mac server.
- Validate provider model listing and manual model selection according to existing app behavior.
- Verify read/list/search, writes, edits, diffs, approvals, and read-only mode in a disposable Linux workspace.
- Verify terminal startup, working directory, interactive PTY behavior, exit status, Stop, and child-process cleanup.
- Test paths with spaces, Unicode filenames, symlinks, executable permissions, and case-sensitive filesystem behavior.
- Verify core Git operations and worktree-based workflows where supported by the existing provider route; do not expand into a new remote-worker implementation.
- Provide Linux window controls and appropriate shortcuts without disturbing Mac chrome. If full chrome proves costly, the app's existing tray-based Minimal mode is an acceptable reduced MVP surface—confirm with maintainers before narrowing.
- Check close-versus-quit behavior and tray behavior where enabled; do not make the tray essential on desktops without it.
- Agree upstream on credential-storage behavior with and without a Linux secret service. No silent plaintext or Electron `basic_text` fallback: require an explicit error or informed consent under upstream policy. Verify saved provider settings and secrets after restart and ensure logs never expose credentials.
- Test file-tool path traversal and symlink handling against existing enforcement; document that terminal commands are not confined by workspace selection.
- Record the Mac server version, model/revision, and a repeatable tool-call smoke scenario. Add mock SSE tests for truncated streams, malformed tool arguments, and connection loss.

**Exit criterion:** an approved model-driven edit and terminal/test run complete on Linux, with working Stop and no orphaned managed processes.

### Phase 4 — Add minimal Linux packaging

- Agree on the first artifact with maintainers; prefer one format initially (AppImage or DEB), rather than promising every format.
- Add an explicit Linux packaging command using the existing build stack.
- Exclude Mac engine assets and Apple native helpers from the Linux artifact.
- Scope Apple signing/notarization requirements to Mac release paths without weakening them.
- Verify native module compatibility and packaging hooks on Linux, including hooks originally written for Mac. Two specific landmines: `forceCodeSigning: true` must not break unsigned Linux builds (scope it to Mac if needed), and the `darwin/arm64` ONNX `asarUnpack` rule must be revisited if voice is ever enabled on Linux (kokoro-js/onnxruntime-node Linux binaries).
- Add Linux icons/desktop metadata as needed, executable permissions, and license notices.
- Preserve Electron security settings; do not solve launch failures with `--no-sandbox` or by disabling security fuses indiscriminately.
- Test launch from the installed artifact rather than only the development server.
- Explicitly disable incompatible update feed/install paths on Linux. For the first release, a manual-update path is acceptable if agreed upstream; document trusted release sources and checksum/signature verification according to upstream policy without weakening Mac release trust.
- Record the exact target distro release and architecture. Test the installed artifact as a non-root user in a clean VM with no development checkout, build-time node_modules, or system Node required. Check artifact architecture, runtime dependencies, and functioning Electron sandbox.

**Exit criterion:** a clean supported Linux environment installs/launches the artifact and completes the remote-inference workflow.

### Phase 5 — CI, documentation, and upstream review

- Add Linux build/typecheck/test coverage consistent with current CI conventions.
- Add tests for platform capability resolution and startup without bundled engine assets.
- Add onboarding coverage for external providers with unavailable local inference.
- Add packaging smoke checks where CI supports a display or virtual display; distinguish UI launch from mocked/unit coverage.
- Keep existing Mac CI gates intact and obtain Mac regression validation.
- Document supported environments, artifact installation, provider setup, SSH tunneling, known limitations, and execution boundaries.
- Include screenshots and a concise end-to-end test receipt in the PR.

**Exit criterion:** maintainers can reproduce the Linux flow, required CI passes, and the patch is ready for upstream review. Merge is a maintainer decision, not a guaranteed project outcome.

## Mac connection plan

### Preferred first setup

Run a separately managed TensorFold server on the Mac with a stable loopback port and authentication, then forward it to Linux over SSH. Keep the server bound to loopback rather than exposing an unauthenticated LAN endpoint.

Example tunnel from Linux, with illustrative port values:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18080:127.0.0.1:8080 user@mac-host
```

Configure the desktop provider with a base URL such as `http://127.0.0.1:18080/v1`, the server's API key, and an advertised model ID. Confirm the current provider's base-URL convention before documenting final steps.

- Verify SSH host identity normally; do not disable host-key checking.
- Transfer/store API credentials deliberately; never commit keys or paste them into PR logs.
- Do not expose the API publicly as part of this work.
- Treat model output and tool arguments as untrusted; existing approval policies must still apply.
- Test dropped tunnels, stale credentials, unavailable models, and server restarts.

The Mac desktop's managed endpoint is an alternative, but its port and key change when the engine restarts. Document how to copy fresh connection details and recreate the tunnel if that mode is used. Do not promise stable managed endpoint coordinates.

## Acceptance checklist

- [ ] Maintainers have been asked to confirm scope and target environment.
- [ ] Exact upstream commit and baseline results are recorded.
- [ ] Linux launches with no bundled engine and no Apple helper binaries.
- [ ] Fresh onboarding permits an external provider without local model setup.
- [ ] Authenticated Mac connection works through an SSH tunnel.
- [ ] Model listing/selection and streamed chat work.
- [ ] File-tool restrictions and approval semantics are preserved; traversal/symlink tests pass and shell confinement is not promised.
- [ ] Approve-edits, read-only, rejection, and cancellation behave correctly.
- [ ] Linux terminal, Git operations, and process cleanup are validated.
- [ ] Unsupported features are unavailable in both UI and backend, including direct IPC calls and restored settings; no helper spawn or download occurs.
- [ ] Credential fallback behavior is explicitly approved, restart persistence is tested, and secrets do not leak into logs.
- [ ] Packaged artifact passes non-root clean-VM tests on a recorded distro/release/architecture without development dependencies or system Node, with Electron sandbox functioning.
- [ ] Incompatible updates are disabled and trusted manual artifact verification is documented.
- [ ] Existing Mac behavior/security gates are preserved and regression checks are recorded.
- [ ] CI, documentation, known limitations, and PR test evidence are complete.

## Test matrix

Start narrow, then broaden only as agreed upstream.

| Dimension | Initial coverage |
| --- | --- |
| Architecture | Linux x86-64 first; ARM64 only if agreed and tested |
| Distribution | One maintainer-approved supported Ubuntu LTS baseline |
| Display | Test both X11 and Wayland where available; record any gap |
| Desktop | One primary desktop; tray absence must not block core use |
| Inference | Mac TensorFold server over authenticated SSH tunnel |
| Profiles | Fresh profile and restart with saved external provider |
| Network faults | Tunnel absent/dropped, server stopped/restarted, bad API key |
| Stream faults | Interruption mid-tool-call against the real Mac server, plus mock SSE fault suite |
| Workspace | Disposable Git repository with spaces, Unicode, and symlinks |
| Launch mode | Development and packaged artifact |
| Mac regression | Existing tests plus Mac runtime validation when available |

## Suggested PR sequence

1. **Platform/client-only foundations:** capability guards, safe Linux startup, external-provider onboarding, focused tests.
2. **Linux desktop integration and packaging:** chrome/shortcuts, artifact generation, packaging smoke checks, Linux CI.
3. **Documentation and qualification:** Mac-server connection guide, environment receipts, known limitations, regression fixes.

Combine these if maintainers prefer one PR, but retain reviewable commits. Each intermediate PR must be safe on Mac and include minimal setup documentation and regression tests for the Linux functionality it enables. Xvfb smoke coverage is not Wayland runtime validation; record these separately. Revisit the effort estimate after the baseline audit.

## Effort and risks

Planning estimate for one experienced Electron/Linux developer:

- Baseline and working client-only prototype: roughly 2–5 working days.
- Upstream-ready initial contribution: roughly 1–3 weeks total if portability issues are modest.
- Allow additional time for maintainer feedback, Mac validation, and native module or packaging problems (roughly ±1 week if the Node/npm `devEngines` gate or native rebuilds fight the environment).

Main risks are mandatory local-engine onboarding, unconditional Mac helper startup, the pinned Node >=22 <23 / npm 10.9.8 toolchain gate (which also constrains CI image selection), native dependency packaging, Linux keyring behavior, Electron sandbox requirements, update assumptions, and worker flows coupled to the managed local engine. If a worker feature is engine-specific, document it as unavailable rather than silently redirecting it or enlarging the scope.

## Next concrete actions

1. Confirm which Linux distribution/architecture should be the first target.
2. Draft/open the upstream scope issue.
3. Clone and pin upstream, read contribution instructions, and record baseline results.
4. Implement the smallest external-provider Linux path before spending time on release polish.

## Reference sources

Reviewed during planning; re-check at the pinned implementation commit:

- [Repository and Linux engine overview](https://github.com/ashhart/TensorFold)
- [Desktop README and architecture](https://github.com/ashhart/TensorFold/blob/main/app/README.md)
- [Desktop package/build configuration](https://github.com/ashhart/TensorFold/blob/main/app/package.json)
- [Bundled engine manifest](https://github.com/ashhart/TensorFold/blob/main/app/build/engine.json)
- [Engine path resolution](https://github.com/ashhart/TensorFold/blob/main/app/electron/agent/engine/paths.ts)
- [Engine service and platform gates](https://github.com/ashhart/TensorFold/blob/main/app/electron/agent/engine/service.ts)
- [Apple computer-use helper build](https://github.com/ashhart/TensorFold/blob/main/app/scripts/build-computer-helper.mjs)
