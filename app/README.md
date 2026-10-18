# TensorFold

An agentic AI coding app for the Mac, with the TensorFold engine built in. It also works with
any OpenAI‑compatible LLM endpoint, local or cloud. The look comes from the TensorFold brand:
graphite and navy surfaces, an electric-blue primary and the fold gradient.

It reads, edits, and runs your code through a tool‑use loop, with streaming responses,
live diffs, and per‑action approval.

The Mac app requires Apple silicon and macOS 13.3 or later. The bundled offline voice runtime sets this
minimum; isolated bot VMs additionally require macOS 26 or later.



## Features

- **Bring your own model.** Any OpenAI‑compatible `/chat/completions` endpoint works:
  - **Local** — oMLX, Ollama (`http://127.0.0.1:11434/v1`), LM Studio (`http://127.0.0.1:1234/v1`), vLLM, llama.cpp.
  - **Cloud** — OpenAI, OpenRouter, z.ai GLM, or any compatible provider. Configure as many as you like.
  - **Rooms** — pin each room seat to a different provider/model so local and cloud models can collaborate on the same task.
- **Full agentic loop.** The model can `read_file`, `write_file`, `edit_file`, `list_dir`,
  `search`, and `run_command` inside a workspace folder you choose.
- **Inline visualizations.** Chat and Code can render structured `visualize` widgets for flow
  diagrams, bar comparisons, metric cards, and timelines. CLI-style providers can use
  `vontra-widget` JSON fences for the same renderer.
- **Approval gates.** Three modes: *Approve edits* (reads run freely, writes/commands ask),
  *Full auto*, and *Read only*. Reasoning, streaming, tool cards, and unified diffs all render live.
- **Native chrome.** Frameless window with inset traffic lights, draggable titlebar, a task
  sidebar grouped by time, and a project‑grouped home launcher.
- **Local‑first.** Settings and conversations are stored only on your machine.
- **Local AI workspace.** Open Models for the installed library, downloads, Workers and the local API, with engine state, memory requirements and context visible together. Chip and memory details guide a recommendation after compatibility checks; download it, then Start chatting loads it; first-run setup finishes only after the welcome guide. Decode/prefill speed and first-token time show the latest completed TensorFold request, using reported engine timings or measured stream usage; unknown values stay blank.
- **Independent local workers.** Choose a local Git project and the loaded TensorFold model, then give each task a verified worktree and branch. Workers remain visible while you use the app, with approval, Stop, recovery and Changes review actions; a document-task preset supports repository documentation work.
- **Use the local API.** Loading a model starts its authenticated loopback endpoint. The API tab copies the base URL or an authenticated streaming curl example on request; the port and key change when the engine restarts, and the key is excluded from normal snapshots and events.

## Run it

```bash
npm ci                         # npm 10.9.8; lifecycle scripts stay disabled by policy
npm run deps:trusted-scripts  # explicitly install Electron and rebuild node-pty
npm run dev                    # starts Vite + launches the Electron app
```

Use `npm run install:verify-clean` to validate the lock with a real isolated `npm ci`, the
package-scoped Electron install/node-pty rebuild, and a native import smoke check without touching the
current `node_modules`. If `install:check` reports pnpm or a
self-referential/hybrid install, move/remove that `node_modules` and run `npm ci`.
After recreating the tree, rerun `npm run deps:trusted-scripts` before starting TensorFold; the
project-level `ignore-scripts=true` intentionally prevents `npm ci` from installing native
Electron/node-pty runtime pieces by itself.

Then open **Settings → Providers**, paste an API key (or point at a local endpoint), pick a model
in the composer, choose a workspace folder, and start a task.

### Browser demo mode

Open `http://localhost:5173` in a browser (no Electron) and TensorFold runs in **demo mode** with
sample data and a simulated agent — handy for previewing the UI. Real tool execution only happens
in the desktop app.

### Headless Orchestrator

```bash
npm run start:headless
```

Headless startup builds the app, enables the remote-control server, persists a strong token, and
binds `127.0.0.1` by default. The first successful bootstrap prints the host token and a ten-minute
browser pairing link once; later starts never reprint secrets. For a remote host, use the printed
SSH tunnel command and open the loopback pairing URL locally.

If the bootstrap credential is lost, explicitly rotate it (and revoke existing paired sessions):

```bash
VONTRA_HEADLESS_BOOTSTRAP=1 npm run start:headless
```

`VONTRA_HEADLESS_PORT` changes the port. The built-in server is always loopback-only:
`VONTRA_HEADLESS_BIND_HOST` accepts only `127.0.0.1`, and any other value fails startup. For
external access, keep TensorFold on loopback and expose it through the SSH tunnel or a separate TLS
reverse proxy you control; do not bind TensorFold itself to `0.0.0.0`.

### Verify

```bash
npm test          # guarded agent/runtime suite
npm run typecheck # TypeScript check without emitting files
npm run verify    # non-build preflight: tests + typecheck + git diff checks
```

`npm test` includes guards that fail when a `*.test.ts`, `*.test.tsx`, or `*.test.mjs` file is not
wired into the main test suite, or when source/docs contain non-placeholder email addresses.

### Phone control from chat

Open **Phone** beside the model picker in a chat, refresh devices and connect your phone.
Android works through installed Android Platform Tools over USB or paired Wireless debugging;
iPhone uses Apple's Wi-Fi Mirroring or your own signed WebDriverAgent over a loopback USB
forward or private Wi-Fi endpoint. TensorFold does not bundle, copy or install phone-harness.

Choose **Allow this chat** to give one local conversation access for up to 15 minutes.
That conversation then stays on local models with Cortheon and public web research off,
including after disconnection, so derived phone text cannot silently move to a cloud route.
Screen previews stay in memory and are captured only on request; tool output contains bounded
accessible/OCR labels and bounds rather than screenshot images. Device/app text is untrusted.
Every model input action asks for approval, checks the complete live screen before input,
and reads the result; Stop cancels work and clears its grant and preview. Chat control pauses
after 40 dispatched actions or three unchanged results. Protected, unlabelled or committing
steps need manual takeover, and Android typing supports printable ASCII only.

See [phone connection and privacy details](docs/phone-control.md) for platform prerequisites,
known limits and the distinction between implemented adapters and physical-device qualification.

### First launch and desktop modes

First-run setup offers **Minimal** or **Full**, then model setup and a short guide.
Minimal has a native menu-bar icon, server controls, downloads and authenticated localhost API
connection details; Full opens Chat, Code and Assistant. Switch in **Settings → App & onboarding**
or reopen the guide there. Setup does not download or start a model without a deliberate action.
Closing the Minimal window keeps the server running; Quit stops it, including a safe wait for
an active learning checkpoint save. Relaunching Minimal restores its menu-bar controls without
automatically allocating model memory.

**Settings → Model folders** optionally scans the configured and default Hugging Face cache or
folders chosen in the native picker. Scans stay local, are bounded and cancellable, and check
architecture, quantization, required shards and memory before enabling **Load**. Supported models
load from their existing location. Removing a search folder unlinks its models without deleting
files or interrupting an already running API; moved folders must be selected again. Checkpoints
borrowed from another cache are read-only in TensorFold, and unsupported formats remain labelled.
The API connection page copies the URL, generated session key or authenticated example only on
request; restart the model and copy fresh details when the port and key change.

### Local engine modes and voice

With the built-in TensorFold model selected, **AI settings** beside the chat model picker opens
the Living Weights controls and Cortheon's status. Cortheon web research remains disabled
in this beta, including previously saved On choices and old API enable calls. Local chat and API
serving remain available; independently configured servers and the standalone CLI keep their
own settings. Changing a supported engine mode restarts a loaded model and is blocked while
managed local work or a learning checkpoint save is active.

**Settings → Voice → Kokoro** downloads a pinned, checksum-verified 92 MB speech model;
subsequent playback runs offline in a managed native CPU worker. This version offers American
English voices, preview/Stop, retry/cancel and removal of the managed download. No API key,
Python installation or separately running speech server is required. Its bundled English
phonemizer uses the MIT HeadTTS implementation and BSD CMU dictionary; it does not ship or run
the upstream eSpeak phonemizer, and required notices ship with the app.

**Local AI → Learning** shows the experimental Living Weights learner and its engine-reported
fact graph, recall checks and neuron/layer details. Only the exact Nemotron 3.5 Lightning MLX
4-bit checkpoint on Apple Silicon is supported. Enabling it prepares an independent private
copy; lessons are submitted explicitly, never taken automatically from chat history. Learned
facts persist when teaching is turned off and across restarts; there is no per-fact undo.
The shared original checkpoint stays untouched. The private checkpoint and graph are local
files without app-level encryption. API credential sharing pauses while learning is enabled
so external clients cannot bypass the managed checkpoint-save protections; turn learning off
to share the ordinary chat API again. A graceful Quit waits for any lesson to stop and save.
The owner-confirmed **Erase local memory** operation also removes the complete private learning
copy, graph and lesson history, while preserving the original downloaded checkpoint.

**Plugins** separates searchable built-in tools, installed registrations, connections and
advanced MCP configuration. Worker-required tools and configured trust records are labelled
explicitly; configuring a connection opens its real setup flow. Registered plugins still use
the existing execution/trust policy, and configured permissions are not a sandbox guarantee.

### Build

```bash
npm run build     # builds renderer (dist/) + electron main & preload (dist-electron/)
npm run pack:mac:dev # explicit unsigned local .app; never distribute it
npm run smoke:mac:packaged # isolated loopback smoke for the current packaged .app
npm run pack:mac     # Developer ID signed + notarized unpacked release app
npm run dist:mac     # Developer ID signed + notarized dmg + zip release artifacts
```

The pack scripts first run `scripts/fetch-engine.mjs`, which stages the TensorFold engine pinned in
`build/engine.json` (1.0.12, macos-arm64) into `build/engine/`. It takes the archive from
`TENSORFOLD_ENGINE_ARCHIVE` when set, otherwise from the pinned release URL (cached in
`build/.engine-cache/` after the first download). Anything whose sha256 or `SHA256SUMS` does not match
is refused. The app ships it at `Contents/Resources/engine/{bin,lib}`;
packaging checks that `--version` prints the pinned version, and release mode also checks each engine
Mach-O for the pinned Developer ID signer, team and hardened runtime. For `npm run dev`, run
`npm run engine:fetch` once; `TENSORFOLD_STUDIO_ENGINE_BIN` points a dev build at another binary and is
ignored by packaged apps. Models download into the shared Hugging Face cache (`HF_HUB_CACHE`, else
`HF_HOME/hub`, else `~/.cache/huggingface/hub`), the same folder the engine's `pull` uses.

Release packaging requires `CSC_NAME` and `APPLE_TEAM_ID` to exactly match the production-ready
identity pinned in `build/release-trust.json`, plus either a pre-stored notarytool keychain profile
(`APPLE_KEYCHAIN_PROFILE`, optionally `APPLE_KEYCHAIN`) or the App Store Connect API-key credential
set. The repository pins the production public update-verification key and Developer ID signer;
private signing keys and notarization credentials must stay outside the repository. Raw Apple
ID/app-specific-password mode is refused so the password cannot leak through process arguments or
command-failure logs. The wrapper verifies the unpacked app and the app independently extracted from
each DMG/ZIP, including exact signer, team, bundle metadata, notarization, and CDHash equality.
Release update manifests require `--private-key`; only `release:manifest:dev` permits an explicitly
unsigned local fixture.

After packaging, `npm run smoke:mac:packaged` launches the current-architecture app headlessly with
an internally generated mode-0700 temporary `VONTRA_HOME`. The command accepts no profile-path
override, suppresses one-time bootstrap credentials, checks only a newly selected loopback port
(`health 200`, nonempty UI `200`, and unauthenticated snapshot `401`), then terminates the app,
verifies the listener closed, and removes only that exact temporary profile. It never repurposes
`HOME` for isolation and must not be replaced with a hand-written `HOME=...` smoke command.
Before Electron becomes ready, that `VONTRA_HOME` also redirects Electron `appData`, `userData`,
`sessionData` (including Chromium cookies and cache), temporary files, logs, and crash dumps into
private subdirectories of the disposable root. With isolation active, legacy application-data
migration is skipped without looking up or copying from the real macOS app-data directory.

Security release and operations controls are tracked in
[`docs/security-operations.md`](docs/security-operations.md).

## Upgrade compatibility

The app, bundle identifier, package and release artifacts now use TensorFold. Existing
Electron data is migrated from the previous app names without overwriting current files. The
primary preload bridge is `window.tensorfoldStudio`; `window.vontra` remains a
compatibility alias. The existing `~/.vontra` profile, `VONTRA_*` environment controls, local
storage keys and wire-format names remain stable so installed integrations, encrypted data and
saved sessions continue to work. These are compatibility identifiers rather than displayed
product names. The signed update feed is `https://tensorfold.dev/studio/update.json`; publishing
and verifying that feed remains a separate release prerequisite.

## Architecture

```
electron/
  main.ts            Window, IPC, run orchestration (abort + approval plumbing)
  preload.ts         contextBridge → window.tensorfoldStudio
  store.ts           Settings + threads persistence (userData JSON)
  agent/
    openai.ts        Streaming OpenAI-compatible client (SSE, tool-call accumulation)
    tools.ts         Sandboxed fs + shell tools, OpenAI function schemas
    loop.ts          The agentic loop: stream → tool calls → approval → execute → repeat
    diff.ts          LCS line diff → unified hunks
    types.ts         Shared types (also used type-only by the renderer)
src/
  styles/index.css   The TensorFold design system (Tailwind v4 @theme)
  lib/               store (zustand), ipc bridge, icons, formatting, browser mock
  components/        Sidebar, TitleBar, Home, Composer, Thread, timeline, visualizations, Settings, ui
```

**Stack:** Electron · React + TypeScript · Vite · Tailwind CSS v4 · Zustand.
