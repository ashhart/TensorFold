# TensorFold Orchestrator Mode

Orchestrator mode is the always-on control path for TensorFold.

The same app can run in two roles:

- **Local workstation**: work directly on this Mac, with local folders, local providers, and local tools.
- **Orchestrator host**: keep rooms, missions, queues, connectors, and overnight jobs alive on an always-on machine.

That means a MacBook can keep doing local work while also controlling the TensorFold Orchestrator on a Mac Studio.

## Product Shape

- Every installed TensorFold app can work locally.
- Any installed TensorFold app can enable Orchestrator hosting.
- A signed-in app should eventually discover all trusted Orchestrators attached to the account.
- Each workspace can choose a run location: `Local` or a named `Orchestrator`.
- Pairing a remote app saves a named Orchestrator profile in `settings.remoteOrchestrators`.
- The Orchestrator keeps authority over its files, shells, worktrees, credentials, rooms, missions, and provider sessions.
- The client app sends actions and subscribes to state/events. It does not receive raw secrets or unrestricted shell access.

## Network Modes

### Self-hosted direct

Best for personal use and trusted networks.

- The Orchestrator exposes a small HTTP/SSE bridge.
- Default bind is `127.0.0.1`.
- LAN bind must be explicitly enabled.
- Use Tailscale, WireGuard, or another trusted private network for cross-machine access.
- Every non-health endpoint requires the pairing bearer token.

### Hosted relay

Best when direct networking is awkward.

- TensorFold Cloud brokers an authenticated event/action stream.
- The Orchestrator keeps local secrets and executes actions locally.
- The relay should store only routing metadata, redacted snapshots, and delivery logs.
- This unlocks sign-in discovery, multi-device presence, phone control, and account-level Orchestrator lists.

### Hosted Orchestrator

Optional commercial path.

- TensorFold can run the Orchestrator worker for users who do not have an always-on Mac.
- This should use isolated workspaces, explicit spend limits, encrypted secrets, audit logs, and approval gates.
- It is a separate trust tier from local self-hosting.

## Setup

On the always-on Mac:

1. Open TensorFold.
2. Go to `Settings -> Orchestrator`.
3. Enable `Host this TensorFold as an Orchestrator`.
4. Keep `Local` for same-machine tests, or choose `LAN` for a trusted private network.
5. Use `Show pairing` to reveal the URL and token.
6. Enable `Keep computer alive` under `Settings -> Power` for overnight jobs.

For a dev checkout:

```bash
npm run start:orchestrator
```

The older alias still works:

```bash
npm run start:headless
```

Packaged builds can pass the same flag to Electron:

```bash
open -a TensorFold --args --headless-server
```

To survive reboots on the always-on Mac, install a LaunchAgent —
`~/Library/LaunchAgents/dev.tensorfold.studio.orchestrator.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.tensorfold.studio.orchestrator</string>
  <key>ProgramArguments</key><array>
    <string>/Applications/TensorFold.app/Contents/MacOS/TensorFold</string>
    <string>--headless-server</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/vontra-orchestrator.log</string>
  <key>StandardErrorPath</key><string>/tmp/vontra-orchestrator.log</string>
</dict></plist>
```

```bash
launchctl load ~/Library/LaunchAgents/dev.tensorfold.studio.orchestrator.plist
```

On another Mac:

1. Open TensorFold.
2. Go to `Settings -> Orchestrator`.
3. Enable `Connect to an Orchestrator`.
4. Paste the Orchestrator URL and token.
5. Click `Test connection`.

## Current API Surface

Authenticated endpoints:

- `GET /v1/snapshot`
- `GET /v1/events`
- `POST /v1/threads/:id/start`
- `POST /v1/threads/:id/guide`
- `POST /v1/threads/:id/cancel`
- `POST /v1/rooms/:id/start`
- `POST /v1/rooms/:id/cancel`
- `POST /v1/missions/:id/status`
- `POST /v1/missions/:id/heartbeat`

Unauthenticated endpoint:

- `GET /health`

## Global mode: run everything on the Orchestrator

`Settings -> Orchestrator -> Run everything on an Orchestrator` turns the whole machine
into a thin client (VST-041): **new tasks, chats, and terminals started here run on the
selected host**. Close the laptop and the work keeps going — the host owns every run; a
disconnected client only drops the live event stream. On reconnect the stream gap-fills
via `Last-Event-ID` (snapshot fallback), and for long gaps the durable event spine
(`GET /v1/spine/events`) reports exactly what changed since the client's cursor.

Precedence, always visible in the composer chip ("Mac mini · all new work"):

1. A project explicitly pinned to its own Orchestrator keeps that host.
2. A **local-only** project is a hard local pin the mode never overrides (its folders and
   providers may not exist on the host).
3. Everything else — including project-less chats and quick tasks — follows the mode.

Threads are stamped with their origin host (`thread.orchestratorHostId`) at creation and
when mirrored from snapshots, so runs, guidance, and cancels keep routing to the thread's
real home even if the mode or a project pin changes later. If the mode is on but no
enabled host exists, creation reports "Orchestrator unavailable" — nothing silently falls
back to local.

Approvals requested while you are away wait durably (they survive host restarts) and can
be resolved from the client when it returns; approval decisions are recorded in the
ordered event spine (tool name only, never arguments). For zero-touch overnight runs,
prefer Approve-Edits plus saved approval policies over Full Auto.

## Workspace Routing

The first project-routing slice stores `project.executionTarget` and `project.orchestratorId`:

- `local`: run work on this Mac.
- `orchestrator`: run work on a paired Orchestrator profile when remote routing is enabled.

The global execution target (`settings.defaultExecutionTarget` +
`settings.defaultOrchestratorId`) layers on top of the per-project pins with the
precedence described above; `src/lib/execution-target.ts` is the single resolution
helper the store, composer chip, and terminals tab all share.

Current UI:

- `Settings -> Orchestrator` can host this app, test a remote pairing, and save named Orchestrator profiles.
- `New project` can choose Local or Orchestrator and select the target host.
- `Project Settings` can switch the run location and target host after creation.
- `Command Centre` lists paired Orchestrators and can refresh a compact snapshot with active task, room, mission, and workspace counts.

Next work should add remote action buttons for chats, rooms, missions, terminal sessions, file browsing, and Git review against the selected host.

## Next Work

- Remote Git-review action buttons against the selected host.
- Hosted relay adapter for sign-in discovery and streaming.
- Optional mTLS or Tailscale identity checks.
- Pairing QR code for phone or laptop setup.
- Renderer catch-up projector over the spine replay (with VST-035's transport unification).
