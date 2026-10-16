# OMP engine (VST-050)

Run a TensorFold thread **on the installed omp coding agent** (`@oh-my-pi/pi-coding-agent`)
via its headless rpc mode. TensorFold stays the GUI: timeline, plan panel (todos),
subagent tree, Guide steering, and session resume all mirror engine state; omp owns the
agent loop — its specialists, prewalk, worktrees, and todo discipline.

Wire facts were taken from the INSTALLED package source (`src/modes/rpc/*`,
`src/session/agent-session-events.ts`) plus a live `--mode=rpc` probe on this machine,
following the pi engine precedent (VST-033, `electron/agent/pi-cli.ts`).

## Architecture

```
ThreadPanel (Engine: Native loop | OMP agent)
      │ threads:setEngine / omp:status IPC
      ▼
thread-runtime.ts  ── engine === 'omp' ? ──▶  runOmpThread (electron/agent/omp-engine.ts)
      │                                            │
      │ native loop (default)                      │ OmpRpcClient (electron/agent/omp-rpc.ts)
      ▼                                            ▼
   runAgent                                     `omp --mode=rpc` child (LF-JSONL stdio)
```

- **Transport** (`omp-rpc.ts`): spawns `omp --mode=rpc --cwd <thread.cwd> [--resume <path>]`,
  negotiates the protocol (`ready` frame), sends newline-JSON commands, splits stdout on
  LF only (U+2028/U+2029 are valid inside JSON strings), reassembles `rpc_chunk`-split
  responses, correlates `response` frames by id, streams `AgentSessionEvent`s to
  listeners, and disposes via stdin EOF with a force-kill fallback plus a silence
  timeout.
- **Mapping** (`omp-engine.ts`, pure + tested): omp `message_start/update/end` →
  assistant items + text deltas (thinking is not mirrored); `tool_execution_start/end` →
  tool items with untrusted-output framing; `todo_reminder`/`notice` → notices;
  `agent_end` → done. Todo phases round-trip with `PlanStep` (statuses, blockers,
  nesting): multi-phase lists aggregate per phase and nest tasks as subtasks.
- **Run loop**: `set_subagent_subscription`, push the existing plan via `set_todos`,
  `prompt` with the user turn, Guide cues become `steer` commands, abort becomes the
  `abort` command, then a final `get_state` syncs todo phases + the session cursor
  (`thread.ompResumeCursor`, resumed with `--resume` next run) and `get_subagents`
  projects the engine's subagents into TensorFold's subagent store (stable omp ids).

## Security posture

The **host-tools bridge** (slice 2, shipped): whenever the host provides its approval
channel, omp's own mutating tools (`bash`/`edit`/`write`/`ast_edit`) are disabled via the
spawn toolset (`OMP_BRIDGED_TOOLSET` = reads, structural search, todos, task/hub
orchestration, web search) and TensorFold's `write_file`/`edit_file`/`apply_patch`/
`run_command` are registered as omp **host tools**. The engine's model plans; every
mutation executes through TensorFold's approval gate (`executeOmpHostTool` mirrors the
native loop's decision order: read-only auto-run → readonly deny → package-install
saved-policy gate → Full Auto risk bands → interactive prompt), the real `executeTool`
executor, and the audit trail. `host_tool_cancel` aborts pending calls with the run's
signal. Read-only runs keep omp restricted to `read,grep,glob` with no host tools.

**Propagation (verified in omp source):** subagent toolsets derive from the parent
session's enabled tools (`executor.ts`: `session.getEnabledToolNames()`), so the spawn
filter reaches every subagent, and host tools registered on the main rpc session do not
leak into subagent sessions. Net posture for a bridged run: omp subagents can research
(read/grep/glob/task orchestration) but cannot mutate — mutations exist ONLY as bridged
host tools on the main agent, which run through TensorFold's gate.

## Testing

- `omp-rpc.test.ts` runs the transport against a fixture child that speaks the real
  protocol (ready, correlated responses, chunk-split `get_state`, streamed events).
- `omp-engine.test.ts` covers the pure mappings, a full `runOmpThread` pass against a
  duck-typed client (prompt, todo push/sync, session cursor, subagent projection), and a
  **bridged round-trip**: a `host_tool_call` frame flows through the approval gate and
  back as a `host_tool_result`, with live `subagent_lifecycle` updates.
- `omp-host-tools.test.ts` walks the gate matrix against the real policy engine:
  readonly deny, suggest prompt/allow/deny, fail-closed without an approver, Full Auto
  risk bands, and the package-install saved-policy refusal.
- Live checks on this machine: `omp --mode=rpc` handshake probe and binary discovery
  (`omp/18.1.19` at `~/.local/bin/omp`).

## Follow-on slices

1. ~~Host-tools bridge~~ — shipped (see Security posture).
2. Subagent transcripts (`get_subagent_messages`) into the subagent viewer; verify
   spawn-policy propagation of the `--tools` filter to omp subagents.
3. `set_model` / `set_thinking_level` controls in the Engine panel.
4. Room seats on the omp engine; `switch_session`/`branch` in the timeline history menu.
