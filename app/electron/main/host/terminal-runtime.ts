import { spawn as ptySpawn, type IPty } from '@lydell/node-pty'
import { spawn } from 'node:child_process'
import os from 'node:os'
import { randomUUID } from 'node:crypto'
import { builtinClis, cliToolStatuses, launchString, resumeString, makeCustomCli } from '../../agent/cli-registry'
import { fetchCliLatestVersions, type CliLatestVersion } from '../../agent/cli-update-feeds'
import { createConnectorFetch } from '../../agent/secure-fetch'
import { nextStatus } from '../../agent/cli-status'
import { classifyVersionProbe, type VersionProbeResult, type VersionProbeRun } from '../../agent/cli-version-probe'
import { capScrollbackAnsiSafe, TerminalOutputBuffer } from '../../agent/terminal-multiplex'
import { detachedProcessGroup, terminateChildProcess, terminatePtyProcess } from '../../agent/child-process-lifecycle'
import { computeAugmentedPath, loginShellPath } from '../../agent/cli-path'
import type { CliDefinition, CliSession, CliSessionStatus, CliToolStatus } from '../../agent/types/cli'
import type { Project } from '../../agent/types'
import { store } from '../state'
import { saveCliSessions } from '../../store'
import { handle } from './ipc'
import { auditEvent, requireSecurityRole } from './audit'
import { auditPreview } from '../../agent/security'
import { blockedCommandReason } from '../../agent/command-policy'
import { agentTerminalRefusal } from '../../agent/planning-guard'
import { clearLiveTerminalSessions, registerLiveTerminalSession, removeLiveTerminalSession } from './terminal-live-state'
import { makeOutputCoalescer, terminalStatusEvent, terminalSessionsEvent, terminalExitEvent, type TerminalEvent, type TerminalStartInput } from '../../agent/terminal-forward'
import { lexicalLocalWorkspaceKey, type WorkspaceMutationGuard, type WorkspaceMutationLease } from './workspace-mutation-guard'

interface LiveTerminal {
  session: CliSession
  pty: IPty
  idleTimer: NodeJS.Timeout | null
  buffer: string // recent raw output, replayed when a pane re-opens (the PTY outlives the xterm)
  // Sequenced output ring (VST-039): a reconnecting/paired client gap-fills from its cursor
  // via terminal:since; a cursor below the retained window reports a gap so it re-baselines
  // instead of trusting a partial replay. Bounds an absent consumer without unbounded growth.
  output: TerminalOutputBuffer
  workspaceLease: WorkspaceMutationLease
}

// How much recent output to keep per session for scrollback replay on re-open (~512KB).
const BACKLOG_CAP = 512 * 1024
// Sequenced-ring cap for reconnect gap-fill (VST-039) — smaller than the scrollback buffer;
// a longer gap surfaces as a gap:true re-baseline, which is the correct, bounded behavior.
const OUTPUT_RING_CAP = 256 * 1024

// The PTY below always runs on this host. An SSH workspace path belongs to another
// machine, and node-pty would silently use a same-named local directory if one exists.
// Keep the dedicated Orchestrator-forwarded API separate and fail closed here until
// project terminals have an SSH-backed PTY implementation.
export function spawnLocalPtyForProject<T>(project: Pick<Project, 'workspace' | 'archivedAt'>, spawnPty: () => T): T {
  if (project.archivedAt) throw new Error('Restore the archived project before starting or restoring a terminal.')
  if (project.workspace?.kind === 'ssh') {
    throw new Error('Project terminals do not support SSH workspaces yet. Use an Orchestrator terminal or connect to the SSH host outside TensorFold.')
  }
  return spawnPty()
}

// All built-in CLIs the user can launch. Custom CLIs are passed inline on terminal:start (the
// renderer owns that list as local UI state — no Settings-schema change), so they don't appear here.
function knownClis(): CliDefinition[] {
  return builtinClis()
}

function versionArgs(cli: CliToolStatus): string[] | null {
  if (!cli.effectiveCommand) return null
  return ['--version']
}

async function probeCliVersion(cli: CliToolStatus): Promise<VersionProbeResult | null> {
  const args = versionArgs(cli)
  if (!cli.installed || !cli.effectiveCommand || !args) return null
  const bin = cli.configuredPath?.trim() || cli.effectiveCommand.trim().split(/\s+/)[0]
  if (!bin) return null
  const run = await new Promise<VersionProbeRun>((resolve) => {
    let settled = false
    let stdout = ''
    let stderr = ''
    const finish = (value: Omit<VersionProbeRun, 'stdout' | 'stderr'>) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve({ ...value, stdout, stderr })
    }
    const child = spawn(bin, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PATH: computeAugmentedPath(process.env.PATH) },
      detached: detachedProcessGroup(),
    })
    const timer = setTimeout(() => {
      terminateChildProcess(child, { processGroup: detachedProcessGroup() })
      finish({ code: null, timedOut: true })
    }, 5000)
    child.stdout.on('data', (data) => { stdout = (stdout + String(data)).slice(0, 2000) })
    child.stderr.on('data', (data) => { stderr = (stderr + String(data)).slice(0, 2000) })
    child.on('error', (error) => finish({ code: null, error: error.message }))
    child.on('close', (code, signal) => finish({ code, signal }))
  })
  return classifyVersionProbe(run, os.homedir())
}

// Module-level live map so killAllTerminals() can reach it without going through the
// closure returned by registerTerminalIpc.
const live = new Map<string, LiveTerminal>()

export function liveTerminalCount(): number {
  return live.size
}
let publishProjectTerminalSessions: ((projectId: string) => void) | null = null

// Set true while the app is quitting. killAllTerminals() kills every PTY, which FIRES pty.onExit —
// but those are exactly the sessions we want to RESTORE next boot, so onExit must NOT prune the
// persisted file during shutdown. A natural exit or an explicit user-kill (shuttingDown === false)
// still prunes. This is the clean-quit-vs-kill distinction that makes restore correct.
let shuttingDown = false

// --- persisted "was live" set (store.cliSessions ⇄ cli-sessions.json) -------------------------
// The persisted file represents the terminals that were running. Start adds; natural exit / user
// kill removes; rename updates the title. A finished/closed terminal must NOT be restored, so it is
// pruned the moment it ends on its own or is killed by the user (but never during app shutdown).
function persistCliSessions() {
  // Fire-and-forget: a failed write must not crash the main process. The next mutation re-saves.
  saveCliSessions(store.cliSessions).catch((e) => {
    console.warn(`[vontra] persist failed for cliSessions: ${String((e as { message?: string })?.message ?? e)}`)
  })
}

function rememberSession(session: CliSession) {
  // De-dupe by id (ids are uuids, so collisions don't happen in practice — defensive).
  store.cliSessions = [...store.cliSessions.filter((s) => s.id !== session.id), session]
  persistCliSessions()
}

function forgetSession(sessionId: string) {
  const next = store.cliSessions.filter((s) => s.id !== sessionId)
  if (next.length === store.cliSessions.length) return
  store.cliSessions = next
  persistCliSessions()
}

function renamePersisted(sessionId: string, title: string) {
  let changed = false
  store.cliSessions = store.cliSessions.map((s) => (s.id === sessionId ? (changed = true, { ...s, title }) : s))
  if (changed) persistCliSessions()
}

// Kill every live PTY. On app QUIT (the default) we keep the persisted "was live" set intact and
// suppress onExit's prune (shuttingDown) so the terminals come back next launch. A SECURITY PURGE
// passes { purge: true }: it still suppresses the prune (the caller shreds the file itself), but it
// resets the flag afterwards because the app keeps running — a later kill/exit must prune normally.
export function killAllTerminals(opts: { purge?: boolean } = {}) {
  shuttingDown = true
  for (const t of live.values()) {
    if (t.idleTimer) { try { clearTimeout(t.idleTimer) } catch { /* best effort */ } }
    // before-quit/security-purge cannot await a grace timer: synchronously
    // reap the whole PTY process group so descendants cannot survive teardown.
    terminatePtyProcess(t.pty, { force: true })
    t.workspaceLease.release()
  }
  live.clear()
  clearLiveTerminalSessions()
  // After a purge the process does NOT exit, so leave the flag false for normal pruning to resume.
  if (opts.purge) shuttingDown = false
}

/** Stop and forget every host-local PTY owned by an archived project. */
export function killProjectTerminals(projectId: string): number {
  let killed = 0
  for (const [sessionId, t] of [...live.entries()]) {
    if (t.session.projectId !== projectId) continue
    killed++
    if (t.idleTimer) { try { clearTimeout(t.idleTimer) } catch { /* best effort */ } }
    forgetSession(sessionId)
    terminatePtyProcess(t.pty)
    t.workspaceLease.release()
    live.delete(sessionId)
    removeLiveTerminalSession(sessionId)
  }
  if (killed) publishProjectTerminalSessions?.(projectId)
  return killed
}

export function registerTerminalIpc(opts: { remoteSink?: (ev: TerminalEvent) => void; workspaceMutationGuard?: Pick<WorkspaceMutationGuard, 'reserveStart'> } = {}) {
  const shell = process.env.SHELL || loginShellPath() || '/bin/zsh'
  const terminalWorkspaceGuard = opts.workspaceMutationGuard

  // Remote forwarding (orchestrator terminals): when a remoteSink is provided (the mini's
  // remote-control publish), sessions this machine hosts FOR a remote client (managedBy:'external')
  // mirror their output/status/session-list over it. Local-only sessions are never forwarded. PTY
  // output is high-rate, so it's coalesced and flushed on a ~25ms tick instead of per-chunk.
  const remoteSink = opts.remoteSink ?? null
  const coalescer = remoteSink ? makeOutputCoalescer() : null
  let flushTimer: ReturnType<typeof setInterval> | null = null
  function drainToSink() {
    if (!remoteSink || !coalescer) return
    for (const b of coalescer.drain()) remoteSink({ type: 'data', sessionId: b.sessionId, chunk: b.chunk })
  }
  function ensureFlushTimer() {
    if (!remoteSink || flushTimer) return
    flushTimer = setInterval(drainToSink, 25)
    if (typeof (flushTimer as { unref?: () => void }).unref === 'function') (flushTimer as { unref: () => void }).unref()
  }
  function forwardData(t: LiveTerminal, data: string) {
    if (!remoteSink || !coalescer || t.session.managedBy !== 'external') return
    ensureFlushTimer()
    if (coalescer.push(t.session.id, data)) drainToSink()
  }
  function forwardStatus(t: LiveTerminal) {
    if (remoteSink && t.session.managedBy === 'external') remoteSink(terminalStatusEvent(t.session.id, t.session.status))
  }
  function forwardSessions(projectId: string) {
    if (!remoteSink) return
    const ext = [...live.values()].filter((t) => t.session.projectId === projectId && t.session.managedBy === 'external').map((t) => t.session)
    remoteSink(terminalSessionsEvent(projectId, ext))
  }

  function emit(channel: string, payload: unknown) {
    // FIX 6: guard against send-after-destroy
    if (store.win && !store.win.webContents.isDestroyed()) store.win.webContents.send(channel, payload)
  }

  function setStatus(t: LiveTerminal, status: CliSessionStatus) {
    if (t.session.status === status) return
    t.session.status = status
    emit('terminal:status', { sessionId: t.session.id, status, orchestratorHostId: null })
    forwardStatus(t)
  }

  function scheduleIdle(t: LiveTerminal) {
    if (t.idleTimer) clearTimeout(t.idleTimer)
    t.idleTimer = setTimeout(() => {
      // FIX 3: also clear 'attention' (bell latches it; the idle timer must recover it)
      if (t.session.status === 'busy' || t.session.status === 'attention') setStatus(t, 'idle')
    }, 1100)
  }

  // Broadcast a project's current live sessions so the renderer (sidebar + terminal view)
  // stays in sync as terminals are started, renamed, killed, or exit on their own.
  function emitSessions(projectId: string) {
    emit('terminal:sessions', {
      projectId,
      sessions: [...live.values()].filter((t) => t.session.projectId === projectId).map((t) => t.session),
      orchestratorHostId: null,
    })
    forwardSessions(projectId)
  }
  publishProjectTerminalSessions = emitSessions

  // Spawn one PTY + wire data/exit, returning the LiveTerminal. Shared by terminal:start (fresh
  // launch) and terminal:restore (reattach). `resume` chooses resumeString (reattach) vs
  // launchString (fresh) for the command typed into the new shell after the 400ms settle.
  function spawnSession(cli: CliDefinition, project: Project, opts: { resume: boolean; session?: CliSession; managed?: 'local' | 'external' }): CliSession {
    // Agent CLIs would give their cloud model the whole folder; external models only plan.
    const refusal = agentTerminalRefusal(cli)
    if (refusal) throw new Error(refusal)
    const cwd = opts.session?.cwd || project.cwd || os.homedir()
    const env: Record<string, string | undefined> = {
      ...process.env,
      TERM: 'xterm-256color',
      COLORTERM: 'truecolor',
      LANG: process.env.LANG || 'en_US.UTF-8',
      VONTRA_TERMINAL: '1',
      PATH: computeAugmentedPath(process.env.PATH),
    }

    let workspaceLease: WorkspaceMutationLease | null = null
    const pty = spawnLocalPtyForProject(project, () => {
      if (!terminalWorkspaceGuard) throw new Error('Project terminals are unavailable because the workspace mutation guard is not configured.')
      const lease = terminalWorkspaceGuard.reserveStart({ workspaceKeys: [lexicalLocalWorkspaceKey(cwd)] })
      workspaceLease = lease
      try {
        return ptySpawn(shell, ['-l'], { name: 'xterm-256color', cols: 80, rows: 24, cwd, env })
      } catch (error) {
        lease.release()
        throw error
      }
    })
    if (!workspaceLease) throw new Error('Could not reserve the project terminal workspace.')
    // Reuse the restored session's identity/title on resume; build a fresh one on a normal start.
    const session: CliSession = opts.session
      ? { ...opts.session, status: 'idle', hostLocation: 'local', orchestratorHostId: undefined, managedBy: opts.managed ?? 'local' }
      : {
          id: randomUUID(), projectId: project.id, cliId: cli.id,
          title: cli.command ? cli.name : 'Shell', status: 'idle', createdAt: Date.now(),
          cwd, command: cli.builtin ? undefined : cli.command ?? undefined, builtin: cli.builtin,
          // PTY runs on THIS machine; 'external' when opened for a remote orchestrator client.
          hostLocation: 'local', managedBy: opts.managed ?? 'local',
        }
    const t: LiveTerminal = { session, pty, workspaceLease, idleTimer: null, buffer: '', output: new TerminalOutputBuffer({ maxBytes: OUTPUT_RING_CAP }) }
    live.set(session.id, t)
    registerLiveTerminalSession(session)

    pty.onData((data: string) => {
      // Keep a capped rolling buffer of output so a re-opened pane can replay the scrollback
      // (the PTY survives the xterm being unmounted; a fresh xterm would otherwise be blank).
      // ANSI-safe cap (VST-039): a plain .slice() can cut mid-escape, corrupting the colors/
      // cursor state of every replayed line — capScrollbackAnsiSafe trims to a clean boundary.
      t.buffer = capScrollbackAnsiSafe(t.buffer + data, BACKLOG_CAP)
      // Also feed the sequenced ring (VST-039) so a reconnecting/paired client can gap-fill
      // precisely from its cursor instead of re-fetching the whole scrollback.
      t.output.append(data)
      emit('terminal:data', { sessionId: session.id, chunk: data, orchestratorHostId: null })
      forwardData(t, data) // mirror to a remote client when this is an externally-managed session
      // FIX 4: route through setStatus so busy→busy no-ops don't emit redundant events
      setStatus(t, nextStatus(t.session.status, Buffer.from(data, 'utf8')))
      // Always reset the idle debounce on every chunk, even when status didn't change
      scheduleIdle(t)
    })
    pty.onExit(() => {
      setStatus(t, 'done')
      if (t.idleTimer) clearTimeout(t.idleTimer)
      // FIX 1a + FIX 2: remove dead session so terminal:list never returns 'done' entries
      live.delete(session.id)
      removeLiveTerminalSession(session.id)
      t.workspaceLease.release()
      // A process that ended on its own (or a user kill) must NOT be restored — prune it. But during
      // app shutdown killAllTerminals() also fires this; those are the ones to keep, so skip the prune.
      if (!shuttingDown) forgetSession(session.id)
      emitSessions(session.projectId)
    })

    // Type the launch/resume command into the fresh shell (Clyde's startIfNeeded). On restore we use
    // resumeString so codex/claude reattach to their last session instead of starting a new one.
    const cmd = opts.resume ? resumeString(cli) : launchString(cli)
    if (cmd) setTimeout(() => { try { pty.write(cmd + '\n') } catch { /* exited */ } }, 400)

    return session
  }

  // Resolve the CliDefinition for a persisted session: a builtin by id, else a custom CLI rebuilt
  // from its stored command (so a non-builtin launcher re-spawns its own command on restore).
  function resolveCli(session: CliSession): CliDefinition | undefined {
    const builtin = knownClis().find((c) => c.id === session.cliId)
    if (builtin) return builtin
    if (session.command) return makeCustomCli(session.title || session.command, session.command)
    return undefined
  }

  // Latest-version cache for CLI update advisories (VST-037). Non-blocking: a stale cache
  // kicks off a background refresh and serves what it has, so `terminal:listCliTools` never
  // waits up to the fetch timeout — the doctor's Refresh button naturally picks up the
  // result on the next pass. Gated by the same setting as app update checks; an opted-out
  // user's machine never phones the npm registry.
  let cliLatestCache: { at: number; values: Record<string, CliLatestVersion> } = { at: 0, values: {} }
  let cliLatestInFlight = false
  const npmRegistryFetch = createConnectorFetch('https://registry.npmjs.org', 'npm CLI update registry')
  const CLI_LATEST_TTL_MS = 6 * 60 * 60_000
  function cliLatestVersions(): Record<string, CliLatestVersion> {
    if (!store.settings.autoCheckUpdates) return {}
    if (Date.now() - cliLatestCache.at > CLI_LATEST_TTL_MS && !cliLatestInFlight) {
      cliLatestInFlight = true
      // Exact-origin validation plus resolve-once DNS pinning prevents a registry lookup
      // from being rebound to loopback/private/metadata services between policy and connect.
      void fetchCliLatestVersions((url, init) => npmRegistryFetch(url, init))
        .then((values) => { cliLatestCache = { at: Date.now(), values } })
        .catch(() => { /* offline registry: keep whatever we had */ })
        .finally(() => { cliLatestInFlight = false })
    }
    return cliLatestCache.values
  }

  // Lists the CLIs that are actually installed on this machine (plus the always-present plain shell),
  // so the renderer only offers launchers the user can really run. Reuses the binary-presence probe.
  handle('terminal:listClis', async () => {
    return cliToolStatuses(knownClis(), store.settings.providers, { overrides: store.settings.cliToolOverrides })
      .filter((tool) => tool.installed)
      .map(({ installed, version, effectiveCommand, configuredPath, providerIds, ...cli }) => ({
        ...cli,
        command: effectiveCommand,
      }))
  })

  handle('terminal:listCliTools', async (): Promise<CliToolStatus[]> => {
    // Two passes: probe installed versions first, then rebuild statuses WITH the probed
    // versions + cached latest-version feeds so `cliToolStatuses` can compute version
    // advisories (VST-037). Building advisories from a null version would always read
    // "unknown" and never surface a real update.
    const base = cliToolStatuses(knownClis(), store.settings.providers, { overrides: store.settings.cliToolOverrides })
    const versions: Record<string, string | null> = {}
    const problems: Record<string, string | null> = {}
    await Promise.all(base.map(async (tool) => {
      const probe = await probeCliVersion(tool)
      versions[tool.id] = probe?.version ?? null
      problems[tool.id] = probe?.problem ?? null
    }))
    const latest = cliLatestVersions()
    const latestVersions: Record<string, string | null> = {}
    const updateCommands: Record<string, string | null> = {}
    for (const [id, entry] of Object.entries(latest)) {
      latestVersions[id] = entry.latestVersion
      updateCommands[id] = entry.updateCommand
    }
    return cliToolStatuses(knownClis(), store.settings.providers, {
      overrides: store.settings.cliToolOverrides,
      versions,
      problems,
      latestVersions,
      updateCommands,
    })
  })

  // Plan 1a launches in the project dir with no tmux (tmux persistence is Phase 2). A custom CLI is
  // launched by passing `command` inline (the renderer's "Add" launchers, e.g. Gemini/Pi/OpenCode):
  // when no registry id matches we build an ad-hoc CliDefinition from the typed command.
  handle('terminal:start', async (_e, payload: { projectId: string; cliId: string; command?: string; label?: string }) => {
    requireSecurityRole('owner', 'start a project terminal')
    const project = store.projects.find((p) => p.id === payload.projectId)
    if (!project) throw new Error('Project not found.')
    const custom = typeof payload.command === 'string' ? payload.command.trim() : ''
    // Custom launch strings run in an unsandboxed login PTY; the same hard blocklist the
    // agent loop enforces must reject disguised destructive commands here too.
    const blockReason = custom ? blockedCommandReason(custom) : null
    if (blockReason) throw new Error(`Custom terminal command blocked: ${blockReason}`)
    const cli = knownClis().find((c) => c.id === payload.cliId)
      ?? (custom ? makeCustomCli(payload.label?.trim() || custom, custom) : undefined)
    if (!cli) throw new Error('CLI not found.')

    const session = spawnSession(cli, project, { resume: false })
    // Persist the now-live session (with cwd + custom command) so it can be restored next launch.
    rememberSession(session)

    await auditEvent('terminal.start', { cliId: cli.id, projectId: project.id, cwd: auditPreview(session.cwd ?? '') })
    emitSessions(project.id)
    return session
  })

  handle('terminal:write', async (_e, payload: { sessionId: string; data: string }) => {
    requireSecurityRole('owner', 'write to a project terminal')
    const t = live.get(payload.sessionId)
    if (t) { try { t.pty.write(payload.data) } catch { /* exited */ } }
  })

  handle('terminal:resize', async (_e, payload: { sessionId: string; cols: number; rows: number }) => {
    requireSecurityRole('owner', 'resize a project terminal')
    const t = live.get(payload.sessionId)
    if (t && payload.cols > 0 && payload.rows > 0) { try { t.pty.resize(payload.cols, payload.rows) } catch { /* exited */ } }
  })

  handle('terminal:kill', async (_e, payload: { sessionId: string }) => {
    requireSecurityRole('owner', 'kill a project terminal')
    const t = live.get(payload.sessionId)
    // The user explicitly closed this terminal — drop it from the persisted set so it is NOT restored.
    // (Do this even if the PTY already exited: the session may linger in the persisted file.)
    forgetSession(payload.sessionId)
    if (!t) return
    if (t.idleTimer) clearTimeout(t.idleTimer)
    terminatePtyProcess(t.pty)
    t.workspaceLease.release()
    live.delete(payload.sessionId)
    emitSessions(t.session.projectId)
  })

  // Rename a live session's tab/title. Persisted alongside the live set so a restored terminal keeps
  // its user-chosen name.
  handle('terminal:rename', async (_e, payload: { sessionId: string; title: string }) => {
    requireSecurityRole('owner', 'rename a project terminal')
    const t = live.get(payload.sessionId)
    if (!t) return
    const title = payload.title.trim()
    if (!title) return
    t.session.title = title
    renamePersisted(payload.sessionId, title)
    emitSessions(t.session.projectId)
  })

  handle('terminal:list', async (_e, projectId: string) => {
    return [...live.values()].filter((t) => t.session.projectId === projectId).map((t) => t.session)
  })

  // Persisted-but-not-running sessions for a project: terminals that were live at the last quit and
  // can be restored. The renderer auto-restores these when the project's terminal view opens.
  handle('terminal:restorable', async (_e, projectId: string) => {
    return store.cliSessions.filter((s) => s.projectId === projectId && !live.has(s.id))
  })

  // Re-spawn a persisted session in its saved cwd and reattach the CLI to its prior session
  // (resumeString). The session keeps its id/title/createdAt so the renderer treats it as the same
  // terminal coming back. A custom CLI rebuilds its definition from the stored command.
  handle('terminal:restore', async (_e, payload: { sessionId: string }) => {
    requireSecurityRole('owner', 'restore a project terminal')
    if (live.has(payload.sessionId)) return live.get(payload.sessionId)!.session
    const saved = store.cliSessions.find((s) => s.id === payload.sessionId)
    if (!saved) throw new Error('Session not found.')
    const project = store.projects.find((p) => p.id === saved.projectId)
    if (!project) throw new Error('Project not found.')
    const cli = resolveCli(saved)
    if (!cli) throw new Error('CLI not found.')

    const session = spawnSession(cli, project, { resume: true, session: saved })
    // Re-persist (refreshes status/fields; keeps it in the live set).
    rememberSession(session)

    await auditEvent('terminal.restore', { cliId: cli.id, projectId: project.id, cwd: auditPreview(session.cwd ?? '') })
    emitSessions(project.id)
    return session
  })

  // Replayed by a re-opened pane to restore its scrollback (the PTY outlives the xterm).
  handle('terminal:backlog', async (_e, sessionId: string) => {
    return live.get(sessionId)?.buffer ?? ''
  })

  // Sequenced gap-fill (VST-039): a reconnecting/paired client passes its last cursor and
  // gets only the chunks after it — or gap:true when its cursor fell below the retained ring,
  // telling it to clear and take the live tail. `head` is its new cursor. Passing the returned
  // head as `since` next time IS the ack (the ring evicts to the byte cap regardless, so an
  // absent consumer can never grow it without bound).
  handle('terminal:since', async (_e, payload: { sessionId: string; since?: number }) => {
    const t = live.get(payload.sessionId)
    if (!t) return { chunks: [], gap: false, head: 0 }
    const cursor = Number.isFinite(payload?.since) ? Number(payload.since) : 0
    t.output.ack(cursor) // the client confirmed everything through its cursor → free the ring
    return t.output.since(cursor)
  })

  // Remote-facing API: lets the mini's remote-control server drive terminals ON BEHALF OF a remote
  // client. Sessions started this way are managedBy:'external', so their output/status/list mirror
  // over the remote sink (and show as "externally managed" on the mini's own UI). Mirrors the
  // terminal:* IPC handlers but callable directly (no ipcMain event / renderer round-trip).
  function remoteLive(sessionId: string): LiveTerminal {
    const t = live.get(sessionId)
    if (!t || t.session.managedBy !== 'external') throw new Error('Remote terminal session not found.')
    return t
  }

  const remoteApi = {
    listClis(): CliDefinition[] {
      return cliToolStatuses(knownClis(), store.settings.providers, { overrides: store.settings.cliToolOverrides })
        .filter((tool) => tool.installed)
        .map(({ installed, version, effectiveCommand, configuredPath, providerIds, ...cli }) => ({
          ...cli,
          command: effectiveCommand,
        }))
    },
    start(input: TerminalStartInput): CliSession {
      const project = store.projects.find((p) => p.id === input.projectId)
      if (!project) throw new Error('Project not found.')
      const cli = knownClis().find((c) => c.id === input.cliId)
      if (!cli) throw new Error('CLI not found.')
      const session = spawnSession(cli, project, { resume: false, managed: 'external' })
      rememberSession(session)
      emitSessions(project.id)
      return session
    },
    write(sessionId: string, data: string): void {
      const t = remoteLive(sessionId)
      try { t.pty.write(data) } catch { /* exited */ }
    },
    resize(sessionId: string, cols: number, rows: number): void {
      const t = remoteLive(sessionId)
      if (t && cols > 0 && rows > 0) { try { t.pty.resize(cols, rows) } catch { /* exited */ } }
    },
    rename(sessionId: string, title: string): void {
      const t = remoteLive(sessionId)
      const clean = title.trim()
      if (!clean) return
      t.session.title = clean
      renamePersisted(sessionId, clean)
      emitSessions(t.session.projectId)
    },
    kill(sessionId: string): void {
      const t = remoteLive(sessionId)
      forgetSession(sessionId)
      if (t.idleTimer) clearTimeout(t.idleTimer)
      terminatePtyProcess(t.pty)
      t.workspaceLease.release()
      live.delete(sessionId)
      emitSessions(t.session.projectId)
    },
    list(projectId: string): CliSession[] {
      return [...live.values()].filter((t) => t.session.projectId === projectId && t.session.managedBy === 'external').map((t) => t.session)
    },
    backlog(sessionId: string): string {
      return remoteLive(sessionId).buffer
    },
  }

  return { remoteApi }
}

// The remote-control server's terminal deps are exactly this API surface.
export type TerminalRemoteApi = ReturnType<typeof registerTerminalIpc>['remoteApi']
