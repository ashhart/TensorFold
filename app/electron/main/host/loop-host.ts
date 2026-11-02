// Own Loop records and adapt thread, model, verification, and workspace operations to the pure engine.

import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { ApprovalMode, ArchiveKind, Project, Room, StartRunPayload, Thread, Ticket, WorkspaceLocation } from '../../agent/types'
import type { LoopContractSource, LoopRecord, RunContract, LoopPhaseId } from '../../agent/loop-types'
import { BUILD_PHASES, SHIP_PHASES } from '../../agent/loop-types'
import { READ_ONLY_ROLES } from '../../agent/loop-guards'
import { buildRunContract } from '../../agent/loop-contract'
import { runLoop, loopConfigurationBlocker } from '../../agent/loop-engine'
import { createLoopRuntime, type RoleTimelineItem, type LoopModelRef } from '../../agent/loop-runtime'
import { resolveProviderModel } from '../../agent/provider-accounts'
import { journalPath } from '../../agent/thread-journal'
import { execCmd, prepareLoopBaseline, rollbackLoopWorkspace, type LoopCommandRunner } from './loop-workspace'
export { prepareLoopBaseline, rollbackLoopWorkspace, type LoopCommandRunner, type LoopBaseline } from './loop-workspace'
import { store } from '../state'
import { persistThreads, persistTickets } from './persistence'
import { requireSecurityRole } from './audit'
import { readProtectedJson, vontraPath, writeProtectedJson } from '../../store'
import { resolveWorkspaceCwd } from './workspace'
import { canonicalLocalWorkspaceKey, lexicalLocalWorkspaceKey, type WorkspaceMutationGuard, type WorkspaceMutationLease } from './workspace-mutation-guard'

export interface LoopHostDeps {
  startThreadRun: (payload: StartRunPayload) => Promise<{ runId: string }>
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'reserveStart'>
  // Test seam; production uses the local command runner.
  baselineRunner?: LoopCommandRunner
}

export interface StartLoopInput {
  source: LoopContractSource
  surfaceId: string
  goal: string
  mode: 'build' | 'ship'
  baseCwd: string
  projectId?: string | null
  criteria?: string[]
  verify?: string[]
  workerModel?: LoopModelRef
  criticModel?: LoopModelRef
  ticket?: { acceptanceCriteria?: string[]; requiredEvidence?: string[]; allowedCapabilities?: string[]; risk?: string }
}

type LoopSurfaceState = {
  projects: Array<Pick<Project, 'id' | 'cwd' | 'workspace' | 'archivedAt'>>
  threads: Array<Pick<Thread, 'id' | 'projectId' | 'cwd' | 'workspace' | 'archivedAt'>>
  rooms: Array<Pick<Room, 'id' | 'projectId' | 'workspace' | 'workspaceCwd' | 'archivedAt'>>
  tickets: Array<Pick<Ticket, 'id' | 'projectId' | 'roomId' | 'threadId' | 'archivedAt'>>
}

// Resolve ownership from the actual surface, never from renderer-supplied project metadata.
export function assertLoopSurfaceSupported(
  input: Pick<StartLoopInput, 'source' | 'surfaceId' | 'projectId' | 'baseCwd'>,
  state: LoopSurfaceState = store,
): { projectId: string | null; cwd: string } {
  let projectId: string | null = null
  let expectedCwd = ''
  const localCwd = (workspace: WorkspaceLocation | null | undefined, fallback: string | null | undefined): string => {
    if (workspace?.kind === 'ssh') throw new Error('Local Loop Mode does not support SSH workspaces; run this loop on the remote Orchestrator.')
    return workspace?.kind === 'local' && workspace.path?.trim() ? workspace.path.trim() : String(fallback ?? '').trim()
  }
  const activeProject = (id: string | null): Project | null => {
    if (!id) return null
    const project = state.projects.find((candidate) => candidate.id === id) as Project | undefined
    if (!project) throw new Error('Loop source project not found.')
    if (project.archivedAt) throw new Error('Restore the archived project before starting a loop.')
    if (project.workspace?.kind === 'ssh') throw new Error('Local Loop Mode does not support SSH workspaces; run this loop on the remote Orchestrator.')
    return project
  }
  if (input.source === 'solo') {
    const thread = state.threads.find((candidate) => candidate.id === input.surfaceId)
    if (!thread) throw new Error('Loop source thread not found.')
    if (thread.archivedAt) throw new Error('Restore the archived thread before starting a loop.')
    projectId = thread.projectId ?? null
    if (input.projectId != null && input.projectId !== projectId) throw new Error('Loop project does not match its source surface.')
    const project = activeProject(projectId)
    expectedCwd = localCwd(thread.workspace, thread.cwd || project?.cwd)
  } else if (input.source === 'room') {
    const room = state.rooms.find((candidate) => candidate.id === input.surfaceId)
    if (!room) throw new Error('Loop source room not found.')
    if (room.archivedAt) throw new Error('Restore the archived room before starting a loop.')
    projectId = room.projectId
    if (input.projectId != null && input.projectId !== projectId) throw new Error('Loop project does not match its source surface.')
    const project = activeProject(projectId)
    if (room.workspace?.kind === 'ssh') throw new Error('Local Loop Mode does not support SSH workspaces; run this loop on the remote Orchestrator.')
    expectedCwd = String(room.workspaceCwd ?? '').trim() || localCwd(room.workspace, project?.cwd)
  } else {
    const ticket = state.tickets.find((candidate) => candidate.id === input.surfaceId)
    if (!ticket) throw new Error('Loop source ticket not found.')
    if (ticket.archivedAt) throw new Error('Restore the archived ticket before starting a loop.')
    projectId = ticket.projectId
    if (input.projectId != null && input.projectId !== projectId) throw new Error('Loop project does not match its source surface.')
    const project = activeProject(projectId)
    if (ticket.threadId) {
      const thread = state.threads.find((candidate) => candidate.id === ticket.threadId)
      if (!thread || thread.archivedAt) throw new Error('Restore the active ticket thread before starting a loop.')
      if (thread.projectId !== projectId) throw new Error('Loop ticket thread must belong to the same project.')
      expectedCwd = localCwd(thread.workspace, thread.cwd)
    } else if (ticket.roomId) {
      const room = state.rooms.find((candidate) => candidate.id === ticket.roomId)
      if (!room) throw new Error('Loop ticket room not found.')
      if (room.archivedAt) throw new Error('Restore the archived ticket room before starting a loop.')
      if (room.projectId !== projectId) throw new Error('Loop ticket project does not match its room project.')
      if (room.workspace?.kind === 'ssh') throw new Error('Local Loop Mode does not support SSH workspaces; run this loop on the remote Orchestrator.')
      expectedCwd = String(room.workspaceCwd ?? '').trim() || localCwd(room.workspace, project?.cwd)
    } else {
      expectedCwd = localCwd(project?.workspace, project?.cwd)
    }
  }
  if (!expectedCwd) throw new Error('Loop source does not have a local workspace folder.')
  const requested = String(input.baseCwd ?? '').trim()
  if (!requested || path.resolve(requested) !== path.resolve(expectedCwd)) throw new Error('Loop workspace does not match its source surface.')
  return { projectId, cwd: expectedCwd }
}

export function reserveLoopWorkspace(
  guard: Pick<WorkspaceMutationGuard, 'reserveStart'>,
  cwd: string,
): WorkspaceMutationLease {
  return guard.reserveStart({ workspaceKeys: [lexicalLocalWorkspaceKey(cwd)] })
}

const LOOPS_FILE = ['loops.json']

// Wait for the background role run to settle before reading its timeline; the injected clock bounds stalled runs.
export async function waitForThreadSettled(
  getStatus: () => 'idle' | 'running' | 'error' | undefined,
  opts: { pollMs?: number; timeoutMs?: number; now?: () => number; sleep?: (ms: number) => Promise<void> } = {},
): Promise<'idle' | 'running' | 'error' | undefined> {
  const pollMs = opts.pollMs ?? 100
  const timeoutMs = opts.timeoutMs ?? 31 * 60_000 // slightly above runAgent's 30min wall cap
  const now = opts.now ?? Date.now
  const sleep = opts.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)))
  const deadline = now() + timeoutMs
  let status = getStatus()
  // Missing or settled threads return immediately; a deadline bounds a stuck role.
  while (status === 'running' && now() < deadline) {
    await sleep(pollMs)
    status = getStatus()
  }
  return status
}

// Planning and critic threads are read-only; implementation roles retain the user's saved approval posture.
export function loopRoleApprovalMode(phase: LoopPhaseId | undefined, globalApprovalMode: ApprovalMode): ApprovalMode {
  return !phase || READ_ONLY_ROLES.includes(phase) ? 'readonly' : globalApprovalMode
}

// Skip utility/non-chat models (MarkItDown, embeddings, etc.) when auto-selecting loop models.
const NON_CHAT_MODEL = /^markitdown$|(^|[-_/ ])(embed(ding)?s?|rerank(er)?|whisper|moderation)([-_/ ]|$)/i
function firstChatModel(models: string[] | undefined): string | undefined {
  return (models ?? []).find((m) => m && !NON_CHAT_MODEL.test(m))
}

function modelForThread(threadId: string): LoopModelRef {
  const t = store.threads.find((x) => x.id === threadId)
  const providerId = t?.providerId ?? store.settings.activeProviderId
  const provider = store.settings.providers.find((p) => p.id === providerId)
  const resolved = t?.model ?? (provider ? resolveProviderModel(provider, null, store.settings.activeModel) : store.settings.activeModel)
  const model = resolved && !NON_CHAT_MODEL.test(resolved) ? resolved : (firstChatModel(provider?.models) ?? resolved ?? '')
  return { providerId: providerId ?? '', model: model ?? '' }
}

// Prefer a distinct critic model, falling back to the worker when no alternative is available.
function differentCriticModel(worker: LoopModelRef): LoopModelRef {
  for (const p of store.settings.providers) {
    if (p.id === worker.providerId) continue
    const m = firstChatModel(p.models)
    if (m) return { providerId: p.id, model: m }
  }
  const wp = store.settings.providers.find((p) => p.id === worker.providerId)
  const alt = wp?.models.find((m) => m !== worker.model && !NON_CHAT_MODEL.test(m))
  if (wp && alt) return { providerId: wp.id, model: alt }
  return worker
}

// Room seats staff worker and critic roles; missing seats fall back to the thread model and a distinct critic.
function seatModelRef(seat: { providerId?: string; model?: string } | undefined): LoopModelRef | null {
  if (!seat?.providerId || !seat.model) return null
  const provider = store.settings.providers.find((p) => p.id === seat.providerId)
  // A non-chat seat without a chat alternative falls back to the normal role selection.
  const model = !NON_CHAT_MODEL.test(seat.model) ? seat.model : firstChatModel(provider?.models)
  if (!model) return null
  return { providerId: seat.providerId, model }
}
function modelsForRoom(roomId: string, fallback: LoopModelRef): { worker: LoopModelRef; critic: LoopModelRef } {
  const seats = store.rooms.find((r) => r.id === roomId)?.seats ?? []
  if (!seats.length) return { worker: fallback, critic: differentCriticModel(fallback) }
  const builder = seats.find((s) => s.lane === 'development' || s.roleProfileId === 'builder' || s.roleProfileId === 'operator')
  const worker = seatModelRef(builder) ?? seatModelRef(seats[0]) ?? fallback
  const reviewer = seats.find((s) => (s.lane === 'review' || s.lane === 'planning' || s.roleProfileId === 'reviewer' || s.roleProfileId === 'planner') && s !== builder)
    ?? seats.find((s) => s !== builder)
  const critic = seatModelRef(reviewer) ?? differentCriticModel(worker)
  return { worker, critic }
}

function newLoopRecord(input: StartLoopInput, contract: RunContract, cwd: string, branch: string): LoopRecord {
  return {
    id: randomUUID(), source: input.source, surfaceId: input.surfaceId,
    goal: contract.goal, mode: input.mode, status: 'running', cwd, branch,
    phase: null, cycle: 0, maxCycles: contract.maxCycles, checkpoints: 0,
    lastVerdict: null, blockers: [], log: [], roleThreadIds: [],
    createdAt: Date.now(), updatedAt: Date.now(),
  }
}

export function createLoopHost(deps: LoopHostDeps) {
  const loops = new Map<string, LoopRecord>()
  const aborted = new Set<string>()
  const promptByThread = new Map<string, string>()
  const activeRuns = new Map<string, Promise<void>>()
  let persistTail: Promise<void> = Promise.resolve()
  let quiesced = false

  function persistLoops(): Promise<void> {
    if (quiesced) return Promise.resolve()
    const snapshot = structuredClone([...loops.values()])
    const next = persistTail.catch(() => {}).then(() => writeProtectedJson(vontraPath(...LOOPS_FILE), snapshot))
    persistTail = next.catch(() => {})
    return next
  }

  function emit(rec: LoopRecord) {
    rec.updatedAt = Date.now()
    if (quiesced) return
    store.win?.webContents.send('loop:update', rec)
    void persistLoops().catch(() => {})
  }
  function log(rec: LoopRecord, line: string) {
    rec.log.push(`${new Date().toISOString().slice(11, 19)} ${line}`)
    if (rec.log.length > 200) rec.log.splice(0, rec.log.length - 200)
    emit(rec)
  }

  // Mirror a ticket's loop state, placing finished work in Review or Needs clarification.
  function reflectOnTicket(rec: LoopRecord, runState: 'running' | 'done', status?: import('../../agent/types').TicketStatus) {
    if (rec.source !== 'ticket') return
    const ticket = store.tickets.find((t) => t.id === rec.surfaceId)
    if (!ticket) return
    ticket.runState = runState
    if (status) ticket.status = status
    if (runState === 'running' && !ticket.startedAt) ticket.startedAt = Date.now()
    ticket.updatedAt = Date.now()
    persistTickets()
    store.win?.webContents.send('ticket:update', ticket)
  }

  function makeRoleThread(cwd: string, projectId: string | null | undefined, m: LoopModelRef, systemPrompt: string, kind: 'task' | 'chat'): { id: string } {
    const t: Thread = {
    documentSourceIds: [],
      id: randomUUID(),
      title: 'loop role',
      kind,
      projectId: projectId ?? null,
      cwd,
      providerId: m.providerId,
      providerAccountId: null,
      model: m.model,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      status: 'idle',
      apiMessages: [],
      timeline: [],
    }
    store.threads.unshift(t)
    promptByThread.set(t.id, systemPrompt)
    persistThreads()
    return { id: t.id }
  }

  // Remove captured role threads from memory, persistence, prompts, and journals to avoid accumulating sidebar entries.
  function removeRoleThread(id: string): void {
    const idx = store.threads.findIndex((t) => t.id === id)
    if (idx === -1) { promptByThread.delete(id); return }
    store.threads.splice(idx, 1)
    promptByThread.delete(id)
    persistThreads()
    void fs.rm(journalPath(vontraPath('thread-journals'), id), { force: true }).catch(() => {})
  }

  async function runThreadCapture(threadId: string, instruction: string, phase?: LoopPhaseId): Promise<RoleTimelineItem[]> {
    const sys = promptByThread.get(threadId) ?? ''
    const before = store.threads.find((t) => t.id === threadId)?.timeline.length ?? 0
    const userText = sys ? `${sys}\n\n---\nTASK:\n${instruction}` : instruction
    // Implementation roles keep the user's approval posture; planning and critic roles cannot write source.
    const approvalMode = loopRoleApprovalMode(phase, store.settings.approvalMode)
    await deps.startThreadRun({ threadId, userText, approvalMode })
    // startThreadRun returns before completion; capture evidence only after the role actually settles.
    await waitForThreadSettled(() => store.threads.find((t) => t.id === threadId)?.status)
    const after = store.threads.find((t) => t.id === threadId)
    return (after?.timeline.slice(before) ?? []).map((i) => ({ kind: i.kind, text: i.text, tool: i.tool, state: i.state }))
  }

  async function startLoop(input: StartLoopInput): Promise<LoopRecord> {
    requireSecurityRole('operator', 'start a loop')
    if (quiesced) throw new Error('Loop Mode is temporarily unavailable while local memory is being purged.')
    const ownership = assertLoopSurfaceSupported(input)
    // Reserve the host-owned workspace before any await so checkpoint restores cannot interleave with a Loop phase.
    const runLease = reserveLoopWorkspace(deps.workspaceMutationGuard, ownership.cwd)
    let leaseTransferred = false
    try {
    const cwd = await resolveWorkspaceCwd(ownership.cwd)
    runLease.addWorkspaceKeys?.([await canonicalLocalWorkspaceKey(cwd)])
    // Room seats staff roles; solo and ticket runs use the surface model with a distinct critic.
    const roomModels = input.source === 'room' ? modelsForRoom(input.surfaceId, modelForThread(input.surfaceId)) : null
    const worker = input.workerModel ?? roomModels?.worker ?? modelForThread(input.surfaceId)
    const critic = input.criticModel ?? roomModels?.critic ?? differentCriticModel(worker)
    // Reject an unconfigured model before spawning role threads.
    if (!worker.providerId || !worker.model) throw new Error('No chat model is configured for this loop. Connect a model in Settings first.')
    if (!store.settings.providers.some((provider) => provider.id === worker.providerId)) throw new Error('The Loop worker provider is not configured.')
    if (!critic.providerId || !critic.model || !store.settings.providers.some((provider) => provider.id === critic.providerId)) {
      throw new Error('The Loop critic provider/model is not configured.')
    }
    const contract: RunContract = buildRunContract({
      source: input.source,
      mode: input.mode,
      goal: input.goal,
      criteria: input.criteria,
      verify: input.verify,
      ticket: input.ticket,
    })
    const phases = input.mode === 'ship' ? SHIP_PHASES : BUILD_PHASES
    const branch = `vontra/loop-${Date.now().toString(36)}`
    const configuration = loopConfigurationBlocker(contract)
    if (configuration) {
      const blocked = newLoopRecord(input, contract, cwd, branch)
      blocked.status = 'blocked'
      blocked.lastVerdict = configuration.lastVerdict ?? null
      blocked.blockers = configuration.blockers
      loops.set(blocked.id, blocked)
      log(blocked, configuration.lastVerdict?.summary ?? 'Verification commands must be configured.')
      return blocked
    }

    // Verify the rollback baseline before publishing a running record or changing a ticket.
    const baseline = await prepareLoopBaseline(cwd, deps.baselineRunner ?? execCmd)

    const rec = newLoopRecord(input, contract, cwd, branch)
    loops.set(rec.id, rec)
    log(rec, `Loop started (${input.mode}) — worker ${worker.providerId}/${worker.model}, critic ${critic.providerId}/${critic.model}`)
    if (baseline.snapshottedDirtyTree) log(rec, 'Snapshotted pre-existing uncommitted work into a verified Loop baseline commit')
    reflectOnTicket(rec, 'running')
    const baseSha = baseline.baseSha
    const checkpointShas: string[] = []

    const runtime = createLoopRuntime(contract, {
      cwd,
      projectId: ownership.projectId,
      workerModel: worker,
      criticModel: critic,
      createRoleThread: async (opts) => {
        const t = makeRoleThread(opts.cwd, opts.projectId, { providerId: opts.providerId, model: opts.model }, opts.systemPrompt, opts.kind)
        rec.roleThreadIds.push(t.id)
        return t
      },
      runRoleThread: async (threadId, instruction, phase) => {
        if (aborted.has(rec.id)) throw new Error('Loop stopped')
        return runThreadCapture(threadId, instruction, phase)
      },
      cleanupThread: async (threadId) => { removeRoleThread(threadId) },
      runShell: (shellCwd, command) => execCmd(command, shellCwd),
      isCancelled: () => aborted.has(rec.id),
      runCritic: async (system, user) => {
        const t = makeRoleThread(cwd, ownership.projectId, critic, system, 'chat')
        rec.roleThreadIds.push(t.id)
        try {
          const items = await runThreadCapture(t.id, user)
          const last = [...items].reverse().find((i) => i.kind === 'assistant' && (i.text ?? '').trim())
          return last?.text ?? ''
        } finally {
          // Critic threads bypass the phase cleanup hook and need their own removal.
          removeRoleThread(t.id)
        }
      },
      checkpoint: async (cycle) => {
        const checkpoint = await prepareLoopBaseline(cwd, deps.baselineRunner ?? execCmd, 'checkpoint')
        const head = checkpoint.baseSha
        if (!head) throw new Error('Loop checkpoint could not verify a target commit.')
        if (checkpoint.snapshottedDirtyTree) {
          checkpointShas.push(head)
          rec.checkpoints++
          log(rec, `checkpoint cycle ${cycle}`)
        } else {
          log(rec, `checkpoint cycle ${cycle} (no workspace changes)`)
        }
        return head
      },
      rollback: async () => {
        // Restore the latest checkpoint or baseline, preserving earlier user commits and successful cycles.
        const target = checkpointShas[checkpointShas.length - 1] || baseSha
        await rollbackLoopWorkspace(cwd, target, deps.baselineRunner ?? execCmd)
        log(rec, `rolled back to ${target ? `checkpoint ${target.slice(0, 8)}` : 'empty baseline'} (regression)`)
      },
      emit: (line) => log(rec, line),
    })

    // Drive the loop in the background; surface terminal state on the record.
    const activeRun = (async () => {
      try {
        const result = await runLoop(contract, phases, {
          ...runtime,
          runPhase: async (phase, c, cycle) => {
            if (aborted.has(rec.id)) throw new Error('Loop stopped')
            rec.phase = phase
            rec.cycle = cycle + 1
            emit(rec)
            return runtime.runPhase(phase, c, cycle)
          },
          critique: async (c, evidence) => {
            if (aborted.has(rec.id)) throw new Error('Loop stopped')
            rec.phase = 'verify'
            emit(rec)
            const v = await runtime.critique(c, evidence)
            log(rec, v.passed ? 'Critic accepted the evidence; required checks still govern completion.' : `Critic failed: ${v.summary}`)
            return v
          },
        })
        rec.status = result.status
        rec.lastVerdict = result.lastVerdict ?? null
        rec.blockers = result.blockers
        rec.phase = null
        log(rec, `Loop ${result.status} after ${result.cycles} cycle(s)`)
        reflectOnTicket(rec, 'done', result.status === 'shipped' ? 'review' : 'needs_clarification')
      } catch (e) {
        rec.status = aborted.has(rec.id) ? 'stopped' : 'failed'
        rec.phase = null
        log(rec, `Loop ${rec.status}: ${e instanceof Error ? e.message : String(e)}`)
        reflectOnTicket(rec, 'done', 'needs_clarification')
      } finally {
        // Clean up any role or critic threads left behind by a failed phase.
        for (const id of rec.roleThreadIds.splice(0)) removeRoleThread(id)
        runLease.release()
      }
    })()
    activeRuns.set(rec.id, activeRun)
    void activeRun.finally(() => { activeRuns.delete(rec.id) }).catch(() => {})

    leaseTransferred = true
    return rec
    } finally {
      if (!leaseTransferred) runLease.release()
    }
  }

  function getLoops(): LoopRecord[] {
    return [...loops.values()].sort((a, b) => b.createdAt - a.createdAt)
  }
  function stopLoop(id: string): { stopped: boolean } {
    const rec = loops.get(id)
    if (!rec || rec.status !== 'running') return { stopped: false }
    aborted.add(id)
    log(rec, 'Stop requested')
    return { stopped: true }
  }

  function loopMatchesArchive(rec: LoopRecord, kind: ArchiveKind, id: string): boolean {
    if (kind === 'thread') {
      if (rec.source === 'solo') return rec.surfaceId === id
      if (rec.source === 'ticket') return store.tickets.find((ticket) => ticket.id === rec.surfaceId)?.threadId === id
      return false
    }
    if (kind === 'room') {
      if (rec.source === 'room') return rec.surfaceId === id
      if (rec.source === 'ticket') return store.tickets.find((ticket) => ticket.id === rec.surfaceId)?.roomId === id
      return false
    }
    if (kind === 'ticket') return rec.source === 'ticket' && rec.surfaceId === id
    const projectId = rec.source === 'solo'
      ? store.threads.find((thread) => thread.id === rec.surfaceId)?.projectId
      : rec.source === 'room'
        ? store.rooms.find((room) => room.id === rec.surfaceId)?.projectId
        : store.tickets.find((ticket) => ticket.id === rec.surfaceId)?.projectId
    return projectId === id
  }

  // Stop and await direct Git/role activity before an owning archive operation returns.
  async function stopLoopsForArchive(kind: ArchiveKind, id: string): Promise<string[]> {
    const ids = [...loops.values()]
      .filter((rec) => rec.status === 'running' && loopMatchesArchive(rec, kind, id))
      .map((rec) => rec.id)
    for (const loopId of ids) {
      aborted.add(loopId)
      const rec = loops.get(loopId)
      if (rec) log(rec, `Stopped because owning ${kind} was archived`)
    }
    await Promise.allSettled(ids.map((loopId) => activeRuns.get(loopId)).filter((run): run is Promise<void> => !!run))
    return ids
  }

  async function init() {
    try {
      const parsed = await readProtectedJson<LoopRecord[]>(vontraPath(...LOOPS_FILE), [])
      for (const r of parsed) {
        // Mark loops that were mid-run when the app closed as stopped.
        if (r.status === 'running') r.status = 'stopped'
        loops.set(r.id, r)
      }
    } catch {
      // no prior loops
    }
  }

  async function quiesceAndClear(): Promise<void> {
    // Disable persistence before purge, then await active-run finalizers so no late writer can recreate loop records.
    quiesced = true
    for (const rec of loops.values()) if (rec.status === 'running') aborted.add(rec.id)
    await Promise.allSettled([...activeRuns.values()])
    await persistTail.catch(() => {})
    loops.clear()
    aborted.clear()
    promptByThread.clear()
  }

  function resumeAfterPurge(): void {
    quiesced = false
  }

  return { startLoop, getLoops, stopLoop, stopLoopsForArchive, init, quiesceAndClear, resumeAfterPurge }
}
