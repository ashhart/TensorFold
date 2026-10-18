// Loop Mode — main-process host. Turns the pure engine + bridge into real runs:
// spawns role threads through the existing thread runtime, runs verification
// shell, calls an independent critic model, and persists/streams an observable
// LoopRecord. Owns its own loops.json so it doesn't touch the central store shape.

import { randomUUID } from 'node:crypto'
import { exec as cpExec } from 'node:child_process'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { ApprovalMode, ArchiveKind, Project, Room, StartRunPayload, Thread, Ticket, WorkspaceLocation } from '../../agent/types'
import type { LoopContractSource, LoopRecord, RunContract, LoopPhaseId } from '../../agent/loop-types'
import { BUILD_PHASES, SHIP_PHASES } from '../../agent/loop-types'
import { READ_ONLY_ROLES } from '../../agent/loop-guards'
import { buildRunContract } from '../../agent/loop-contract'
import { runLoop } from '../../agent/loop-engine'
import { createLoopRuntime, type RoleTimelineItem, type LoopModelRef } from '../../agent/loop-runtime'
import { resolveProviderModel } from '../../agent/provider-accounts'
import { journalPath } from '../../agent/thread-journal'
import { store } from '../state'
import { persistThreads, persistTickets } from './persistence'
import { requireSecurityRole } from './audit'
import { readProtectedJson, vontraPath, writeProtectedJson } from '../../store'
import { resolveWorkspaceCwd } from './workspace'
import { canonicalLocalWorkspaceKey, lexicalLocalWorkspaceKey, type WorkspaceMutationGuard, type WorkspaceMutationLease } from './workspace-mutation-guard'

export interface LoopHostDeps {
  startThreadRun: (payload: StartRunPayload) => Promise<{ runId: string }>
  workspaceMutationGuard: Pick<WorkspaceMutationGuard, 'reserveStart'>
  /** Test seam; production uses the local command runner. */
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

/** Resolve ownership from the actual surface, never from renderer-supplied project metadata. */
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

// startThreadRun is fire-and-forget: it sets thread.status = 'running' synchronously and resolves
// with { runId } BEFORE the agent run completes (the run drives runWithRoute(...).finally(...) in
// the background, flipping status to 'idle'/'error' only when it settles). Loop Mode must wait for
// the run to actually finish before slicing the role's timeline, otherwise it captures an empty
// (or user-only) slice and the engine sees no role/critic output every cycle. Poll the observable
// thread status until it leaves 'running'. Returns the final observed status: 'idle'/'error' when
// the run settles, undefined if the thread vanished underneath us, or 'running' if the wall-clock
// cap elapsed (so a stuck run can't hang the loop forever). Pure + injectable (status getter +
// clock + sleep) so it can be unit-tested deterministically without the store or real timers.
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
  // Loop only while the run is still in flight; undefined (thread gone) and any settled status
  // ('idle'/'error') fall straight through. Bail at the deadline so a stuck run can't hang here.
  while (status === 'running' && now() < deadline) {
    await sleep(pollMs)
    status = getStatus()
  }
  return status
}

// The approvalMode a Loop role thread runs with. Read-only planning roles and the independent
// critic (which has no phase) always run readonly. Implementation/review roles inherit the
// user's actual saved posture rather than silently escalating to auto.
export function loopRoleApprovalMode(phase: LoopPhaseId | undefined, globalApprovalMode: ApprovalMode): ApprovalMode {
  return !phase || READ_ONLY_ROLES.includes(phase) ? 'readonly' : globalApprovalMode
}

function execCmd(command: string, cwd: string): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    cpExec(command, { cwd, timeout: 120_000, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
      const code = err ? (typeof (err as NodeJS.ErrnoException).code === 'number' ? (err as { code: number }).code : 1) : 0
      resolve({ code, stdout: String(stdout ?? ''), stderr: String(stderr ?? '') })
    })
  })
}

export type LoopCommandRunner = (command: string, cwd: string) => Promise<{ code: number; stdout: string; stderr: string }>

export interface LoopBaseline {
  baseSha: string
  snapshottedDirtyTree: boolean
}

function commandFailure(label: string, result: { stdout: string; stderr: string }): Error {
  const detail = (result.stderr || result.stdout).trim().slice(-1000)
  return new Error(`Could not create a safe Loop baseline (${label})${detail ? `: ${detail}` : '.'} No loop was started.`)
}

/**
 * Turn pre-existing staged, unstaged, and untracked work into a verified baseline commit before
 * Loop Mode can ever reset/clean the tree. On any add/commit/verification failure, restore the
 * original HEAD ref and exact index bytes; git add/commit do not change worktree file contents.
 */
export async function prepareLoopBaseline(cwd: string, run: LoopCommandRunner = execCmd, kind: 'baseline' | 'checkpoint' = 'baseline'): Promise<LoopBaseline> {
  const inside = await run('git rev-parse --is-inside-work-tree', cwd)
  if (inside.code !== 0 || inside.stdout.trim() !== 'true') throw commandFailure('this folder is not a Git worktree', inside)
  const conflicts = await run('git diff --name-only --diff-filter=U', cwd)
  if (conflicts.code !== 0) throw commandFailure('could not inspect unresolved conflicts', conflicts)
  if (conflicts.stdout.trim()) throw new Error('Resolve Git conflicts before starting Loop Mode. No loop was started.')

  const oldHeadResult = await run('git rev-parse --verify --quiet HEAD', cwd)
  const oldHead = oldHeadResult.code === 0 ? oldHeadResult.stdout.trim() : ''
  const status = await run('git status --porcelain', cwd)
  if (status.code !== 0) throw commandFailure('could not inspect the working tree', status)
  if (!status.stdout.trim()) return { baseSha: oldHead, snapshottedDirtyTree: false }

  const indexResult = await run('git rev-parse --git-path index', cwd)
  if (indexResult.code !== 0 || !indexResult.stdout.trim()) throw commandFailure('could not locate the Git index', indexResult)
  const rawIndexPath = indexResult.stdout.trim()
  const indexPath = path.isAbsolute(rawIndexPath) ? rawIndexPath : path.resolve(cwd, rawIndexPath)
  const originalIndex = await fs.readFile(indexPath).catch((error: NodeJS.ErrnoException) => {
    if (error?.code === 'ENOENT') return null
    throw error
  })
  const originalIndexMode = await fs.stat(indexPath).then((stat) => stat.mode).catch(() => 0o600)

  const restoreOriginalState = async (): Promise<void> => {
    const currentHeadResult = await run('git rev-parse --verify --quiet HEAD', cwd)
    const currentHead = currentHeadResult.code === 0 ? currentHeadResult.stdout.trim() : ''
    let refError: Error | null = null
    if (currentHead && currentHead !== oldHead) {
      const restoreRef = oldHead
        ? await run(`git update-ref HEAD ${oldHead} ${currentHead}`, cwd)
        : await run(`git update-ref -d HEAD ${currentHead}`, cwd)
      if (restoreRef.code !== 0) refError = commandFailure('could not restore the original HEAD after snapshot failure', restoreRef)
    }
    try {
      if (originalIndex == null) {
        await fs.rm(indexPath, { force: true })
      } else {
        await fs.mkdir(path.dirname(indexPath), { recursive: true })
        const tmp = `${indexPath}.vontra-restore-${randomUUID()}`
        await fs.writeFile(tmp, originalIndex, { mode: originalIndexMode })
        await fs.rename(tmp, indexPath)
      }
    } catch (error) {
      throw new Error(`Loop baseline failed and the original Git index could not be restored: ${error instanceof Error ? error.message : String(error)}`)
    }
    if (refError) throw refError
  }

  const add = await run('git add -A', cwd)
  if (add.code !== 0) {
    await restoreOriginalState()
    throw commandFailure('git add failed', add)
  }
  const message = kind === 'checkpoint' ? 'loop checkpoint' : 'loop baseline (pre-existing uncommitted work)'
  const commit = await run(`git -c user.name="TensorFold" -c user.email=tensorfold-studio@example.test commit -q -m "${message}" --no-verify --no-gpg-sign`, cwd)
  if (commit.code !== 0) {
    await restoreOriginalState()
    throw commandFailure('git commit failed', commit)
  }

  const [newHeadResult, cleanResult] = await Promise.all([
    run('git rev-parse --verify --quiet HEAD', cwd),
    run('git status --porcelain', cwd),
  ])
  const newHead = newHeadResult.code === 0 ? newHeadResult.stdout.trim() : ''
  if (!newHead || newHead === oldHead || cleanResult.code !== 0 || cleanResult.stdout.trim()) {
    await restoreOriginalState()
    throw new Error('Could not verify a clean, new Loop baseline commit. The original Git state was restored and no loop was started.')
  }
  return { baseSha: newHead, snapshottedDirtyTree: true }
}

export async function rollbackLoopWorkspace(cwd: string, target: string, run: LoopCommandRunner = execCmd): Promise<void> {
  const reset = await run(target ? `git reset --hard ${target}` : 'git reset --hard', cwd)
  if (reset.code !== 0) throw commandFailure('rollback reset failed', reset)
  if (target) {
    const head = await run('git rev-parse --verify --quiet HEAD', cwd)
    if (head.code !== 0 || head.stdout.trim() !== target) throw commandFailure('rollback target could not be verified', head)
  }
  // Never clean untracked files if reset failed or landed on the wrong commit.
  const clean = await run('git clean -fd', cwd)
  if (clean.code !== 0) throw commandFailure('rollback clean failed', clean)
  const status = await run('git status --porcelain', cwd)
  if (status.code !== 0 || status.stdout.trim()) throw commandFailure('rollback did not leave a clean worktree', status)
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

/** Pick a critic model distinct from the workers' (cloud for rigor or a local one); fall back to same. */
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

// #31 — when a loop runs from a room, use the room's seats as its roles instead
// of one default model: the builder/development seat does the work, a
// reviewer/planner seat critiques. Falls back to the thread default + a distinct
// critic when seats don't cover a role.
function seatModelRef(seat: { providerId?: string; model?: string } | undefined): LoopModelRef | null {
  if (!seat?.providerId || !seat.model) return null
  const provider = store.settings.providers.find((p) => p.id === seat.providerId)
  // If the seat is pinned to a non-chat model (MarkItDown etc.) and the provider
  // has no chat alternative, return null so the caller falls back rather than
  // staffing a loop role with a model that can't converse.
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

  // #32 — a loop launched from a kanban ticket stays bound to it: the ticket
  // reflects the loop's run state (and lands in Review / Needs clarification when
  // the loop finishes), so the board mirrors what the loop is doing.
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

  // Remove a throwaway "loop role" thread once its timeline has been captured. Without
  // this, every phase/critique of every cycle permanently accumulated a "loop role"
  // thread in memory, on disk (threads.json), and in the sidebar (cleanupThread used to
  // be a no-op and critic threads were never removed). Mirrors threads:delete: drop it
  // from the store + prompt map, persist, and remove its NDJSON journal.
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
    // Loop role threads run autonomously (#33) — never stall waiting for a human to approve a tool
    // call mid-loop in a way the engine can't drive forward. Read-only planning roles
    // (product/architect) run in readonly mode so they cannot mutate source; they produce
    // planning text and are evidence-exempt, so this is safe regardless of the global posture.
    // Every other role inherits the user's saved global approvalMode (#16) rather than a
    // hardcoded 'auto', so Loop Mode respects the same approval/sandbox posture as any other run.
    const approvalMode = loopRoleApprovalMode(phase, store.settings.approvalMode)
    await deps.startThreadRun({ threadId, userText, approvalMode })
    // startThreadRun is fire-and-forget (returns { runId } before the run settles); wait for the
    // run to actually finish so the timeline slice below captures the role's real output rather
    // than an empty/user-only slice. See waitForThreadSettled for details.
    await waitForThreadSettled(() => store.threads.find((t) => t.id === threadId)?.status)
    const after = store.threads.find((t) => t.id === threadId)
    return (after?.timeline.slice(before) ?? []).map((i) => ({ kind: i.kind, text: i.text, tool: i.tool, state: i.state }))
  }

  async function startLoop(input: StartLoopInput): Promise<LoopRecord> {
    requireSecurityRole('operator', 'start a loop')
    if (quiesced) throw new Error('Loop Mode is temporarily unavailable while local memory is being purged.')
    const ownership = assertLoopSurfaceSupported(input)
    // Reserve synchronously from the host-owned surface path before the first filesystem/Git
    // await. Checkpoint/full-file restore now cannot interleave between Loop phases.
    const runLease = reserveLoopWorkspace(deps.workspaceMutationGuard, ownership.cwd)
    let leaseTransferred = false
    try {
    const cwd = await resolveWorkspaceCwd(ownership.cwd)
    runLease.addWorkspaceKeys?.([await canonicalLocalWorkspaceKey(cwd)])
    // Room loops staff their roles from the room's seats (#31); solo/ticket use
    // the surface thread's model with a distinct critic.
    const roomModels = input.source === 'room' ? modelsForRoom(input.surfaceId, modelForThread(input.surfaceId)) : null
    const worker = input.workerModel ?? roomModels?.worker ?? modelForThread(input.surfaceId)
    const critic = input.criticModel ?? roomModels?.critic ?? differentCriticModel(worker)
    // Fail fast with a clear message rather than spawning role threads on an empty
    // model (#7) — modelForThread can resolve to '' when nothing is configured.
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

    // Prove the rollback baseline before publishing a running record or changing a ticket.
    // prepareLoopBaseline restores the exact pre-call index/HEAD if snapshot creation fails.
    const baseline = await prepareLoopBaseline(cwd, deps.baselineRunner ?? execCmd)

    const rec: LoopRecord = {
      id: randomUUID(),
      source: input.source,
      surfaceId: input.surfaceId,
      goal: contract.goal,
      mode: input.mode,
      status: 'running',
      cwd,
      branch,
      phase: null,
      cycle: 0,
      maxCycles: contract.maxCycles,
      checkpoints: 0,
      lastVerdict: null,
      blockers: [],
      log: [],
      roleThreadIds: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }
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
      runCritic: async (system, user) => {
        const t = makeRoleThread(cwd, ownership.projectId, critic, system, 'chat')
        rec.roleThreadIds.push(t.id)
        try {
          const items = await runThreadCapture(t.id, user)
          const last = [...items].reverse().find((i) => i.kind === 'assistant' && (i.text ?? '').trim())
          return last?.text ?? ''
        } finally {
          // Critic threads are throwaway too and bypass the engine's cleanupThread hook,
          // so remove the captured critic thread here to avoid unbounded accumulation.
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
        // Return the worktree to the most recent loop checkpoint, or the pre-loop
        // baseline if none — discarding only THIS cycle's uncommitted changes. Never
        // `reset --hard HEAD~1`: that would delete a real commit (the user's own commit
        // on cycle 0, or a still-good earlier checkpoint later).
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
            rec.lastVerdict = v
            log(rec, v.passed ? `✅ critic passed (${Math.round(v.confidence * 100)}%)` : `❌ critic failed: ${v.summary}`)
            return v
          },
        })
        rec.status = result.status
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
        // Safety net: drop any role/critic threads still left over (e.g. a phase that
        // threw before its cleanupThread finally ran), so a settled loop never leaves
        // throwaway "loop role" threads behind in the store/sidebar.
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

  /** Stop and await direct Git/role activity before an owning archive operation returns. */
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
    // Stop persistence first so neither a status/log update nor a late finalizer can recreate
    // loops.json after the security purge shreds it. The host aborts underlying thread runs too;
    // awaiting activeRuns makes their cleanup/persistence finalizers a real barrier.
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
