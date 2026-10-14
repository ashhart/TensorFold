import { randomUUID } from 'node:crypto'
import { RUN_BUDGETS } from './connector-spine'
import { isLegalMissionTransition } from './mission-invariants'
import type {
  CreateMissionInput,
  Mission,
  MissionApprovalDecision,
  MissionBudgetSnapshot,
  MissionCheckpoint,
  MissionHeartbeat,
  MissionJournalActor,
  MissionJournalEvent,
  MissionJournalEventKind,
  MissionPendingApproval,
  MissionInspectionSnapshot,
  MissionRuntimeSnapshot,
  MissionScratchMemoryEntry,
  MissionScratchMemoryKind,
  MissionStatus,
  RunBudgetClass,
  RunBudgetDefinition,
} from './types'

export const DEFAULT_MISSION_BUDGET: RunBudgetClass = 'mission'

export const ACTIVE_MISSION_STATUSES = new Set<MissionStatus>([
  'queued',
  'running',
  'waiting_approval',
  'paused',
  'stalled',
])

export const TERMINAL_MISSION_STATUSES = new Set<MissionStatus>([
  'completed',
  'failed',
  'cancelled',
])

// Future IPC contract for the renderer/preload slice. This module owns the names
// so the eventual handlers can register the same channels without hunting.
export const MISSION_IPC_HOOKS = {
  list: 'missions:get',
  create: 'missions:create',
  patchStatus: 'missions:status',
  heartbeat: 'missions:heartbeat',
  inspect: 'missions:inspect',
  checkpoint: 'missions:checkpoint',
  journal: 'missions:journal',
  requestApproval: 'missions:approval:request',
  resolveApproval: 'missions:approval:resolve',
  snapshot: 'missions:snapshot',
} as const

export type MissionIpcHook = typeof MISSION_IPC_HOOKS[keyof typeof MISSION_IPC_HOOKS]

export interface MissionCheckpointInput {
  label: string
  summary: string
  data?: Record<string, unknown>
  artifactRefs?: string[]
  status?: MissionStatus
}

export interface MissionApprovalInput {
  callId: string
  itemId?: string
  title: string
  summary: string
  expiresAt?: number | null
  data?: Record<string, unknown>
}

export interface MissionJournalInput {
  kind?: MissionJournalEventKind
  actor?: MissionJournalActor
  message: string
  status?: MissionStatus
  checkpointId?: string
  approvalId?: string
  data?: Record<string, unknown>
}

export interface MissionScratchMemoryInput {
  kind?: MissionScratchMemoryKind
  text: string
  source?: MissionJournalActor
  runIndex?: number | null
  checkpointId?: string | null
}

export function createMission(input: CreateMissionInput, now = Date.now()): Mission {
  const runBudgetClass = input.runBudgetClass ?? DEFAULT_MISSION_BUDGET
  const budget = budgetSnapshot(runBudgetClass, now, false)
  const mission: Mission = {
    id: randomUUID(),
    title: input.title.trim() || 'Untitled mission',
    objective: input.objective.trim(),
    stopCondition: input.stopCondition?.trim() || '',
    repeatMode: input.repeatMode ?? 'until_condition',
    contextMode: input.contextMode ?? 'build_on_previous',
    roleProfileId: input.roleProfileId ?? 'operator',
    fanOutBudget: input.fanOutBudget ?? 'single',
    maxRuns: input.maxRuns ?? 15,
    costCapUsd: input.costCapUsd ?? null,
    expiresAt: input.expiresAt ?? null,
    destructiveApprovalRequired: input.destructiveApprovalRequired ?? true,
    status: 'queued',
    runBudgetClass,
    budget,
    heartbeat: heartbeatFromBudget(budget),
    checkpoints: [],
    journal: [],
    scratchMemory: [],
    pendingApprovals: [],
    threadId: input.threadId ?? null,
    projectId: input.projectId ?? null,
    cwd: input.cwd ?? null,
    providerId: input.providerId ?? null,
    providerAccountId: input.providerAccountId ?? null,
    model: input.model ?? null,
    createdAt: now,
    updatedAt: now,
    startedAt: null,
    finishedAt: null,
    metadata: input.metadata,
  }
  return appendJournal(mission, {
    kind: 'created',
    actor: 'system',
    message: `Mission queued with ${budget.label} budget.`,
    status: mission.status,
  }, now)
}

export function startMission(mission: Mission, now = Date.now()): Mission {
  if (TERMINAL_MISSION_STATUSES.has(mission.status)) return mission
  const startedAt = mission.startedAt ?? now
  const budget = budgetSnapshot(mission.runBudgetClass, startedAt, true)
  return appendJournal({
    ...mission,
    status: 'running',
    budget,
    heartbeat: heartbeatFromBudget(budget, now),
    startedAt,
    finishedAt: null,
    updatedAt: now,
  }, {
    kind: mission.startedAt ? 'resumed' : 'started',
    actor: 'system',
    message: mission.startedAt ? 'Mission resumed.' : 'Mission started.',
    status: 'running',
  }, now)
}

export function pauseMission(mission: Mission, message = 'Mission paused.', now = Date.now()): Mission {
  if (TERMINAL_MISSION_STATUSES.has(mission.status)) return mission
  return appendJournal({
    ...mission,
    status: 'paused',
    heartbeat: clearMissedHeartbeat(mission.heartbeat),
    updatedAt: now,
  }, { kind: 'paused', actor: 'system', message, status: 'paused' }, now)
}

export function completeMission(mission: Mission, message = 'Mission completed.', now = Date.now()): Mission {
  return finishMission(mission, 'completed', 'completed', message, now)
}

export function failMission(mission: Mission, message: string, now = Date.now()): Mission {
  return finishMission(mission, 'failed', 'error', message, now)
}

export function cancelMission(mission: Mission, message = 'Mission cancelled.', now = Date.now()): Mission {
  return finishMission(mission, 'cancelled', 'cancelled', message, now)
}

export function recordMissionHeartbeat(mission: Mission, message = 'Heartbeat recorded.', now = Date.now()): Mission {
  if (TERMINAL_MISSION_STATUSES.has(mission.status)) return mission
  const heartbeat = nextHeartbeat(mission.heartbeat, now)
  return appendJournal({
    ...mission,
    status: mission.status === 'stalled' ? 'running' : mission.status,
    heartbeat,
    updatedAt: now,
  }, { kind: 'heartbeat', actor: 'agent', message, status: mission.status === 'stalled' ? 'running' : mission.status }, now)
}

export function evaluateMissionHeartbeat(mission: Mission, now = Date.now()): Mission {
  if (!mission.heartbeat.required || TERMINAL_MISSION_STATUSES.has(mission.status)) return mission
  if (mission.status !== 'running' && mission.status !== 'waiting_approval') return mission
  if (mission.heartbeat.nextDueAt == null) return mission

  const missedAt = mission.heartbeat.nextDueAt + mission.heartbeat.stallGraceSeconds * 1000
  if (now < missedAt) return mission

  const heartbeat: MissionHeartbeat = {
    ...mission.heartbeat,
    missedSince: mission.heartbeat.missedSince ?? mission.heartbeat.nextDueAt,
  }
  return appendJournal({
    ...mission,
    status: 'stalled',
    heartbeat,
    updatedAt: now,
  }, {
    kind: 'status',
    actor: 'system',
    message: 'Mission heartbeat is overdue.',
    status: 'stalled',
  }, now)
}

export function addMissionCheckpoint(mission: Mission, input: MissionCheckpointInput, now = Date.now()): Mission {
  const checkpoint: MissionCheckpoint = {
    id: randomUUID(),
    sequence: mission.checkpoints.length + 1,
    label: input.label.trim() || `Checkpoint ${mission.checkpoints.length + 1}`,
    summary: input.summary.trim(),
    status: input.status ?? mission.status,
    createdAt: now,
    data: input.data,
    artifactRefs: input.artifactRefs,
  }
  return appendJournal({
    ...mission,
    checkpoints: [...mission.checkpoints, checkpoint],
    updatedAt: now,
  }, {
    kind: 'checkpoint',
    actor: 'agent',
    message: checkpoint.summary || checkpoint.label,
    status: checkpoint.status,
    checkpointId: checkpoint.id,
  }, now)
}

export function requestMissionApproval(mission: Mission, input: MissionApprovalInput, now = Date.now()): Mission {
  if (TERMINAL_MISSION_STATUSES.has(mission.status)) return mission
  // Only request approval from a status that may legally move to waiting_approval
  // (running, stalled, queued, or already waiting). A user-paused mission must NOT be
  // silently re-activated into an approval-wait by an in-flight/stale agent path —
  // paused -> waiting_approval is an illegal transition (mission-invariants.ts).
  if (!isLegalMissionTransition(mission.status, 'waiting_approval')) return mission
  const approval: MissionPendingApproval = {
    id: randomUUID(),
    callId: input.callId,
    itemId: input.itemId,
    title: input.title.trim() || 'Approval required',
    summary: input.summary.trim(),
    requestedAt: now,
    expiresAt: input.expiresAt ?? null,
    data: input.data,
  }
  return appendJournal({
    ...mission,
    status: 'waiting_approval',
    pendingApprovals: [...mission.pendingApprovals, approval],
    updatedAt: now,
  }, {
    kind: 'approval_requested',
    actor: 'agent',
    message: approval.summary || approval.title,
    status: 'waiting_approval',
    approvalId: approval.id,
  }, now)
}

export function resolveMissionApproval(mission: Mission, approvalId: string, decision: MissionApprovalDecision, now = Date.now()): Mission {
  const approval = mission.pendingApprovals.find((item) => item.id === approvalId || item.callId === approvalId)
  if (!approval) return mission
  const pendingApprovals = mission.pendingApprovals.filter((item) => item.id !== approval.id)
  // A mission can be flipped to 'stalled' by the heartbeat check while it waits for approval; once
  // the approval resolves it must resume, not stay stalled forever (lost resumption).
  const resumable = mission.status === 'waiting_approval' || mission.status === 'stalled'
  const nextStatus: MissionStatus = pendingApprovals.length ? 'waiting_approval' : resumable ? 'running' : mission.status
  const heartbeat = nextStatus === 'running' && mission.status === 'stalled'
    ? nextHeartbeat(mission.heartbeat, now) // clear the missed-heartbeat state so it isn't immediately re-stalled
    : mission.heartbeat
  return appendJournal({
    ...mission,
    status: nextStatus,
    pendingApprovals,
    heartbeat,
    updatedAt: now,
  }, {
    kind: 'approval_resolved',
    actor: 'user',
    message: `${approval.title}: ${decision}.`,
    status: nextStatus,
    approvalId: approval.id,
    data: { decision },
  }, now)
}

export function appendMissionJournal(mission: Mission, input: MissionJournalInput, now = Date.now()): Mission {
  return appendJournal(mission, {
    kind: input.kind ?? 'note',
    actor: input.actor ?? 'system',
    message: input.message,
    status: input.status,
    checkpointId: input.checkpointId,
    approvalId: input.approvalId,
    data: input.data,
  }, now)
}

export function addMissionScratchMemory(mission: Mission, input: MissionScratchMemoryInput, now = Date.now()): Mission {
  const text = input.text.replace(/\s+/g, ' ').trim()
  if (!text) return mission
  const entry: MissionScratchMemoryEntry = {
    id: randomUUID(),
    kind: input.kind ?? 'note',
    text: text.length > 1800 ? `${text.slice(0, 1799).trimEnd()}...` : text,
    source: input.source ?? 'agent',
    runIndex: input.runIndex == null || !Number.isFinite(Number(input.runIndex)) ? null : Math.max(0, Math.floor(Number(input.runIndex))),
    checkpointId: input.checkpointId ?? null,
    createdAt: now,
    updatedAt: now,
  }
  return appendJournal({
    ...mission,
    scratchMemory: [entry, ...(mission.scratchMemory ?? [])].slice(0, 48),
    updatedAt: now,
  }, {
    kind: 'note',
    actor: entry.source,
    message: `Scratch memory saved: ${entry.text.slice(0, 180)}`,
    checkpointId: entry.checkpointId ?? undefined,
    data: { scratchMemoryId: entry.id, scratchMemoryKind: entry.kind, runIndex: entry.runIndex },
  }, now)
}

export function forgetMissionScratchMemory(mission: Mission, id: string, now = Date.now()): Mission {
  const current = mission.scratchMemory ?? []
  const next = current.filter((entry) => entry.id !== id)
  if (next.length === current.length) return mission
  return appendJournal({
    ...mission,
    scratchMemory: next,
    updatedAt: now,
  }, {
    kind: 'note',
    actor: 'system',
    message: 'Mission scratch memory removed.',
    data: { scratchMemoryId: id },
  }, now)
}

export function formatMissionScratchMemory(mission: Pick<Mission, 'scratchMemory'>, maxEntries = 8): string {
  const entries = (mission.scratchMemory ?? [])
    .filter((entry) => entry.text.trim())
    .slice(0, Math.max(0, maxEntries))
    .map((entry) => {
      const run = entry.runIndex != null ? ` run ${entry.runIndex}` : ''
      return `- [${entry.kind}; ${entry.source}${run}] ${entry.text.trim()}`
    })
  return entries.join('\n').slice(0, 5000)
}

export function normalizeMissions(missions: Mission[], now = Date.now()): Mission[] {
  return missions.map((mission) => normalizeMission(mission, now))
}

export function normalizeMission(mission: Mission, now = Date.now()): Mission {
  const budget = mission.budget ?? budgetSnapshot(mission.runBudgetClass ?? DEFAULT_MISSION_BUDGET, mission.startedAt ?? now, Boolean(mission.startedAt))
  const heartbeat = mission.heartbeat ?? heartbeatFromBudget(budget, mission.startedAt ?? now)
  return {
    ...mission,
    stopCondition: mission.stopCondition ?? '',
    repeatMode: mission.repeatMode ?? 'until_condition',
    contextMode: mission.contextMode ?? 'build_on_previous',
    roleProfileId: mission.roleProfileId ?? 'operator',
    fanOutBudget: mission.fanOutBudget ?? 'single',
    maxRuns: mission.maxRuns ?? null,
    costCapUsd: mission.costCapUsd ?? null,
    expiresAt: mission.expiresAt ?? null,
    destructiveApprovalRequired: mission.destructiveApprovalRequired ?? true,
    runBudgetClass: mission.runBudgetClass ?? budget.executionClass,
    budget,
    heartbeat,
    checkpoints: mission.checkpoints ?? [],
    journal: mission.journal ?? [],
    scratchMemory: normalizeMissionScratchMemory(mission.scratchMemory, now),
    pendingApprovals: mission.pendingApprovals ?? [],
    threadId: mission.threadId ?? null,
    projectId: mission.projectId ?? null,
    cwd: mission.cwd ?? null,
    providerId: mission.providerId ?? null,
    providerAccountId: mission.providerAccountId ?? null,
    model: mission.model ?? null,
    startedAt: mission.startedAt ?? null,
    finishedAt: mission.finishedAt ?? null,
    updatedAt: mission.updatedAt ?? now,
  }
}

function normalizeMissionScratchMemory(input: MissionScratchMemoryEntry[] | undefined, now: number): MissionScratchMemoryEntry[] {
  return (input ?? [])
    .filter((entry) => entry && typeof entry.text === 'string' && entry.text.trim())
    .map((entry) => ({
      id: typeof entry.id === 'string' && entry.id ? entry.id : randomUUID(),
      kind: entry.kind ?? 'note',
      text: entry.text.replace(/\s+/g, ' ').trim(),
      source: entry.source ?? 'agent',
      runIndex: entry.runIndex == null || !Number.isFinite(Number(entry.runIndex)) ? null : Math.max(0, Math.floor(Number(entry.runIndex))),
      checkpointId: entry.checkpointId ?? null,
      createdAt: entry.createdAt ?? now,
      updatedAt: entry.updatedAt ?? entry.createdAt ?? now,
    }))
    .slice(0, 48)
}

export function buildMissionRuntimeSnapshot(missions: Mission[], now = Date.now()): MissionRuntimeSnapshot {
  const normalized = normalizeMissions(missions, now)
  return {
    generatedAt: now,
    missions: normalized,
    activeMissionIds: normalized.filter((mission) => ACTIVE_MISSION_STATUSES.has(mission.status)).map((mission) => mission.id),
    pendingApprovalCount: normalized.reduce((sum, mission) => sum + mission.pendingApprovals.length, 0),
  }
}

export function buildMissionInspectionSnapshot(mission: Mission, now = Date.now()): MissionInspectionSnapshot {
  const normalized = normalizeMission(mission, now)
  const latestCheckpoint = normalized.checkpoints.at(-1)
  const latestJournal = normalized.journal.at(-1)
  return {
    generatedAt: now,
    mission: normalized,
    terminal: TERMINAL_MISSION_STATUSES.has(normalized.status),
    active: ACTIVE_MISSION_STATUSES.has(normalized.status),
    rerunRequired: false,
    checkpointCount: normalized.checkpoints.length,
    journalCount: normalized.journal.length,
    scratchMemoryCount: normalized.scratchMemory.length,
    pendingApprovalCount: normalized.pendingApprovals.length,
    latestCheckpointId: latestCheckpoint?.id ?? null,
    latestJournalId: latestJournal?.id ?? null,
  }
}

export function inspectMissionById(missions: Mission[], id: string, now = Date.now()): MissionInspectionSnapshot {
  const mission = missions.find((candidate) => candidate.id === id)
  if (!mission) throw new Error('Mission not found')
  return buildMissionInspectionSnapshot(mission, now)
}

function finishMission(mission: Mission, status: MissionStatus, kind: MissionJournalEventKind, message: string, now: number): Mission {
  if (TERMINAL_MISSION_STATUSES.has(mission.status)) return mission
  return appendJournal({
    ...mission,
    status,
    heartbeat: clearMissedHeartbeat(mission.heartbeat),
    pendingApprovals: [],
    finishedAt: now,
    updatedAt: now,
  }, { kind, actor: 'system', message, status }, now)
}

function appendJournal(mission: Mission, input: Required<Pick<MissionJournalInput, 'kind' | 'actor' | 'message'>> & Omit<MissionJournalInput, 'kind' | 'actor' | 'message'>, now: number): Mission {
  const event: MissionJournalEvent = {
    id: randomUUID(),
    kind: input.kind,
    actor: input.actor,
    message: input.message,
    createdAt: now,
    status: input.status,
    checkpointId: input.checkpointId,
    approvalId: input.approvalId,
    data: input.data,
  }
  return {
    ...mission,
    journal: [...mission.journal, event],
    updatedAt: now,
  }
}

function budgetSnapshot(executionClass: RunBudgetClass, startAt: number, running: boolean): MissionBudgetSnapshot {
  const definition = budgetDefinition(executionClass)
  const startedAt = running ? startAt : null
  return {
    ...definition,
    startedAt,
    deadlineAt: startedAt && definition.maxWallSeconds > 0 ? startedAt + definition.maxWallSeconds * 1000 : null,
  }
}

function budgetDefinition(executionClass: RunBudgetClass): RunBudgetDefinition {
  return RUN_BUDGETS.find((budget) => budget.executionClass === executionClass) ?? RUN_BUDGETS.find((budget) => budget.executionClass === DEFAULT_MISSION_BUDGET)!
}

function heartbeatFromBudget(budget: MissionBudgetSnapshot, now: number | null = null): MissionHeartbeat {
  const lastBeatAt = now
  return {
    intervalSeconds: budget.heartbeatIntervalSeconds,
    stallGraceSeconds: budget.stallGraceSeconds,
    required: budget.requiresHeartbeat,
    count: lastBeatAt == null ? 0 : 1,
    lastBeatAt,
    nextDueAt: budget.requiresHeartbeat && lastBeatAt != null ? lastBeatAt + budget.heartbeatIntervalSeconds * 1000 : null,
    missedSince: null,
  }
}

function nextHeartbeat(heartbeat: MissionHeartbeat, now: number): MissionHeartbeat {
  return {
    ...heartbeat,
    count: heartbeat.count + 1,
    lastBeatAt: now,
    nextDueAt: heartbeat.required ? now + heartbeat.intervalSeconds * 1000 : null,
    missedSince: null,
  }
}

function clearMissedHeartbeat(heartbeat: MissionHeartbeat): MissionHeartbeat {
  return { ...heartbeat, missedSince: null }
}
