import { randomUUID } from 'node:crypto'
import type { Project, Room, Thread, Ticket } from '../../agent/types'
import {
  normalizeStringList,
  normalizeTicketCapabilities,
  normalizeTicketEvidence,
  normalizeTicketExecution,
} from '../../agent/tickets'
import { assertTicketDependencyGraph, normalizeTicketDependencyIds } from '../../../shared/ticket-dependencies'

export interface TicketRelationState {
  projects: Project[]
  rooms: Room[]
  tickets: Ticket[]
  threads: Thread[]
}

/** Enforce that every ticket relation stays inside one active project ownership graph. */
export function assertTicketRelations(
  state: TicketRelationState,
  projectId: string,
  roomId?: string | null,
  relations: { selfId?: string; parentId?: string | null; threadId?: string | null; requireActiveThread?: boolean; dependsOn?: string[] } = {},
): void {
  const project = state.projects.find((candidate) => candidate.id === projectId)
  if (!project || project.archivedAt) throw new Error('Tickets can only belong to an active project.')
  if (roomId) {
    const room = state.rooms.find((candidate) => candidate.id === roomId)
    if (!room || room.archivedAt || room.projectId !== projectId) throw new Error('Tickets can only belong to an active room in the same project.')
  }
  if (relations.parentId) {
    const seen = new Set<string>()
    let cursorId: string | null = relations.parentId
    while (cursorId) {
      if (cursorId === relations.selfId) throw new Error('A ticket cannot be its own parent or descendant.')
      if (seen.has(cursorId)) throw new Error('Ticket parent hierarchy cannot contain a cycle.')
      seen.add(cursorId)
      const parent = state.tickets.find((candidate) => candidate.id === cursorId)
      if (!parent) throw new Error('Parent ticket not found.')
      if (parent.archivedAt) throw new Error('Restore the parent ticket before assigning this subticket.')
      if (parent.projectId !== projectId) throw new Error('Parent ticket must belong to the same project.')
      cursorId = parent.parentId ?? null
    }
  }
  if (relations.threadId) {
    const thread = state.threads.find((candidate) => candidate.id === relations.threadId)
    if (!thread) throw new Error('Ticket thread not found.')
    if (thread.projectId !== projectId) throw new Error('Ticket thread must belong to the same project.')
    if (relations.requireActiveThread !== false && thread.archivedAt) throw new Error('Restore the ticket thread before assigning it.')
  }
  if (relations.dependsOn) {
    assertTicketDependencyGraph(state.tickets, {
      id: relations.selfId ?? '__new_ticket__',
      projectId,
      dependsOn: relations.dependsOn,
    })
  }
}

export function mkTicket(projectId: string, p: Partial<Ticket>): Ticket {
  const execution = normalizeTicketExecution(p, { roomScoped: !!p.roomId })
  return {
    id: randomUUID(), projectId, title: p.title || 'Untitled', body: p.body || '',
    roomId: p.roomId ?? null,
    type: p.type || 'task', status: p.status || 'todo', priority: p.priority || 'medium', effort: p.effort || 'medium',
    tags: normalizeStringList(p.tags, []), ...execution,
    dependsOn: normalizeTicketDependencyIds(p.dependsOn),
    dependencyFailurePolicy: p.dependencyFailurePolicy === 'continue' ? 'continue' : 'block',
    parentId: p.parentId || null, assignee: p.assignee ?? null, threadId: p.threadId ?? null, runState: p.runState ?? null,
    startedAt: p.startedAt ?? null, order: Date.now() + Math.random(), createdAt: Date.now(), updatedAt: Date.now(),
  }
}

export function describeTicketForRoom(t: Ticket): string {
  const assignee = t.assignee ? ` · @${t.assignee}` : ''
  const risk = t.risk ? ` · risk:${t.risk}` : ''
  const caps = normalizeTicketCapabilities(t.allowedCapabilities, ['read']).join(',')
  const required = normalizeStringList(t.requiredEvidence, [])
  const evidenceCount = t.evidence?.length ?? 0
  const evidence = required.length ? ` · evidence:${evidenceCount}/${required.length}` : ''
  const criteria = normalizeStringList(t.acceptanceCriteria, [])
  const details = [
    required.length ? `  required: ${required.join('; ')}` : '',
    criteria.length ? `  acceptance: ${criteria.join('; ')}` : '',
  ].filter(Boolean).join('\n')
  return `- [${t.status}${assignee}${risk} · caps:${caps}${evidence}] ${t.title}${details ? `\n${details}` : ''}`
}

export function ticketEvidenceFromArgs(args: any, seat: string): NonNullable<Ticket['evidence']> {
  const raw = Array.isArray(args?.evidence)
    ? args.evidence
    : (args?.summary || args?.ref)
      ? [{ kind: args?.kind, summary: args?.summary ?? args?.ref, ref: args?.ref }]
      : []
  const now = Date.now()
  return normalizeTicketEvidence(raw).map((item) => ({
    ...item,
    id: randomUUID(),
    createdBy: item.createdBy ?? seat,
    createdAt: now,
  }))
}

interface TicketWorkerCommandSignal {
  command: string
  state?: string
  output?: string
  meta?: string
}

interface TicketWorkerDiffSignal {
  path: string | null
  additions: number
  deletions: number
}

export interface TicketWorkerSignals {
  codeChanged: boolean
  testsPassed: boolean
  files: string[]
  commands: TicketWorkerCommandSignal[]
  diffs: TicketWorkerDiffSignal[]
  blockers: string[]
}

function ticketWorkerSignals(thread: Thread): TicketWorkerSignals {
  const files = new Set<string>()
  const commands: TicketWorkerCommandSignal[] = []
  const diffs = new Map<string, TicketWorkerDiffSignal>()
  const blockers: string[] = []
  let codeChanged = false
  let testsPassed = false
  for (const item of thread.timeline) {
    if (item.path) files.add(item.path)
    let additions = 0
    let deletions = 0
    for (const line of item.diff ?? []) {
      if (line.type === 'add') additions += 1
      if (line.type === 'del') deletions += 1
    }
    if (additions || deletions) {
      codeChanged = true
      const key = item.path ?? item.id
      const existing = diffs.get(key)
      diffs.set(key, {
        path: item.path ?? existing?.path ?? null,
        additions: (existing?.additions ?? 0) + additions,
        deletions: (existing?.deletions ?? 0) + deletions,
      })
    }
    if (item.kind === 'tool') {
      const command = item.command || (typeof item.args?.command === 'string' ? item.args.command : '')
      if (command) {
        commands.push({
          command,
          state: item.state,
          output: firstSignalLine(item.output),
          meta: firstSignalLine(item.meta),
        })
        const haystack = `${command}\n${item.output ?? ''}\n${item.meta ?? ''}`.toLowerCase()
        if (isTestCommand(command) && commandSignalPassed(item.state, haystack)) testsPassed = true
      }
      if ((command && item.state !== 'ok') || item.state === 'error' || /blocked|failed|error|denied|cancelled|canceled|interrupted|awaiting/i.test(`${item.title ?? ''} ${item.output ?? ''}`)) {
        blockers.push(`${item.tool ?? item.title ?? 'tool'}: ${String(item.output ?? item.text ?? '').split('\n')[0].slice(0, 120)}`)
      }
    }
    if (item.kind === 'error' && item.text) blockers.push(item.text.split('\n')[0].slice(0, 120))
  }
  return { codeChanged, testsPassed, files: [...files], commands: dedupeCommands(commands).slice(-4), diffs: [...diffs.values()].slice(-8), blockers: blockers.slice(-4) }
}

export function summarizeTicketWorkerOutcome(ticket: Ticket, thread: Thread, phase: 'build' | 'review', failed: boolean, createdBy: string | null): { text: string; codeChanged: boolean; testsPassed: boolean; evidence: NonNullable<Ticket['evidence']>; signals: TicketWorkerSignals } {
  const signals = ticketWorkerSignals(thread)
  const bits = [
    `${phase === 'review' ? 'Reviewer' : 'Builder'} worker ${failed ? 'blocked' : 'finished'} "${ticket.title}".`,
    signals.files.length ? `Files: ${signals.files.slice(0, 5).join(', ')}.` : '',
    signals.diffs.length ? `Diffs: ${signals.diffs.slice(0, 5).map((item) => `${item.path ?? 'changes'} +${item.additions}/-${item.deletions}`).join(', ')}.` : '',
    signals.commands.length ? `Commands: ${signals.commands.map((item) => item.command).join(' | ')}.` : '',
    signals.testsPassed ? 'Verification: tests/checks passed.' : '',
    signals.blockers.length ? `Blockers: ${signals.blockers.join(' | ')}.` : '',
    `Thread: ${thread.id}.`,
  ].filter(Boolean)
  return {
    text: bits.join(' '),
    codeChanged: signals.codeChanged,
    testsPassed: signals.testsPassed,
    signals,
    evidence: ticketWorkerEvidence(ticket, thread, phase, failed, signals, createdBy),
  }
}

function ticketWorkerEvidence(ticket: Ticket, thread: Thread, phase: 'build' | 'review', failed: boolean, signals: TicketWorkerSignals, createdBy: string | null): NonNullable<Ticket['evidence']> {
  const now = Date.now()
  const by = createdBy ?? null
  const evidence: NonNullable<Ticket['evidence']> = [{
    id: randomUUID(),
    kind: phase === 'review' ? 'review' : 'note',
    summary: `${phase === 'review' ? 'Reviewer' : 'Builder'} worker ${failed ? 'blocked' : 'finished'} "${ticket.title}". Thread: ${thread.id}.`,
    ref: `thread:${thread.id}`,
    createdBy: by,
    createdAt: now,
  }]

  for (const file of signals.files.slice(0, 8)) {
    evidence.push({
      id: randomUUID(),
      kind: 'file',
      summary: `${phase === 'review' ? 'Reviewed' : 'Touched'} ${file}.`,
      ref: file,
      createdBy: by,
      createdAt: now,
    })
  }

  for (const diff of signals.diffs) {
    const ref = diff.path ?? `thread:${thread.id}`
    evidence.push({
      id: randomUUID(),
      kind: 'diff',
      summary: `Worker diff ${diff.path ?? 'changes'}: +${diff.additions}/-${diff.deletions}.`,
      ref,
      createdBy: by,
      createdAt: now,
    })
  }

  for (const item of signals.commands) {
    const test = isTestCommand(item.command)
    const passed = test && commandSignalPassed(item.state, `${item.command}\n${item.output ?? ''}\n${item.meta ?? ''}`.toLowerCase())
    evidence.push({
      id: randomUUID(),
      kind: passed ? 'test' : 'command',
      summary: `${passed ? 'Test passed' : 'Command'}: ${item.command}${item.output ? ` -> ${item.output}` : ''}`,
      ref: `thread:${thread.id}`,
      createdBy: by,
      createdAt: now,
    })
  }

  if (signals.testsPassed && !signals.commands.some((item) => isTestCommand(item.command))) {
    evidence.push({
      id: randomUUID(),
      kind: 'test',
      summary: 'Verification passed in worker thread.',
      ref: `thread:${thread.id}`,
      createdBy: by,
      createdAt: now,
    })
  }

  for (const blocker of signals.blockers) {
    evidence.push({
      id: randomUUID(),
      kind: 'note',
      summary: `Blocker: ${blocker}`,
      ref: `thread:${thread.id}`,
      createdBy: by,
      createdAt: now,
    })
  }

  return evidence
}

function firstSignalLine(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const line = value.split('\n').map((item) => item.trim()).find(Boolean)
  return line ? line.slice(0, 160) : undefined
}

function dedupeCommands(commands: TicketWorkerCommandSignal[]): TicketWorkerCommandSignal[] {
  const order: string[] = []
  const selected = new Map<string, TicketWorkerCommandSignal>()
  for (const command of commands) {
    const previous = selected.get(command.command)
    if (!previous) order.push(command.command)
    const passed = commandSignalPassed(command.state, `${command.command}\n${command.output ?? ''}\n${command.meta ?? ''}`)
    const previousPassed = previous && commandSignalPassed(previous.state, `${previous.command}\n${previous.output ?? ''}\n${previous.meta ?? ''}`)
    if (!previous || passed || !previousPassed) selected.set(command.command, command)
  }
  return order.map((command) => selected.get(command)!).filter(Boolean)
}

function isTestCommand(command: string): boolean {
  return /\b(test|tsx|vitest|jest|playwright|tsc|lint|typecheck)\b/i.test(command)
}

function commandSignalPassed(state: string | undefined, haystack: string): boolean {
  if (state !== 'ok') return false
  return !/(?:exit(?:ed)?(?:\s+with)?(?:\s+code)?\s+[1-9]\d*|\bfailed\b|\bfailure\b|\berror\b|not ok|\bdenied\b|\bawaiting\b|\bpending\b|\bcancelled\b|\bcanceled\b|\binterrupted\b|\baborted\b|timed?\s*out)/i.test(haystack)
}
